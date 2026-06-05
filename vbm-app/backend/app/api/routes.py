"""
routes.py — Endpoints de la API VBM
POST /analyze  → sube T1 + metadatos, crea job, retorna job_id
GET  /status/{job_id} → polling del estado
GET  /report/{job_id} → descarga reporte .txt
"""

import os
import uuid
import time
import asyncio
import traceback
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.config import TMP_DIR, API_CONFIG

# Flags de runtime (set en docker-compose.yml o `docker compose run -e`)
# KEEP_TMP_FILES=true → no borra /app/tmp/<job_id>/ al terminar,
#                        para poder inspeccionar mwp1*.nii y batches .m
#
# Nota: SKIP_ROBEX se retiró — ROBEX siempre se omite por defecto (decisión
# de diseño documentada en app/preprocessing/robex.py) para coincidir con el
# pipeline de entrenamiento.
def _truthy(env_value: str) -> bool:
    return (env_value or "").lower() in ("1", "true", "yes", "on")

KEEP_TMP_FILES = _truthy(os.environ.get("KEEP_TMP_FILES"))
from app.api.schemas import (
    ModelType, JobCreatedResponse, JobStatusResponse,
    ProcessStep, StepStatus, AnalysisResult,
)

router = APIRouter()

# ─── Store de jobs (memoria + persistencia ligera en disco) ──────────────────
# Estructura: { job_id: { "status": ..., "steps": [...], "result": ... } }
#
# Persistencia: se guarda un snapshot de "ID de job + status + paso final" a
# disco cada vez que se actualiza un paso. Si el container muere y reinicia,
# el endpoint /status puede devolver "error" en lugar de 404 → el frontend
# muestra un mensaje útil ("el backend reinició a mitad del análisis") en
# lugar del genérico "Job no encontrado".
import json as _json

JOBS: dict = {}
_PERSIST_FILE = TMP_DIR / "_jobs_snapshot.json"


def _persist_snapshot():
    """Guarda un snapshot serializable del dict JOBS — silencioso si falla."""
    try:
        snapshot = {}
        for jid, j in JOBS.items():
            snapshot[jid] = {
                "status":   j.get("status"),
                "progress": j.get("progress", 0),
                "error":    j.get("error"),
            }
        _PERSIST_FILE.write_text(_json.dumps(snapshot), encoding="utf-8")
    except Exception:
        pass


def _load_snapshot_on_start():
    """
    Al arrancar, lee jobs previos del disco. Si alguno quedó "running" o
    "pending" → lo marcamos como "error" porque el container reinició a
    mitad. El frontend lo verá como fallo en lugar de 404.
    """
    if not _PERSIST_FILE.exists():
        return
    try:
        prev = _json.loads(_PERSIST_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    for jid, snap in prev.items():
        status = snap.get("status")
        if status in ("running", "pending"):
            status = "error"
        JOBS[jid] = {
            "status":   status,
            "progress": snap.get("progress", 0),
            "steps":    [],
            "result":   None,
            "error":    snap.get("error") or (
                "El backend se reinició durante el análisis. "
                "Vuelve a lanzar el job."),
        }


# Cargar snapshot al importar el módulo (uvicorn lo importa una sola vez por proceso).
_load_snapshot_on_start()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _steps_for_model(model: ModelType) -> list[ProcessStep]:
    """Retorna la lista de pasos según el modelo seleccionado."""
    base = [
        ProcessStep(step=1, name="Cargando imagen T1",       status=StepStatus.PENDING),
        ProcessStep(step=2, name="Extracción de cráneo · ROBEX", status=StepStatus.PENDING),
        ProcessStep(step=3, name="Preprocesamiento VBM · deepmriprep", status=StepStatus.PENDING),
    ]
    if model == ModelType.DEEPMRIPREP:
        return base + [
            ProcessStep(step=4, name="Clasificación CNN",    status=StepStatus.PENDING),
        ]
    elif model == ModelType.NNUNET:
        return [
            ProcessStep(step=1, name="Cargando imagen T1",       status=StepStatus.PENDING),
            ProcessStep(step=2, name="Extracción de cráneo · ROBEX", status=StepStatus.PENDING),
            ProcessStep(step=3, name="Segmentación nnUNet",       status=StepStatus.PENDING),
            ProcessStep(step=4, name="Análisis de máscara",       status=StepStatus.PENDING),
        ]
    return base


def _update_step(job_id: str, step_num: int, status: StepStatus, msg: str = None):
    """Actualiza el estado de un paso específico."""
    for s in JOBS[job_id]["steps"]:
        if s.step == step_num:
            s.status  = status
            s.message = msg
            break
    completed = sum(1 for s in JOBS[job_id]["steps"] if s.status == StepStatus.COMPLETED)
    total     = len(JOBS[job_id]["steps"])
    JOBS[job_id]["progress"] = int((completed / total) * 100)
    _persist_snapshot()


# ─── Background task principal ────────────────────────────────────────────────
async def _run_analysis(job_id: str, nii_path: Path, model: ModelType,
                        metadata: dict):
    """
    Ejecuta el pipeline completo en background.
    Actualiza JOBS[job_id] en cada paso para que el polling lo vea.
    """
    start = time.time()
    JOBS[job_id]["status"] = "running"

    try:
        # ── Paso 1: Cargar imagen ──────────────────────────────────────────
        _update_step(job_id, 1, StepStatus.IN_PROGRESS, "Validando archivo NIfTI")
        from app.preprocessing.nifti_utils import validate_and_load
        t1_path = validate_and_load(nii_path)
        _update_step(job_id, 1, StepStatus.COMPLETED)

        # ── Paso 2: Skull stripping opcional (controlado por use_robex) ───
        # Por defecto (use_robex=False): la T1 pasa directamente a deepmriprep
        # — deepmriprep hace brain extraction internamente con deepbet.
        # Si use_robex=True: ROBEX se aplica antes (útil si el usuario tiene
        # una T1 con cráneo y prefiere skull-stripping clásico explícito).
        use_robex = bool(metadata.get("use_robex", False))
        _update_step(job_id, 2, StepStatus.IN_PROGRESS,
                     "Aplicando ROBEX..." if use_robex else "Validando imagen T1...")
        from app.preprocessing.robex import skull_strip
        brain_path = skull_strip(t1_path, job_id, skip_robex=not use_robex)
        _update_step(job_id, 2, StepStatus.COMPLETED,
                     "Skull stripping ROBEX aplicado" if use_robex
                     else "T1 pasa directamente a deepmriprep")

        # ── Pasos siguientes según modelo ──────────────────────────────────
        if model == ModelType.DEEPMRIPREP:
            from app.pipelines.deepmriprep_pipeline import run as run_dmp
            result = await asyncio.to_thread(run_dmp, brain_path, job_id,
                                             _update_step, metadata)

        elif model == ModelType.NNUNET:
            from app.pipelines.nnunet import run as run_nnunet
            result = await asyncio.to_thread(run_nnunet, brain_path, job_id,
                                             _update_step, metadata)

        else:
            raise ValueError(f"Modelo no soportado: {model}")

        # ── Resultado final ────────────────────────────────────────────────
        result.processing_time_s = round(time.time() - start, 1)
        result.test_name         = metadata.get("test_name", "")
        result.patient_name      = metadata.get("patient_name")
        result.notes             = metadata.get("notes")

        JOBS[job_id]["result"]   = result
        JOBS[job_id]["status"]   = "completed"
        JOBS[job_id]["progress"] = 100

        # Generar reporte .txt
        _generate_report(job_id, result)
        _persist_snapshot()

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = str(e)
        for s in JOBS[job_id]["steps"]:
            if s.status == StepStatus.IN_PROGRESS:
                s.status  = StepStatus.ERROR
                s.message = str(e)
        print(f"[JOB {job_id}] ERROR: {traceback.format_exc()}")
        _persist_snapshot()

    finally:
        # Limpiar archivos temporales del job (omitible con KEEP_TMP_FILES=true
        # para poder inspeccionar mwp1*.nii, rc1*.nii, batches .m, etc.)
        if KEEP_TMP_FILES:
            print(f"[JOB {job_id}] KEEP_TMP_FILES=true — intermediarios en {TMP_DIR / job_id}")
        else:
            _cleanup(job_id)


def _pct(value):
    """Formatea como porcentaje o '—' si es None."""
    return f"{value * 100:.1f}%" if value is not None else "—"


def _generate_report(job_id: str, result: AnalysisResult):
    """Genera reporte .txt descargable. Soporta clasificación y segmentación."""
    report_path = TMP_DIR / f"{job_id}_report.txt"
    lines = [
        "=" * 55,
        "  VBM App — Reporte de Análisis",
        "=" * 55,
        f"  Prueba      : {result.test_name}",
        f"  Paciente    : {result.patient_name or 'No especificado'}",
        f"  Modelo      : {result.model_used.value}",
        f"  Tiempo      : {result.processing_time_s} s",
        "-" * 55,
    ]

    # ── Bloque CLASIFICACIÓN (solo si hay predicción) ─────────────────────
    if result.prediction is not None:
        lines += [
            "  RESULTADO (clasificación)",
            f"  Predicción  : {'POSIBLE EPILEPSIA' if result.prediction == 'epilepsy' else 'CONTROL'}",
            f"  Confianza   : {_pct(result.confidence)}",
            f"  P(Epilepsia): {_pct(result.prob_epilepsy)}",
            f"  P(Control)  : {_pct(result.prob_control)}",
            "-" * 55,
            "  MÉTRICAS DEL MODELO (validación)",
            f"  AUC-ROC     : {_pct(result.model_auc)}",
            f"  Sensibilidad: {_pct(result.model_sensitivity)}",
            f"  Especificidad:{_pct(result.model_specificity)}",
            f"  Exactitud   : {_pct(result.model_accuracy)}",
        ]
        if result.gm_volume_cm3:
            lines += [
                "-" * 55,
                "  FEATURES VOLUMÉTRICOS",
                f"  Volumen GM  : {result.gm_volume_cm3:.2f} cm³",
                f"  Densidad GM : {result.gm_mean_density:.4f}",
                f"  Vóxeles GM  : {result.gm_voxels}",
            ]

    # ── Bloque SEGMENTACIÓN (solo si hay máscara) ─────────────────────────
    if result.mask_filename is not None:
        lines += [
            "  RESULTADO (segmentación nnU-Net)",
            f"  Vóxeles máscara    : {result.mask_voxels}",
            f"  Volumen total      : {result.mask_volume_cm3:.2f} cm³",
            f"  Clusters conexos   : {result.n_clusters}",
            f"  Cluster más grande : {result.largest_cluster_cm3:.2f} cm³",
            "-" * 55,
            "  MÉTRICAS DEL MODELO (evaluación · 778 sujetos)",
            f"  DSC (mediana)      : {_pct(result.model_dsc)}",
            f"  Hausdorff 95% (mediana): "
            + (f"{result.model_hd95:.2f} mm" if result.model_hd95 is not None else "—"),
            f"  Sensibilidad (sujeto): {_pct(result.model_seg_sensitivity)}",
            f"  Especificidad (sujeto): {_pct(result.model_seg_specificity)}",
            f"  VPP                : {_pct(result.model_seg_ppv)}",
            f"  VPN                : {_pct(result.model_seg_npv)}",
        ]

    if result.notes:
        lines += ["-" * 55, "  NOTAS CLÍNICAS", f"  {result.notes}"]
    lines += [
        "=" * 55,
        "  AVISO MÉDICO",
        "  Este análisis es herramienta de apoyo diagnóstico",
        "  y NO reemplaza el criterio clínico especializado.",
        "=" * 55,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    JOBS[job_id]["report_path"] = str(report_path)


def _cleanup(job_id: str):
    """
    Elimina archivos temporales del job.

    Excepción: jobs nnUNet — conservamos `t1.nii.gz` y `nnunet_out/<case>.nii.gz`
    porque el visor del frontend los consume después de que el job termine
    (via /t1/{job_id} y /mask/{job_id}). Borramos el resto (input crudo,
    intermediarios, _0000 duplicado).
    """
    job_tmp = TMP_DIR / job_id
    if not job_tmp.exists():
        return

    result = JOBS.get(job_id, {}).get("result")
    is_nnunet = result is not None and result.model_used == ModelType.NNUNET

    import shutil
    if not is_nnunet:
        try:
            shutil.rmtree(job_tmp)
        except Exception:
            pass
        return

    # nnUNet: borrar todo excepto el T1 servido y la máscara generada.
    keep = {
        job_tmp / (result.t1_filename or ""),
        job_tmp / "nnunet_out" / (result.mask_filename or ""),
    }
    for path in job_tmp.rglob("*"):
        if path.is_dir():
            continue
        if path in keep:
            continue
        try:
            path.unlink()
        except Exception:
            pass
    # Eliminar carpetas vacías (deja nnunet_out si retiene la máscara)
    for sub in sorted(job_tmp.rglob("*"), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()
            except OSError:
                pass


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=JobCreatedResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    file:         UploadFile = File(...),
    test_name:    str        = Form(...),
    model:        ModelType  = Form(...),
    patient_name: str        = Form(""),
    notes:        str        = Form(""),
    use_robex:    bool       = Form(False),
):
    # Validar extensión
    filename = file.filename or ""
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos .nii o .nii.gz"
        )

    # Validar tamaño
    max_bytes = API_CONFIG["max_upload_mb"] * 1024 * 1024
    content   = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande (máx {API_CONFIG['max_upload_mb']} MB)"
        )

    # Guardar en tmp
    job_id  = str(uuid.uuid4())[:8]
    job_dir = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    nii_path = job_dir / filename
    nii_path.write_bytes(content)

    # Registrar job
    JOBS[job_id] = {
        "status":   "pending",
        "progress": 0,
        "steps":    _steps_for_model(model),
        "result":   None,
        "error":    None,
    }

    metadata = {
        "test_name":    test_name,
        "patient_name": patient_name or None,
        "notes":        notes or None,
        "use_robex":    use_robex,
    }

    background_tasks.add_task(_run_analysis, job_id, nii_path, model, metadata)

    return JobCreatedResponse(job_id=job_id)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    j = JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=j["status"],
        progress=j["progress"],
        steps=j["steps"],
        result=j.get("result"),
        error=j.get("error"),
    )


@router.get("/report/{job_id}")
def download_report(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    report_path = JOBS[job_id].get("report_path")
    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Reporte aún no disponible")
    return FileResponse(
        path=report_path,
        filename=f"vbm_reporte_{job_id}.txt",
        media_type="text/plain",
    )


# ─── Endpoints del visor (nnUNet) ─────────────────────────────────────────────
# El frontend (NiiVue) consume estos dos archivos para renderizar el T1 +
# overlay de la máscara segmentada.

def _resolve_job_file(job_id: str, kind: str) -> Path:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    result = JOBS[job_id].get("result")
    if result is None:
        raise HTTPException(status_code=404, detail="Job sin resultado disponible")
    if result.model_used != ModelType.NNUNET:
        raise HTTPException(
            status_code=400,
            detail=f"El modelo {result.model_used.value} no genera archivos para el visor",
        )

    job_dir = TMP_DIR / job_id
    if kind == "t1":
        fname = result.t1_filename
        path  = job_dir / (fname or "")
    elif kind == "mask":
        fname = result.mask_filename
        path  = job_dir / "nnunet_out" / (fname or "")
    else:
        raise HTTPException(status_code=400, detail=f"Kind inválido: {kind}")

    if not fname or not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Archivo {kind} no encontrado para el job {job_id}",
        )
    return path


@router.get("/t1/{job_id}")
def get_t1(job_id: str):
    """T1 servido al visor NiiVue (después del job nnUNet)."""
    path = _resolve_job_file(job_id, "t1")
    return FileResponse(
        path=path,
        filename=f"t1_{job_id}.nii.gz",
        media_type="application/gzip",
    )


@router.get("/mask/{job_id}")
def get_mask(job_id: str):
    """Máscara binaria producida por nnUNet (overlay en el visor)."""
    path = _resolve_job_file(job_id, "mask")
    return FileResponse(
        path=path,
        filename=f"mask_{job_id}.nii.gz",
        media_type="application/gzip",
    )