"""
deepmriprep_pipeline.py — Pipeline VBM con deepmriprep + CNN para inferencia

Reemplaza al antiguo SPM12/DARTEL (descartado por irreproducibilidad cross-platform).
deepmriprep es un toolkit Python puro basado en PyTorch que hace:
  - Brain extraction (deepbet)
  - Tissue segmentation (3D UNet)
  - Spatial registration to MNI152 (sSYMNet)
  - Modulation (preserva volumen con jacobiano)

Output: mwp1*.nii.gz — mapa de materia gris modulado en espacio MNI152
        (mismo formato que producía SPM12, pero reproducible bit-a-bit
         entre Linux/macOS/Windows porque toda la cadena son modelos
         PyTorch deterministas).

Flujo (steps del frontend ModelType.DEEPMRIPREP):
  1. Cargando imagen T1            ← routes.py
  2. (Opcional) ROBEX skull strip  ← routes.py (toggle use_robex)
  3. Preprocesamiento VBM          ← este módulo (deepmriprep)
  4. Clasificación CNN             ← este módulo
"""

import torch
from pathlib import Path
from typing import Callable

from app.config import TMP_DIR
from app.api.schemas import AnalysisResult, ModelType, StepStatus
from app.classifiers.cnn_model import predict as cnn_predict, CNN_MODEL_METRICS
from app.preprocessing.nifti_utils import extract_volumetric_features


def run(brain_path: Path, job_id: str,
        update_step: Callable, metadata: dict) -> AnalysisResult:
    """
    Pipeline deepmriprep + CNN.

    Args:
        brain_path:  T1 (cruda o post-ROBEX según toggle use_robex).
        job_id:      ID del job.
        update_step: callback(job_id, step_num, StepStatus, msg=None)
        metadata:    dict con test_name, patient_name, notes, use_robex.
    """
    brain_path = Path(brain_path)
    job_dir    = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ── Paso 3: deepmriprep — brain extraction + seg + MNI + modulación ──────
    update_step(job_id, 3, StepStatus.IN_PROGRESS,
                "Preprocesamiento VBM con deepmriprep (deepbet + UNet + sSYMNet)...")
    try:
        gm_map_path = _run_deepmriprep(brain_path, job_dir)
    except Exception as e:
        update_step(job_id, 3, StepStatus.ERROR, str(e))
        raise RuntimeError(f"deepmriprep falló: {e}") from e

    update_step(job_id, 3, StepStatus.COMPLETED,
                f"mapa GM: {gm_map_path.name}")

    # Features volumétricos para el reporte (no afectan la predicción CNN)
    try:
        vol_features = extract_volumetric_features(gm_map_path)
    except Exception:
        vol_features = {}

    # ── Paso 4: CNN predicción ────────────────────────────────────────────────
    update_step(job_id, 4, StepStatus.IN_PROGRESS,
                "Clasificación CNN (MedicalNet ResNet-18 fold 3)...")
    try:
        cnn_result = cnn_predict(gm_map_path)
    except Exception as e:
        update_step(job_id, 4, StepStatus.ERROR, str(e))
        raise RuntimeError(f"Inferencia CNN falló: {e}") from e

    update_step(job_id, 4, StepStatus.COMPLETED,
                f"P(epilepsia)={cnn_result['prob_epilepsy']:.2f} "
                f"[umbral={cnn_result['threshold_used']}]")

    # ── Resultado ─────────────────────────────────────────────────────────────
    return AnalysisResult(
        prediction        = cnn_result["prediction"],
        confidence        = cnn_result["confidence"],
        prob_epilepsy     = cnn_result["prob_epilepsy"],
        prob_control      = cnn_result["prob_control"],
        # Métricas del fold 3 (mejor fold, representativo para la app)
        model_auc         = CNN_MODEL_METRICS["auc"],
        model_sensitivity = CNN_MODEL_METRICS["sensitivity"],
        model_specificity = CNN_MODEL_METRICS["specificity"],
        model_accuracy    = CNN_MODEL_METRICS["accuracy"],
        gm_volume_cm3     = vol_features.get("gm_volume_cm3"),
        gm_mean_density   = vol_features.get("gm_mean_density"),
        gm_voxels         = vol_features.get("gm_voxels"),
        model_used        = ModelType.DEEPMRIPREP,
        test_name         = metadata.get("test_name", ""),
        patient_name      = metadata.get("patient_name"),
        notes             = metadata.get("notes"),
    )


# ─── Wrapper de deepmriprep ───────────────────────────────────────────────────

def _run_deepmriprep(t1_path: Path, job_dir: Path) -> Path:
    """
    Ejecuta el pipeline deepmriprep con outputs='vbm' (produce mwp1*.nii.gz).
    Auto-detecta GPU vía torch.cuda.is_available().

    Returns:
        Path al archivo mwp1*.nii.gz generado.
    """
    # Import perezoso — deepmriprep es pesado (carga modelos al importar)
    from deepmriprep import run_preprocess

    out_dir = job_dir / "deepmriprep_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    no_gpu = not torch.cuda.is_available()
    print(f"[deepmriprep] Iniciando: {t1_path.name} (GPU={'sí' if not no_gpu else 'no'})")

    run_preprocess(
        input_paths      = [str(t1_path)],
        output_dir       = str(out_dir),
        outputs          = 'vbm',
        dir_format       = 'sub',         # crea subcarpeta por sujeto
        no_gpu           = no_gpu,
        skip_broken      = True,
        skip_unprocessed = True,
    )

    # Buscar el mwp1*.nii.gz recursivamente (compat con dir_format='sub'/'flat')
    candidates = list(out_dir.rglob('mwp1*.nii.gz'))
    if not candidates:
        # Fallback: mwp1 sin .gz (algunas versiones de deepmriprep)
        candidates = list(out_dir.rglob('mwp1*.nii'))
    if not candidates:
        raise RuntimeError(
            f"deepmriprep terminó pero no se encontró mwp1*.nii(.gz) en {out_dir}.\n"
            f"Archivos generados: {[p.name for p in out_dir.rglob('*.nii*')]}"
        )

    return candidates[0]
