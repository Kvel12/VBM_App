"""
nnunet.py — Pipeline de segmentación 3D-nnU-Net para localización de zonas
            epileptogénicas.

Modelo: 3d_fullres, nnUNetTrainer_250epochs, fold=all, transfer learning
desde el checkpoint TBI de Castaño (2025). Entrenado sobre Dataset500
(IDEAS_Epilepsy) con máscaras de resección + controles con máscaras vacías.

ARQUITECTURA del modelo:
  - PlainConvUNet 3D, 6 stages, features [32, 64, 128, 256, 320, 320]
  - patch [128, 128, 128], batch 2, spacing [1.05, 1.0, 1.0]
  - Normalización: ZScoreNormalization
  - Output: máscara binaria (0=fondo, 1=zona epileptogénica putativa)

FLUJO en el backend (ModelType.NNUNET):
  1. Cargando imagen T1            ← routes.py
  2. (Opcional) ROBEX skull strip  ← routes.py (toggle use_robex)
  3. Segmentación nnUNet           ← este módulo (inferencia 3D)
  4. Análisis de máscara           ← este módulo (clusters, volumen)

PRESTACIONES:
  - GPU NVIDIA: <60s por sujeto
  - CPU: ~15-30 min por sujeto (aceptable para la app desplegable)
"""

import time
import shutil
import gzip
from pathlib import Path
from typing import Callable

import numpy as np
import nibabel as nib
import torch

from app.config import (
    TMP_DIR, DEVICE,
    NNUNET_MODEL_DIR, NNUNET_FOLD, NNUNET_CHECKPOINT,
    NNUNET_METRICS,
)
from app.api.schemas import AnalysisResult, ModelType, StepStatus


# ─── Singleton del predictor ──────────────────────────────────────────────────
# nnU-Net carga ~235 MB de pesos + planes — cachearlo entre jobs ahorra mucho.
_predictor = None


def _setup_model_folder() -> Path:
    """
    nnU-Net espera una estructura específica al llamar
    initialize_from_trained_model_folder():

        <model_training_output_dir>/
          fold_all/checkpoint_best.pth
          dataset.json
          plans.json
          dataset_fingerprint.json

    El usuario subió los 4 archivos planos en NNUNET_MODEL_DIR. Aquí armamos
    la estructura esperada con symlinks (o copias si symlink falla) en
    /app/tmp/_nnunet_setup/.
    """
    setup_dir = TMP_DIR / "_nnunet_setup"
    fold_dir  = setup_dir / f"fold_{NNUNET_FOLD}"

    setup_dir.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        (NNUNET_MODEL_DIR / NNUNET_CHECKPOINT,  fold_dir / NNUNET_CHECKPOINT),
        (NNUNET_MODEL_DIR / "plans.json",       setup_dir / "plans.json"),
        (NNUNET_MODEL_DIR / "dataset.json",     setup_dir / "dataset.json"),
        (NNUNET_MODEL_DIR / "dataset_fingerprint.json",
                                                setup_dir / "dataset_fingerprint.json"),
    ]
    for src, dst in pairs:
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            raise FileNotFoundError(f"Archivo nnUNet no encontrado: {src}")
        try:
            dst.symlink_to(src)
        except (OSError, NotImplementedError):
            shutil.copy2(str(src), str(dst))

    return setup_dir


def get_predictor():
    """
    Carga el nnUNetPredictor singleton. Primera llamada toma ~30-60s
    (carga pesos + construye red). Siguientes llamadas reusan el predictor.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    # Import perezoso — nnunetv2 es pesado.
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    setup_dir = _setup_model_folder()

    print(f"[nnUNet] Cargando predictor desde {setup_dir} → {DEVICE.upper()}")
    t0 = time.time()

    use_gpu = torch.cuda.is_available()
    # Configuración para inferencia en CPU con poca RAM (~2-4 GB Docker):
    #  - tile_step_size=0.7  → menos parches solapados (era 0.5)
    #  - use_mirroring=False → desactiva TTA con flips (4× menos memoria)
    # Pérdida esperada en DSC: ~1-2 pts. Aceptable para una app interactiva
    # frente al beneficio de no morir por OOM.
    predictor = nnUNetPredictor(
        tile_step_size=0.7,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=use_gpu,
        device=torch.device(DEVICE),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(setup_dir),
        use_folds=(NNUNET_FOLD,),
        checkpoint_name=NNUNET_CHECKPOINT,
    )

    print(f"[nnUNet] Predictor cargado OK en {time.time() - t0:.1f}s "
          f"(device={DEVICE}, GPU={'sí' if use_gpu else 'no'})")
    _predictor = predictor
    return predictor


# ─── Pipeline público ────────────────────────────────────────────────────────

def run(brain_path: Path, job_id: str,
        update_step: Callable, metadata: dict) -> AnalysisResult:
    """
    Pipeline nnUNet completo.

    Args:
        brain_path:  T1 (cruda o post-ROBEX según toggle use_robex).
        job_id:      ID del job.
        update_step: callback(job_id, step_num, StepStatus, msg=None)
        metadata:    dict con test_name, patient_name, notes.
    """
    brain_path = Path(brain_path)
    job_dir    = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ── Paso 3: Inferencia nnUNet ────────────────────────────────────────────
    update_step(job_id, 3, StepStatus.IN_PROGRESS,
                f"Segmentación 3D nnU-Net ({'GPU' if torch.cuda.is_available() else 'CPU'})...")
    try:
        mask_path = _run_nnunet_inference(brain_path, job_dir)
    except Exception as e:
        update_step(job_id, 3, StepStatus.ERROR, str(e))
        raise RuntimeError(f"nnU-Net segmentación falló: {e}") from e

    update_step(job_id, 3, StepStatus.COMPLETED,
                f"máscara: {mask_path.name}")

    # ── Paso 4: Análisis de la máscara ──────────────────────────────────────
    update_step(job_id, 4, StepStatus.IN_PROGRESS,
                "Análisis de clusters y volumetría...")
    try:
        mask_stats = _analyze_mask(mask_path)
    except Exception as e:
        update_step(job_id, 4, StepStatus.ERROR, str(e))
        raise RuntimeError(f"Análisis de máscara falló: {e}") from e

    update_step(job_id, 4, StepStatus.COMPLETED,
                f"{mask_stats['n_clusters']} cluster(s), "
                f"{mask_stats['mask_volume_cm3']:.2f} cm³ totales")

    # ── Resultado ────────────────────────────────────────────────────────────
    # Para que /t1/{job_id} pueda servir el archivo, conservamos el T1 como
    # `t1.nii.gz` en job_dir. Si vino sin .gz lo comprimimos al vuelo para
    # que NiiVue lo cargue uniforme.
    t1_dst_name = "t1.nii.gz"
    t1_dst      = job_dir / t1_dst_name
    if not t1_dst.exists():
        _copy_as_nii_gz(brain_path, t1_dst)

    # ── Clasificación derivada de la máscara ──────────────────────────────
    # Regla del notebook de entrenamiento: cualquier voxel segmentado cuenta
    # como "epilepsia detectada" (tasa de detección 0.9051). Una máscara
    # vacía clasifica como "control" (especificidad 0.7890). NO usamos un
    # umbral de volumen mínimo — eso degradaría la sensibilidad reportada.
    is_epi      = mask_stats["mask_voxels"] > 0
    prediction  = "epilepsy" if is_epi else "control"

    # "Confianza" = sensibilidad/especificidad del modelo a nivel sujeto para
    # la clase predicha. No es una probabilidad bayesiana real (nnUNet no la
    # produce); es la métrica más honesta que podemos asignar sin invertarla.
    confidence  = (NNUNET_METRICS["sensitivity"] if is_epi
                   else NNUNET_METRICS["specificity"])
    prob_epi    = NNUNET_METRICS["sensitivity"] if is_epi else (1 - NNUNET_METRICS["specificity"])
    prob_ctrl   = 1 - prob_epi

    return AnalysisResult(
        # Predicción derivada
        prediction          = prediction,
        confidence          = confidence,
        prob_epilepsy       = prob_epi,
        prob_control        = prob_ctrl,
        # Segmentación
        mask_filename       = mask_path.name,
        t1_filename         = t1_dst_name,
        mask_voxels         = mask_stats["mask_voxels"],
        mask_volume_cm3     = mask_stats["mask_volume_cm3"],
        n_clusters          = mask_stats["n_clusters"],
        largest_cluster_cm3 = mask_stats["largest_cluster_cm3"],
        model_dsc               = NNUNET_METRICS.get("dsc_mean"),
        model_hd95              = NNUNET_METRICS.get("hausdorff_95"),
        model_seg_sensitivity   = NNUNET_METRICS.get("sensitivity"),
        model_seg_specificity   = NNUNET_METRICS.get("specificity"),
        model_seg_ppv           = NNUNET_METRICS.get("ppv"),
        model_seg_npv           = NNUNET_METRICS.get("npv"),
        model_used              = ModelType.NNUNET,
        test_name           = metadata.get("test_name", ""),
        patient_name        = metadata.get("patient_name"),
        notes               = metadata.get("notes"),
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_nnunet_inference(t1_path: Path, job_dir: Path) -> Path:
    """
    Llama al predictor con un solo T1 y devuelve el path a la máscara generada.

    nnU-Net espera input como lista de listas (multi-channel support):
        [[t1_path]]   — single channel T1
    Devuelve un .nii.gz con la máscara binaria (0/1).
    """
    predictor = get_predictor()

    out_dir = job_dir / "nnunet_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Renombrar el T1 al formato nnUNet (<case>_0000.nii.gz) para que la
    # salida tenga un nombre limpio (<case>.nii.gz).
    case_id     = "subject"
    case_input  = out_dir / f"{case_id}_0000.nii.gz"
    if not case_input.exists():
        _copy_as_nii_gz(t1_path, case_input)

    expected_output = out_dir / f"{case_id}.nii.gz"

    print(f"[nnUNet] Inferencia sobre {t1_path.name} → {expected_output.name}")
    t0 = time.time()

    # ── Camino "low-level" sin multiprocessing ─────────────────────────────
    # predict_from_files (con o sin _sequential) usa internamente workers de
    # multiprocessing para preprocess/export. Cuando uno de esos workers
    # llama os._exit(0) (caso típico tras exportar), termina TODO el proceso
    # padre → uvicorn muere → JOBS pierde el job → 404.
    #
    # predict_single_npy_array es la API in-process: le paso un np.ndarray,
    # me devuelve un np.ndarray. Yo me encargo de la I/O. CERO multiprocessing.
    img      = nib.load(str(case_input))
    data     = np.asarray(img.dataobj, dtype=np.float32)
    affine   = img.affine
    # nnUNet espera shape (C, X, Y, Z) — añadimos canal
    input_4d = data[None, :, :, :]
    # `spacing`: tamaño del voxel en mm (X, Y, Z) — derivado del affine
    voxel_sz = tuple(float(s) for s in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))
    image_props = {"spacing": voxel_sz}

    seg = predictor.predict_single_npy_array(
        input_image       = input_4d,
        image_properties  = image_props,
        segmentation_previous_stage = None,
        output_file_truncated       = None,   # devuelve el array, no escribe
        save_or_return_probabilities= False,
    )
    # seg es un np.ndarray (X, Y, Z) con la máscara binaria
    seg_img = nib.Nifti1Image(seg.astype(np.uint8), affine, header=img.header)
    seg_img.set_data_dtype(np.uint8)
    nib.save(seg_img, str(expected_output))

    elapsed = time.time() - t0
    print(f"[nnUNet] Inferencia completada en {elapsed:.1f}s "
          f"(camino in-process, sin workers)")

    if not expected_output.exists():
        raise RuntimeError(
            f"nnUNet no generó la máscara esperada {expected_output}."
        )
    return expected_output


def _analyze_mask(mask_path: Path) -> dict:
    """
    Análisis post-segmentación: conteo de voxeles, volumen, clusters conexos.
    """
    img  = nib.load(str(mask_path))
    data = np.asarray(img.dataobj, dtype=np.uint8)

    # Voxel volume en mm³
    vox     = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    vox_vol = float(np.prod(vox))   # mm³ por voxel

    mask    = data > 0
    n_vox   = int(mask.sum())
    vol_cm3 = (n_vox * vox_vol) / 1000.0

    n_clusters          = 0
    largest_cluster_cm3 = 0.0
    if n_vox > 0:
        try:
            from scipy.ndimage import label
            structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
            labeled, n_clusters = label(mask, structure=structure)
            if n_clusters > 0:
                sizes = np.bincount(labeled.ravel())[1:]  # ignora fondo
                largest_cluster_cm3 = round(float(sizes.max() * vox_vol / 1000.0), 4)
        except Exception as e:
            print(f"[nnUNet] Análisis de clusters falló (no crítico): {e}")
            n_clusters = -1

    return {
        "mask_voxels":         n_vox,
        "mask_volume_cm3":     round(vol_cm3, 4),
        "n_clusters":          int(n_clusters),
        "largest_cluster_cm3": largest_cluster_cm3,
    }


def _copy_as_nii_gz(src: Path, dst: Path) -> None:
    """
    Copia un .nii o .nii.gz a destino .nii.gz (comprime si es necesario).
    Idempotente: si dst ya existe, no hace nada.
    """
    if dst.exists():
        return
    src = Path(src)
    if src.suffix == ".gz":
        shutil.copy2(str(src), str(dst))
        return
    with open(src, "rb") as f_in, gzip.open(str(dst), "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
