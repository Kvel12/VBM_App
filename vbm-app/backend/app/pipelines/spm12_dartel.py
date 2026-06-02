"""
spm12_dartel.py — Pipeline SPM12/DARTEL + CNN para inferencia

Flujo (steps del frontend ModelType.SPM12_DARTEL):
  1. Cargando imagen T1         ← routes.py
  2. Extracción de cráneo ROBEX ← routes.py
  3. Mapa de materia gris SPM12 ← este módulo (seg + DARTEL + norm)
  4. Clasificación CNN          ← este módulo
"""

from pathlib import Path
from typing import Callable

from app.config import TMP_DIR
from app.api.schemas import AnalysisResult, ModelType, StepStatus
from app.spm.spm_runner import run_segmentation, run_dartel_existing_full
from app.classifiers.cnn_model import predict as cnn_predict, CNN_MODEL_METRICS
from app.preprocessing.nifti_utils import extract_volumetric_features


def run(brain_path: Path, job_id: str,
        update_step: Callable, metadata: dict) -> AnalysisResult:
    """
    Pipeline SPM12/DARTEL + CNN.

    Args:
        brain_path:  .nii sin cráneo (salida de ROBEX o T1 cruda si fallback).
        job_id:      ID del job.
        update_step: callback(job_id, step_num, StepStatus, msg=None)
        metadata:    dict con test_name, patient_name, notes.
    """
    brain_path = Path(brain_path)
    job_dir    = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ── Paso 3: SPM12 Segmentación + DARTEL existing + Normalización MNI ──────
    update_step(job_id, 3, StepStatus.IN_PROGRESS,
                "Segmentación unificada SPM12...")
    try:
        rc1_path, rc2_path, _ = run_segmentation(brain_path, job_id)
    except Exception as e:
        update_step(job_id, 3, StepStatus.ERROR, str(e))
        raise RuntimeError(f"Segmentación SPM12 falló: {e}") from e

    update_step(job_id, 3, StepStatus.IN_PROGRESS,
                "DARTEL existing templates + normalización MNI...")
    try:
        gm_map_path, _ = run_dartel_existing_full(rc1_path, rc2_path, job_id)
    except Exception as e:
        update_step(job_id, 3, StepStatus.ERROR, str(e))
        raise RuntimeError(f"DARTEL normalización falló: {e}") from e

    update_step(job_id, 3, StepStatus.COMPLETED,
                f"mapa GM: {gm_map_path.name}")

    # Features volumétricos para el reporte (no afectan la predicción CNN)
    try:
        vol_features = extract_volumetric_features(gm_map_path)
    except Exception:
        vol_features = {}

    # ── Paso 4: CNN predicción ────────────────────────────────────────────────
    update_step(job_id, 4, StepStatus.IN_PROGRESS,
                "Clasificación CNN (MedicalNet ResNet-18)...")
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
        # Métricas del fold 0 (mejor fold, representativo para la app)
        model_auc         = CNN_MODEL_METRICS["auc"],
        model_sensitivity = CNN_MODEL_METRICS["sensitivity"],
        model_specificity = CNN_MODEL_METRICS["specificity"],
        model_accuracy    = CNN_MODEL_METRICS["accuracy"],
        gm_volume_cm3     = vol_features.get("gm_volume_cm3"),
        gm_mean_density   = vol_features.get("gm_mean_density"),
        gm_voxels         = vol_features.get("gm_voxels"),
        model_used        = ModelType.SPM12_DARTEL,
        test_name         = metadata.get("test_name", ""),
        patient_name      = metadata.get("patient_name"),
        notes             = metadata.get("notes"),
    )