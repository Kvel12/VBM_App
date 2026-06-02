"""
hybrid.py — Pipeline Modelo Híbrido (SPM12/DARTEL + SVM + fusión lineal)

Extiende el pipeline SPM12 con clasificación SVM sobre features volumétricos
y fusiona ambas predicciones con peso w_CNN=0.80.

Pasos adicionales respecto a spm12_dartel.py:
  4. Features volumétricos desde el mapa GM (CSV-compatible)
  5. Clasificación SVM sobre features escalares
  6. Ensamblaje lineal CNN + SVM
"""

from pathlib import Path
from typing import Callable

from app.config import TMP_DIR
from app.api.schemas import AnalysisResult, ModelType, StepStatus
from app.spm.spm_runner import run_segmentation, run_dartel_existing_full
from app.classifiers.cnn_model import predict as cnn_predict
from app.classifiers.svm_model import predict as svm_predict, SVM_MODEL_METRICS
from app.classifiers.ensemble import fuse, ENSEMBLE_METRICS
from app.preprocessing.nifti_utils import extract_volumetric_features


def run(brain_path: Path, job_id: str,
        update_step: Callable, metadata: dict) -> AnalysisResult:
    """
    Pipeline híbrido completo: SPM12 → CNN + SVM → ensamblaje.

    Steps del frontend (definidos en routes._steps_for_model HYBRID):
      1. Cargando imagen T1         (routes.py)
      2. Extracción de cráneo ROBEX (routes.py)
      3. Mapa de materia gris SPM12  ← aquí
      4. Features volumétricos (CSV) ← aquí
      5. Clasificación SVM           ← aquí
      6. Ensamblaje SPM12 + SVM      ← aquí
    """
    brain_path = Path(brain_path)
    job_dir    = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # ── Paso 3: SPM12 Segmentación + DARTEL + Normalización ──────────────────
    update_step(job_id, 3, StepStatus.IN_PROGRESS,
                "Segmentación SPM12 + DARTEL existing + normalización MNI...")
    try:
        rc1_path, rc2_path, _ = run_segmentation(brain_path, job_id)
        gm_map_path, ff_path  = run_dartel_existing_full(rc1_path, rc2_path, job_id)
    except Exception as e:
        update_step(job_id, 3, StepStatus.ERROR, str(e))
        raise RuntimeError(f"Pipeline SPM12 falló: {e}") from e

    update_step(job_id, 3, StepStatus.COMPLETED,
                f"mapa GM: {gm_map_path.name}")

    # ── Paso 4: Features volumétricos ────────────────────────────────────────
    update_step(job_id, 4, StepStatus.IN_PROGRESS,
                "Extrayendo features volumétricos del mapa GM...")
    try:
        vol_features = extract_volumetric_features(gm_map_path)
    except Exception as e:
        update_step(job_id, 4, StepStatus.ERROR, str(e))
        raise RuntimeError(f"Extracción de features falló: {e}") from e

    update_step(job_id, 4, StepStatus.COMPLETED,
                f"GM: {vol_features.get('gm_volume_cm3', '?'):.1f} cm³")

    # ── Paso 5: CNN + SVM en paralelo (secuencial por CPU) ───────────────────
    update_step(job_id, 5, StepStatus.IN_PROGRESS,
                "Clasificación SVM + CNN...")
    try:
        cnn_result = cnn_predict(gm_map_path)
        svm_result = svm_predict(vol_features)
    except Exception as e:
        update_step(job_id, 5, StepStatus.ERROR, str(e))
        raise RuntimeError(f"Clasificación falló: {e}") from e

    update_step(job_id, 5, StepStatus.COMPLETED,
                f"CNN: {cnn_result['prob_epilepsy']:.2f} | "
                f"SVM: {svm_result['prob_epilepsy']:.2f}")

    # ── Paso 6: Ensamblaje ────────────────────────────────────────────────────
    update_step(job_id, 6, StepStatus.IN_PROGRESS,
                f"Fusión lineal (CNN×{ENSEMBLE_METRICS['weight_cnn']} "
                f"+ SVM×{ENSEMBLE_METRICS['weight_svm']})...")
    try:
        ensemble_result = fuse(cnn_result, svm_result)
    except Exception as e:
        update_step(job_id, 6, StepStatus.ERROR, str(e))
        raise RuntimeError(f"Ensamblaje falló: {e}") from e

    update_step(job_id, 6, StepStatus.COMPLETED,
                f"P(epilepsia)={ensemble_result['prob_epilepsy']:.2f}")

    # ── Resultado final ───────────────────────────────────────────────────────
    result = AnalysisResult(
        prediction        = ensemble_result["prediction"],
        confidence        = ensemble_result["confidence"],
        prob_epilepsy     = ensemble_result["prob_epilepsy"],
        prob_control      = ensemble_result["prob_control"],
        model_auc         = ENSEMBLE_METRICS["auc"],
        model_sensitivity = ENSEMBLE_METRICS["sensitivity"],
        model_specificity = ENSEMBLE_METRICS["specificity"],
        model_accuracy    = ENSEMBLE_METRICS["accuracy"],
        gm_volume_cm3     = vol_features.get("gm_volume_cm3"),
        gm_mean_density   = vol_features.get("gm_mean_density"),
        gm_voxels         = vol_features.get("gm_voxels"),
        model_used        = ModelType.HYBRID,
        test_name         = metadata.get("test_name", ""),
        patient_name      = metadata.get("patient_name"),
        notes             = metadata.get("notes"),
    )

    return result