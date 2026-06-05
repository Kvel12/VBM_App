"""
config.py — Configuración central del backend VBM
Todas las rutas, parámetros y constantes del sistema se definen aquí.

Pipeline actual: deepmriprep (PyTorch puro) → CNN MedicalNet ResNet-18.
ROBEX se mantiene como skull-stripping OPCIONAL antes de deepmriprep
(toggle del frontend `use_robex` — por defecto OFF, deepmriprep hace
brain extraction internamente con deepbet).
"""

import os
import torch
from pathlib import Path

# ─── Directorios base ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
TMP_DIR    = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# ─── ROBEX (skull stripping opcional) ─────────────────────────────────────────
ROBEX_DIR    = Path(os.environ.get("ROBEX_DIR", "/opt/ROBEX"))
ROBEX_SCRIPT = ROBEX_DIR / "runROBEX.sh"

# ─── Modelo CNN (entrenado sobre mapas mwp1 de deepmriprep) ──────────────────
# El .pt es TorchScript serializado con torch.jit.SCRIPT (NO trace).
# Fold 3 = mejor fold individual (AUC 0.7968 — el más alto de los 5)
#
# HISTORIA: la primera versión usaba torch.jit.trace, que introducía un sesgo
# sistemático de ~0.062 en P(epi) por interactuar mal con `functools.partial`
# en los bloques downsample. Como el umbral clínico es 0.6875, el sesgo
# convertía el 50% de las predicciones de epilepsia en "control" (P(epi) caía
# de >0.69 a ~0.63). torch.jit.script analiza el código estáticamente y
# reproduce la inferencia 1:1 con el modelo original (Δ = 0 en 30 sujetos test).
CNN_MODEL_PATH = MODELS_DIR / "deepmriprep_cnn_fold3_script.pt"

# ─── Parámetros VBM (usados por nifti_utils.extract_volumetric_features) ────
# `gm_threshold`: umbral estándar para la máscara de GM sobre mwp1 modulado
#                 (mismo valor que el notebook de entrenamiento).
# `expected_shape`: shape MNI152 producido por deepmriprep para tag de QC.
VBM_PARAMS = {
    "gm_threshold":   0.1,
    "expected_shape": (113, 137, 113),   # deepmriprep MNI152 — distinto del (121,145,121) que producía SPM12
}

# ─── nnU-Net (segmentación de zonas epileptogénicas) ────────────────────────
# El modelo se entrenó en Colab con Dataset500_IDEAS_Epilepsy,
# nnUNetTrainer_250epochs, 3d_fullres, fold='all', transfer learning desde
# checkpoint de Castaño (modelo TBI). Archivos esperados en NNUNET_MODEL_DIR:
#   - checkpoint_best.pth   (~235 MB)
#   - plans.json
#   - dataset.json
#   - dataset_fingerprint.json
# nnUNet espera una estructura de carpetas específica al llamar
# initialize_from_trained_model_folder — la armamos en runtime con symlinks.
NNUNET_MODEL_DIR    = MODELS_DIR / "nnunet_ideas_fold_all"
NNUNET_DATASET_NAME = "Dataset500_IDEAS_Epilepsy"
NNUNET_TRAINER      = "nnUNetTrainer_250epochs"
NNUNET_PLANS_NAME   = "nnUNetPlans"
NNUNET_CONFIG       = "3d_fullres"
NNUNET_FOLD         = "all"
NNUNET_CHECKPOINT   = "checkpoint_best.pth"

# Métricas reportadas — evaluación sobre 778 sujetos (432 pacientes + 346
# controles) del dataset IDEAS_Epilepsy.
#
# DSC y HD95: se usan los valores MEDIANOS, no la media. La distribución es
# bimodal — el modelo o segmenta bien (DSC > 0.8) o falla por completo (DSC ≈ 0)
# por ausencia de la lesión en la T1. La mediana refleja el comportamiento típico.
#   DSC medio:    0.6320 ± 0.3325  (arrastrado por outliers como sub-18 = 0.0001)
#   DSC mediano:  0.8248           ← reportado
#   HD95 medio:   38.96 ± 40.21 mm
#   HD95 mediano: 9.37 mm          ← reportado
#
# Sensibilidad / Especificidad: a NIVEL SUJETO (tasa de detección y tasa de
# controles sin falsos positivos), que es lo clínicamente útil. La sensibilidad
# por vóxel (0.699) y la precisión por vóxel (0.618) las dejamos como referencia.
NNUNET_METRICS = {
    "dsc_mean":           0.8248,  # mediana sobre 432 pacientes
    "hausdorff_95":       9.37,    # mm — mediana
    "sensitivity":        0.9051,  # tasa de detección sujeto (391/432)
    "specificity":        0.7890,  # controles sin FP (273/346)
    "voxel_sensitivity":  0.6990,
    "voxel_precision":    0.6175,
    "cluster_f1":         0.4135,
    "ppv":                0.8427,
    "npv":                0.8694,
    "fp_per_control":     0.79,
}


CNN_CONFIG = {
    # Shape de entrada al CNN después de resample (igual que en entrenamiento)
    # target_affine = np.diag([1.9, 1.9, 1.9, 1]) → 96×96×96
    "input_shape":    (96, 96, 96),
    "target_affine":  [1.9, 1.9, 1.9],   # voxel size en mm del resize
    "num_classes":    2,

    # Umbral clínico optimizado en validación para Especificidad >= 0.85
    # Del fold 3: umbral=0.6875, Spec=0.884, Sens=0.558
    "clinical_threshold": 0.6875,
}

# ─── API ──────────────────────────────────────────────────────────────────────
API_CONFIG = {
    "max_upload_mb":    500,
    "allowed_suffixes": [".nii", ".gz"],
    "job_timeout_sec":  900,
}

# ─── Hardware ─────────────────────────────────────────────────────────────────
# deepmriprep usa esta misma lógica: no_gpu = not torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Validación de assets al arrancar ────────────────────────────────────────
def validate_assets() -> dict:
    checks = {
        "cnn_model":          CNN_MODEL_PATH.exists(),
        "robex":              ROBEX_SCRIPT.exists(),
        "nnunet_checkpoint":  (NNUNET_MODEL_DIR / NNUNET_CHECKPOINT).exists(),
        "nnunet_plans":       (NNUNET_MODEL_DIR / "plans.json").exists(),
        "nnunet_dataset":     (NNUNET_MODEL_DIR / "dataset.json").exists(),
        "nnunet_fingerprint": (NNUNET_MODEL_DIR / "dataset_fingerprint.json").exists(),
    }

    missing = [k for k, v in checks.items() if not v]
    if missing:
        print(f"[CONFIG] ADVERTENCIA — assets faltantes: {missing}")
    else:
        print(f"[CONFIG] Todos los assets verificados OK ({len(checks)} checks)")
    return checks


if __name__ == "__main__":
    results = validate_assets()
    for asset, ok in results.items():
        print(f"  {'✓' if ok else '✗ FALTA'}  {asset}")
