"""
config.py — Configuración central del backend VBM
Todas las rutas, parámetros VBM y constantes del sistema se definen aquí.
"""

import os
import torch
from pathlib import Path

# ─── Directorios base ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DARTEL_TEMPLATES_DIR = BASE_DIR / "dartel_templates"
MODELS_DIR           = BASE_DIR / "models"
TMP_DIR              = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# ─── SPM Standalone ───────────────────────────────────────────────────────────
SPM_STANDALONE_DIR = Path(os.environ.get(
    "SPM_STANDALONE_DIR", "/opt/spm12_standalone"
))
MCR_DIR = Path(os.environ.get(
    "MCR_DIR", "/opt/mcr/v97"
))
SPM_RUN_SCRIPT = SPM_STANDALONE_DIR / "run_spm12.sh"

# ─── ROBEX ────────────────────────────────────────────────────────────────────
ROBEX_DIR    = Path(os.environ.get("ROBEX_DIR", "/opt/ROBEX"))
ROBEX_SCRIPT = ROBEX_DIR / "runROBEX.sh"

# ─── Templates DARTEL ─────────────────────────────────────────────────────────
DARTEL_TEMPLATES = {
    1: DARTEL_TEMPLATES_DIR / "Template_1.nii",
    2: DARTEL_TEMPLATES_DIR / "Template_2.nii",
    3: DARTEL_TEMPLATES_DIR / "Template_3.nii",
    4: DARTEL_TEMPLATES_DIR / "Template_4.nii",
    5: DARTEL_TEMPLATES_DIR / "Template_5.nii",
    6: DARTEL_TEMPLATES_DIR / "Template_6.nii",
}
DARTEL_TEMPLATE_6     = DARTEL_TEMPLATES_DIR / "Template_6.nii"
DARTEL_TEMPLATE_6_MNI = DARTEL_TEMPLATES_DIR / "Template_6_2mni.mat"

# ─── Parámetros VBM (deben coincidir EXACTAMENTE con el entrenamiento) ────────
VBM_PARAMS = {
    "vox":            [1.5, 1.5, 1.5],
    "bb":             [[-78, -112, -70], [78, 76, 85]],
    "preserve":       1,
    "fwhm":           [0, 0, 0],
    "gm_threshold":   0.1,
    "expected_shape": (121, 145, 121),   # shape del mwp1 antes del resize CNN
}

SEG_PARAMS = {
    "biasreg":  0.001,
    "biasfwhm": 60,
    "samp":     3,
    "mrf":      1,
    "cleanup":  1,
    "reg":      [0, 0.001, 0.5, 0.05, 0.2],
    "affreg":   "mni",
    "ngaus":    [1, 1, 2, 3, 4, 2],
}

# ─── Modelos ──────────────────────────────────────────────────────────────────
# El .pt es TorchScript exportado con torch.jit.trace (fold 0, mejor AUC)
CNN_MODEL_PATH = MODELS_DIR / "spm12_cnn_fold0.pt"
SVM_MODEL_PATH = MODELS_DIR / "svm_volumetric.pkl"
NNUNET_WEIGHTS = MODELS_DIR / "nnunet_weights"

CNN_CONFIG = {
    # Shape de entrada al CNN después de resample (igual que en entrenamiento)
    # target_affine = np.diag([1.9, 1.9, 1.9, 1]) → 96×96×96
    "input_shape":    (96, 96, 96),
    "target_affine":  [1.9, 1.9, 1.9],   # voxel size en mm del resize
    "num_classes":    2,

    # Umbral clínico optimizado en validación para Especificidad >= 0.85
    # Del fold 0: umbral=0.400, Spec=0.855, Sens=0.759
    "clinical_threshold": 0.400,
}

ENSEMBLE_CONFIG = {
    "weight_cnn": 0.80,
    "weight_svm": 0.20,
}

# ─── API ──────────────────────────────────────────────────────────────────────
API_CONFIG = {
    "max_upload_mb":    500,
    "allowed_suffixes": [".nii", ".gz"],
    "job_timeout_sec":  900,
}

# ─── Hardware ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPM_NUM_THREADS = 1

# ─── Validación de assets al arrancar ────────────────────────────────────────
def validate_assets() -> dict:
    checks = {}
    for n, path in DARTEL_TEMPLATES.items():
        checks[f"template_{n}"] = path.exists()
    checks["template_6_2mni"] = DARTEL_TEMPLATE_6_MNI.exists()
    checks["cnn_model"]       = CNN_MODEL_PATH.exists()
    checks["svm_model"]       = SVM_MODEL_PATH.exists()
    checks["spm_script"]      = SPM_RUN_SCRIPT.exists()
    checks["mcr_dir"]         = MCR_DIR.exists()
    checks["robex"]           = ROBEX_SCRIPT.exists()

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