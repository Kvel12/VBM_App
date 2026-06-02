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
# El .pt es TorchScript re-trazado en CPU (ver Colab retrace_cpu.py)
# Fold 3 = mejor fold individual (AUC 0.7968 — el más alto de los 5)
CNN_MODEL_PATH = MODELS_DIR / "deepmriprep_cnn_fold3_cpu.pt"

# ─── Parámetros VBM (usados por nifti_utils.extract_volumetric_features) ────
# `gm_threshold`: umbral estándar para la máscara de GM sobre mwp1 modulado
#                 (mismo valor que el notebook de entrenamiento).
# `expected_shape`: shape MNI152 producido por deepmriprep para tag de QC.
VBM_PARAMS = {
    "gm_threshold":   0.1,
    "expected_shape": (113, 137, 113),   # deepmriprep MNI152 — distinto del (121,145,121) que producía SPM12
}

CNN_CONFIG = {
    # Shape de entrada al CNN después de resample (igual que en entrenamiento)
    # target_affine = np.diag([1.9, 1.9, 1.9, 1]) → 96×96×96
    "input_shape":    (96, 96, 96),
    "target_affine":  [1.9, 1.9, 1.9],   # voxel size en mm del resize
    "num_classes":    2,

    # Umbral clínico optimizado en validación para Especificidad >= 0.85
    # Del fold 3: umbral=0.688, Spec=0.884, Sens=0.558
    "clinical_threshold": 0.688,
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
        "cnn_model": CNN_MODEL_PATH.exists(),
        "robex":     ROBEX_SCRIPT.exists(),
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
