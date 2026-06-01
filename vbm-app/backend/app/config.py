"""
config.py — Configuración central del backend VBM
Todas las rutas, parámetros VBM y constantes del sistema se definen aquí.
Los demás módulos importan desde este archivo, nunca hardcodean rutas.
"""

import os
from pathlib import Path

# ─── Directorios base ─────────────────────────────────────────────────────────
# BASE_DIR apunta a backend/ independientemente de dónde se ejecute
BASE_DIR = Path(__file__).resolve().parent.parent

# Activos pesados (no están en git, se copian manualmente)
DARTEL_TEMPLATES_DIR = BASE_DIR / "dartel_templates"
MODELS_DIR           = BASE_DIR / "models"

# Directorio temporal para procesamiento (se crea en runtime)
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# ─── SPM Standalone ───────────────────────────────────────────────────────────
# Dentro del contenedor Docker estos paths son fijos.
# En desarrollo local se pueden sobreescribir con variables de entorno.
SPM_STANDALONE_DIR = Path(os.environ.get(
    "SPM_STANDALONE_DIR",
    "/opt/spm12_standalone"          # path dentro del contenedor
))
MCR_DIR = Path(os.environ.get(
    "MCR_DIR",
    "/opt/mcr/v97"                   # MATLAB Runtime R2019b = v97
))

# Script de arranque de SPM Standalone (Linux: run_spm12.sh)
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

# ─── Parámetros VBM (deben coincidir EXACTAMENTE con el entrenamiento) ─────────
# Estos valores vienen de current_phase.m / run_vbm_spm12_checkpoint.m
VBM_PARAMS = {
    # Resolución de salida MNI
    "vox": [1.5, 1.5, 1.5],

    # Bounding box MNI estándar
    "bb": [[-78, -112, -70], [78, 76, 85]],

    # preserve=1: modulación (preserva volumen total de GM)
    "preserve": 1,

    # Sin suavizado en la etapa de normalización
    # (el clasificador fue entrenado sobre mapas sin suavizar)
    "fwhm": [0, 0, 0],

    # Umbral de máscara GM para cálculo de features volumétricos
    "gm_threshold": 0.1,

    # Dimensiones esperadas del mapa GM de salida
    "expected_shape": (121, 145, 121),
}

# Parámetros de segmentación unificada (Unified Segmentation SPM12)
SEG_PARAMS = {
    "biasreg":  0.001,
    "biasfwhm": 60,
    "samp":     3,
    "mrf":      1,
    "cleanup":  1,
    "reg":      [0, 0.001, 0.5, 0.05, 0.2],
    "affreg":   "mni",
    # ngaus por clase: GM, WM, CSF, skull, soft, air
    "ngaus":    [1, 1, 2, 3, 4, 2],
}

# ─── Modelos de clasificación ─────────────────────────────────────────────────
CNN_MODEL_PATH = MODELS_DIR / "spm12_cnn_v3.pth"
SVM_MODEL_PATH = MODELS_DIR / "svm_volumetric.pkl"
NNUNET_WEIGHTS = MODELS_DIR / "nnunet_weights"

# Arquitectura CNN (MedicalNet ResNet-18)
CNN_CONFIG = {
    "input_shape": (121, 145, 121),   # shape del mwp1 SPM12
    "num_classes": 2,
    "pretrained":  False,             # pesos propios, no ImageNet
}

# Parámetros de ensamblaje híbrido (SPM12-CNN + SVM)
ENSEMBLE_CONFIG = {
    # Peso de cada modelo en la fusión lineal
    # Ajustar según métricas de validación cuando estén disponibles
    "weight_cnn": 0.6,
    "weight_svm": 0.4,
}

# ─── API ──────────────────────────────────────────────────────────────────────
API_CONFIG = {
    "max_upload_mb":    500,          # tamaño máximo de T1 aceptado
    "allowed_suffixes": [".nii", ".gz"],
    "job_timeout_sec":  900,          # 15 min máximo por job
}

# ─── Hardware ─────────────────────────────────────────────────────────────────
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SPM Standalone corre en single thread por diseño (binario compilado con -singleCompThread)
# Para paralelismo se lanzan varios jobs, no varios threads dentro del mismo proceso
SPM_NUM_THREADS = 1

# ─── Validación al importar ───────────────────────────────────────────────────
def validate_assets() -> dict:
    """
    Verifica que los archivos críticos existan antes de arrancar el servidor.
    Retorna un dict con el estado de cada asset.
    Se llama desde main.py en el startup event de FastAPI.
    """
    checks = {}

    # Templates DARTEL
    for n, path in DARTEL_TEMPLATES.items():
        checks[f"template_{n}"] = path.exists()
    checks["template_6_2mni"] = DARTEL_TEMPLATE_6_MNI.exists()

    # Modelos
    checks["cnn_model"] = CNN_MODEL_PATH.exists()
    checks["svm_model"] = SVM_MODEL_PATH.exists()

    # SPM Standalone
    checks["spm_script"] = SPM_RUN_SCRIPT.exists()
    checks["mcr_dir"]    = MCR_DIR.exists()

    # ROBEX
    checks["robex"] = ROBEX_SCRIPT.exists()

    missing = [k for k, v in checks.items() if not v]
    if missing:
        print(f"[CONFIG] ADVERTENCIA — assets faltantes: {missing}")
    else:
        print(f"[CONFIG] Todos los assets verificados OK ({len(checks)} checks)")

    return checks


if __name__ == "__main__":
    # Ejecutar directamente para diagnóstico rápido
    results = validate_assets()
    for asset, ok in results.items():
        status = "✓" if ok else "✗ FALTA"
        print(f"  {status}  {asset}")