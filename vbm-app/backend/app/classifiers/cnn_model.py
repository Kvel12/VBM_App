"""
cnn_model.py — Loader del CNN (MedicalNet ResNet-18, TorchScript fold 0)

PREPROCESAMIENTO EXACTO DEL ENTRENAMIENTO (GMVolumeDataset._load_and_resize):
  1. Cargar .nii con nibabel
  2. nilearn.image.resample_img con target_affine=np.diag([1.9, 1.9, 1.9, 1])
     y target_shape=(96, 96, 96)
  3. Normalizar: clip al percentil 99, luego dividir por ese valor
  4. Añadir dims batch y canal → (1, 1, 96, 96, 96)

NO usar scipy.ndimage.zoom — produce predicciones incorrectas por diferencia
en el espacio de vóxeles resultante.

Umbral clínico: 0.400 (optimizado para Especificidad >= 0.85, fold 0)
"""

import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from nilearn import image

from app.config import CNN_MODEL_PATH, CNN_CONFIG, DEVICE


# ─── Singleton ────────────────────────────────────────────────────────────────
_model = None


def get_model() -> torch.jit.ScriptModule:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _load_model() -> torch.jit.ScriptModule:
    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo CNN no encontrado: {CNN_MODEL_PATH}\n"
            "Copia final_model_torchscript_fold0.pt a "
            "backend/models/spm12_cnn_fold0.pt"
        )
    print(f"[CNN] Cargando TorchScript desde {CNN_MODEL_PATH.name} → {DEVICE.upper()}")
    model = torch.jit.load(str(CNN_MODEL_PATH), map_location=DEVICE)
    model.eval()
    print("[CNN] Modelo cargado OK")
    return model


# ─── Preprocesamiento (réplica exacta del Dataset de entrenamiento) ───────────

def preprocess_gm_map(gm_map_path: Path) -> torch.Tensor:
    """
    Preprocesa un mapa GM (mwp1) para inferencia.
    Replica GMVolumeDataset._load_and_resize del notebook de entrenamiento.

    Args:
        gm_map_path: Ruta al mwp1*.nii producido por SPM12+DARTEL

    Returns:
        Tensor float32 shape (1, 1, 96, 96, 96) listo para inferencia
    """
    target_shape   = CNN_CONFIG["input_shape"]          # (96, 96, 96)
    vox_size       = CNN_CONFIG["target_affine"]        # [1.9, 1.9, 1.9]
    target_affine  = np.diag([*vox_size, 1]).astype(np.float64)

    # 1. Cargar imagen
    img = nib.load(str(gm_map_path))

    # 2. Resamplear — mismo método que en entrenamiento
    img_resized = image.resample_img(
        img,
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation='linear'
    )
    data = img_resized.get_fdata(dtype=np.float32)

    # 3. Normalizar: clip p99 y dividir — idéntico al Dataset
    p99 = float(np.percentile(data, 99))
    if p99 > 0:
        data = np.clip(data, 0, p99) / p99
    else:
        raise ValueError(
            f"El mapa GM parece estar vacío (p99 ≈ 0): {gm_map_path.name}"
        )

    # 4. Añadir dims batch y canal → (1, 1, 96, 96, 96)
    tensor = torch.from_numpy(data[np.newaxis, np.newaxis, ...]).to(DEVICE)
    return tensor


# ─── Inferencia ───────────────────────────────────────────────────────────────

def predict(gm_map_path: Path) -> dict:
    """
    Ejecuta inferencia con el CNN sobre un mapa GM.

    Usa el umbral clínico 0.400 (optimizado para Spec >= 0.85 en fold 0)
    en lugar del umbral estándar 0.5.

    Returns:
        dict con prob_epilepsy, prob_control, prediction, confidence,
        y prediction_raw (con umbral 0.5, para comparación).
    """
    model     = get_model()
    tensor    = preprocess_gm_map(gm_map_path)
    threshold = CNN_CONFIG["clinical_threshold"]  # 0.400

    with torch.no_grad():
        output = model(tensor)                           # (1, 2)
        probs  = torch.softmax(output, dim=1).squeeze()  # (2,)

    prob_control  = float(probs[0].cpu())
    prob_epilepsy = float(probs[1].cpu())

    # Predicción con umbral clínico (no 0.5)
    prediction = "epilepsy" if prob_epilepsy >= threshold else "control"
    confidence = prob_epilepsy if prediction == "epilepsy" else prob_control

    return {
        "prob_epilepsy":   round(prob_epilepsy, 4),
        "prob_control":    round(prob_control,  4),
        "prediction":      prediction,
        "confidence":      round(confidence, 4),
        "threshold_used":  threshold,
    }


# ─── Métricas del modelo (fold 0 — mejor fold individual) ────────────────────
CNN_MODEL_METRICS = {
    # Fold 0 (representativo — mejor AUC individual)
    "auc":               0.8637,
    "sensitivity":       0.7586,
    "specificity":       0.8551,
    "accuracy":          0.8068,   # balanced accuracy fold 0
    "f1_macro":          0.8009,
    "clinical_threshold": 0.400,
    "fold":              0,
    # Promedio 5-fold (para reporte completo de tesis)
    "auc_cv_mean":       0.7988,
    "auc_cv_std":        0.1005,
    "sens_cv_mean":      0.5201,
    "spec_cv_mean":      0.8671,
}