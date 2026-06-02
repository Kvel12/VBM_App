"""
svm_model.py — Loader del SVM volumétrico para el pipeline híbrido

El SVM fue entrenado sobre features escalares extraídos de los mapas GM
de SPM12+DARTEL (gm_maps_summary_spm12.csv). Los features son:
  gm_volume_cm3, gm_mean_density, gm_std_density,
  gm_p10, gm_p25, gm_p50, gm_p75, gm_p90, gm_max

El scaler (StandardScaler) se aplicó antes del entrenamiento y debe
aplicarse igual en inferencia — por eso se serializa junto al modelo
en el mismo .pkl (dict con claves 'model' y 'scaler').
"""

import joblib
import numpy as np
from pathlib import Path

from app.config import SVM_MODEL_PATH


# ─── Singleton ────────────────────────────────────────────────────────────────
_svm_bundle = None   # dict: {'model': SVC, 'scaler': StandardScaler, 'features': [...]}


def get_svm_bundle() -> dict:
    global _svm_bundle
    if _svm_bundle is None:
        _svm_bundle = _load_bundle()
    return _svm_bundle


def _load_bundle() -> dict:
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo SVM no encontrado: {SVM_MODEL_PATH}\n"
            "Copia el archivo .pkl a backend/models/svm_volumetric.pkl\n"
            "El .pkl debe ser un dict con claves: 'model', 'scaler', 'features'"
        )

    print(f"[SVM] Cargando modelo desde {SVM_MODEL_PATH}")
    bundle = joblib.load(str(SVM_MODEL_PATH))

    # Validar estructura esperada
    required_keys = {"model", "scaler", "features"}
    if not required_keys.issubset(bundle.keys()):
        # Compatibilidad: si el pkl es directamente el modelo (sin scaler)
        # wrapearlo con valores por defecto
        if hasattr(bundle, "predict_proba"):
            print("[SVM] ADVERTENCIA: pkl contiene solo el modelo sin scaler. "
                  "Los features no serán normalizados.")
            bundle = {
                "model":    bundle,
                "scaler":   None,
                "features": EXPECTED_FEATURES,
            }
        else:
            raise ValueError(
                f"El archivo .pkl no tiene la estructura esperada.\n"
                f"Se esperan las claves: {required_keys}\n"
                f"Claves encontradas: {set(bundle.keys())}"
            )

    print(f"[SVM] Modelo cargado OK — features: {bundle['features']}")
    return bundle


# ─── Features esperados (mismo orden que en entrenamiento) ────────────────────
EXPECTED_FEATURES = [
    "gm_volume_cm3",
    "gm_mean_density",
    "gm_std_density",
    "gm_p10",
    "gm_p25",
    "gm_p50",
    "gm_p75",
    "gm_p90",
    "gm_max",
]


# ─── Inferencia ───────────────────────────────────────────────────────────────

def predict(volumetric_features: dict) -> dict:
    """
    Ejecuta inferencia SVM sobre features volumétricos escalares.

    Args:
        volumetric_features: dict con al menos las claves de EXPECTED_FEATURES,
                             tal como retorna nifti_utils.extract_volumetric_features()

    Returns:
        dict con prob_epilepsy, prob_control, prediction, confidence
    """
    bundle  = get_svm_bundle()
    model   = bundle["model"]
    scaler  = bundle["scaler"]
    feature_names = bundle.get("features", EXPECTED_FEATURES)

    # Construir vector de features en el orden correcto
    try:
        X = np.array([[volumetric_features[f] for f in feature_names]],
                     dtype=np.float32)
    except KeyError as e:
        missing = str(e)
        raise ValueError(
            f"Feature faltante para el SVM: {missing}\n"
            f"Features disponibles: {list(volumetric_features.keys())}\n"
            f"Features requeridos: {feature_names}"
        )

    # Aplicar scaler si existe
    if scaler is not None:
        X = scaler.transform(X)

    # Predicción con probabilidades
    if hasattr(model, "predict_proba"):
        probs         = model.predict_proba(X)[0]   # [prob_control, prob_epilepsy]
        prob_control  = float(probs[0])
        prob_epilepsy = float(probs[1])
    else:
        # SVM sin probability=True — usar decision_function como proxy
        decision      = float(model.decision_function(X)[0])
        # Convertir a probabilidad aproximada con sigmoid
        prob_epilepsy = float(1 / (1 + np.exp(-decision)))
        prob_control  = 1.0 - prob_epilepsy

    prediction = "epilepsy" if prob_epilepsy > prob_control else "control"
    confidence = max(prob_epilepsy, prob_control)

    return {
        "prob_epilepsy": round(prob_epilepsy, 4),
        "prob_control":  round(prob_control, 4),
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "features_used": feature_names,
    }


# ─── Métricas del modelo (del conjunto de validación OOF) ────────────────────
# Del notebook classical_models_v2/spm12 — SVM OOF sobre 155 sujetos
SVM_MODEL_METRICS = {
    "auc":         0.615,
    "sensitivity": 0.580,
    "specificity": 0.660,
    "accuracy":    0.614,
}