"""
ensemble.py — Fusión lineal CNN + SVM (Modelo Híbrido)

Implementa el late fusion que combina las predicciones OOF del CNN
y del SVM con peso w_CNN=0.80 (optimizado en validación).

Resultado del ensamblaje en validación (155 sujetos OOF):
  AUC: 0.814  |  Sensibilidad: 0.616  |  Especificidad: 0.855

La fusión es simplemente:
  p_final = w_cnn * p_cnn + w_svm * p_svm

donde p_cnn y p_svm son las probabilidades de clase "epilepsia".
"""

from app.config import ENSEMBLE_CONFIG


def fuse(cnn_result: dict, svm_result: dict) -> dict:
    """
    Fusión lineal ponderada de probabilidades CNN y SVM.

    Args:
        cnn_result: dict de app.classifiers.cnn_model.predict()
        svm_result: dict de app.classifiers.svm_model.predict()

    Returns:
        dict con prob_epilepsy, prob_control, prediction, confidence,
        y los resultados individuales de cada modelo para trazabilidad.
    """
    w_cnn = ENSEMBLE_CONFIG["weight_cnn"]  # 0.80
    w_svm = ENSEMBLE_CONFIG["weight_svm"]  # 0.20 (= 1 - w_cnn implícito)

    # Normalizar pesos por si no suman 1
    total = w_cnn + w_svm
    w_cnn /= total
    w_svm /= total

    # Fusión sobre probabilidad de epilepsia
    p_epilepsy_fused = (w_cnn * cnn_result["prob_epilepsy"] +
                        w_svm * svm_result["prob_epilepsy"])
    p_control_fused  = 1.0 - p_epilepsy_fused

    prediction = "epilepsy" if p_epilepsy_fused > 0.5 else "control"
    confidence = max(p_epilepsy_fused, p_control_fused)

    return {
        "prob_epilepsy": round(p_epilepsy_fused, 4),
        "prob_control":  round(p_control_fused,  4),
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        # Detalle por modelo (útil para el reporte)
        "cnn": {
            "prob_epilepsy": cnn_result["prob_epilepsy"],
            "prob_control":  cnn_result["prob_control"],
            "weight":        round(w_cnn, 3),
        },
        "svm": {
            "prob_epilepsy": svm_result["prob_epilepsy"],
            "prob_control":  svm_result["prob_control"],
            "weight":        round(w_svm, 3),
        },
    }


# ─── Métricas del ensemble (del notebook ensemble_spm12_svm) ─────────────────
ENSEMBLE_METRICS = {
    "auc":         0.814,
    "sensitivity": 0.616,
    "specificity": 0.855,
    "accuracy":    0.735,
    "weight_cnn":  ENSEMBLE_CONFIG["weight_cnn"],
    "weight_svm":  ENSEMBLE_CONFIG["weight_svm"],
}