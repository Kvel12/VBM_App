// Strings (name, desc, tooltip, badge labels, step labels, metric labels) live
// in src/i18n/translations.js under keys derived from the IDs here. Stable IDs:
//   - model.id        → resolves models.{id}.name / .fullName / .desc / .tooltip
//   - model.badge     → resolves badge.{badge}
//   - model.metricLabel → resolves metrics.{metricLabel}
//   - step (string)   → resolves steps.{step}.name / .det
//   - metric tuple [key, value] → key resolves metrics.{key}; value is shown as-is
//
// Nota histórica: la primera tarjeta era SPM12/DARTEL. Se migró a deepmriprep
// porque SPM12 producía mapas GM no reproducibles entre macOS (entrenamiento)
// y Linux Docker (inferencia). deepmriprep es PyTorch puro → cross-platform OK.
//
// El modelo Híbrido (CNN + SVM) se descartó: la fusión óptima sobre validación
// dio w_CNN=1.00 — los features volumétricos globales no aportan señal sobre
// los mapas de deepmriprep. Documentado como hallazgo experimental en la tesis.
export const MODELS = [
  {
    id: 'deepmriprep',
    metricLabel: 'aucRoc',
    metricVal: '80%',
    badge: 'classification',
    type: 'classification',
    steps: ['load', 'robex', 'dmpVbm', 'classify'],
    metrics: [
      ['aucRoc',      '79.7 %'],
      ['sensitivity', '55.8 %'],
      ['specificity', '88.4 %'],
      ['accuracy',    '72.1 %'],
    ],
    recommended: true,
  },
  {
    id: 'nnunet',
    metricLabel: 'dscMean',
    metricVal: '82.5%',
    badge: 'segmentation',
    type: 'segmentation',
    steps: ['load', 'robex', 'nnunetPre', 'nnunetInfer', 'nnunetPost'],
    metrics: [
      ['dscMean',     '82.5 %'],
      ['hausdorff95', '9.4 mm'],
      ['sensitivity', '90.5 %'],
      ['specificity', '78.9 %'],
    ],
  },
];
