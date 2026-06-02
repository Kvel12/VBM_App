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
    id: 'hybrid',
    metricLabel: 'aucRoc',
    metricVal: '81%',
    badge: 'ensemble',
    type: 'classification',
    steps: ['load', 'robex', 'dmpVbm', 'volFeats', 'svm', 'ensemble'],
    metrics: [
      ['aucRoc',      '81 %'],
      ['sensitivity', '80 %'],
      ['specificity', '82 %'],
      ['accuracy',    '81 %'],
    ],
    sim: { epilepsy: 79, control: 21 },
  },
  {
    id: 'nnunet',
    metricLabel: 'dscMean',
    metricVal: '63%',
    badge: 'segmentation',
    type: 'segmentation',
    steps: ['load', 'robex', 'nnunetPre', 'nnunetInfer', 'nnunetPost'],
    metrics: [
      ['dscMean',     '63 %'],
      ['hausdorff95', '12.4 mm'],
      ['sensitivity', '61 %'],
      ['specificity', '98 %'],
    ],
    sim: { dsc: 63 },
  },
];
