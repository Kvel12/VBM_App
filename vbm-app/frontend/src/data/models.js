// Strings (name, desc, tooltip, badge labels, step labels, metric labels) live
// in src/i18n/translations.js under keys derived from the IDs here. Stable IDs:
//   - model.id        → resolves models.{id}.name / .fullName / .desc / .tooltip
//   - model.badge     → resolves badge.{badge}
//   - model.metricLabel → resolves metrics.{metricLabel}
//   - step (string)   → resolves steps.{step}.name / .det
//   - metric tuple [key, value] → key resolves metrics.{key}; value is shown as-is
export const MODELS = [
  {
    id: 'spm12',
    metricLabel: 'aucRoc',
    metricVal: '71%',
    badge: 'classification',
    type: 'classification',
    steps: ['load', 'robex', 'gmMap', 'classify'],
    metrics: [
      ['aucRoc', '71 %'],
      ['sensitivity', '68 %'],
      ['specificity', '74 %'],
      ['accuracy', '71 %'],
    ],
    sim: { epilepsy: 73, control: 27 },
  },
  {
    id: 'hybrid',
    metricLabel: 'aucRoc',
    metricVal: '81%',
    badge: 'ensemble',
    type: 'classification',
    recommended: true,
    steps: ['load', 'robex', 'gmMap', 'volFeats', 'svm', 'ensemble'],
    metrics: [
      ['aucRoc', '81 %'],
      ['sensitivity', '80 %'],
      ['specificity', '82 %'],
      ['accuracy', '81 %'],
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
      ['dscMean', '63 %'],
      ['hausdorff95', '12.4 mm'],
      ['sensitivity', '61 %'],
      ['specificity', '98 %'],
    ],
    sim: { dsc: 63 },
  },
];
