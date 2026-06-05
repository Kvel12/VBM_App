// Cliente HTTP minimalista contra el backend FastAPI.
// Usa paths relativos /api/... — Vite proxy (dev) y Nginx (prod) los enrutan.

const API_BASE = '/api/v1';

// El backend espera ModelType ∈ { deepmriprep, nnunet }.
// Los ids del frontend coinciden 1:1 con los del backend.
const MODEL_ID_MAP = {
  deepmriprep: 'deepmriprep',
  nnunet: 'nnunet',
};

const apiModelId = (frontId) => MODEL_ID_MAP[frontId] || frontId;

const parseError = async (response) => {
  try {
    const body = await response.json();
    return body.detail || body.message || `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status} ${response.statusText}`;
  }
};

// POST /analyze — multipart con archivo + metadatos.
// Retorna { job_id, message }.
export const postAnalyze = async (file, info, frontendModelId) => {
  const form = new FormData();
  form.append('file', file);
  form.append('test_name', info.test);
  form.append('model', apiModelId(frontendModelId));
  if (info.patient) form.append('patient_name', info.patient);
  if (info.notes) form.append('notes', info.notes);
  form.append('use_robex', info.useRobex ? 'true' : 'false');

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: form,
  });

  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

// GET /status/{job_id} — un snapshot del estado del job.
export const getStatus = async (jobId) => {
  const response = await fetch(`${API_BASE}/status/${jobId}`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
};

// URL al reporte .txt generado por el backend (no se descarga aquí — el
// frontend usa su propio genTxt localizado en ResultsScreen).
export const getReportURL = (jobId) => `${API_BASE}/report/${jobId}`;

// URLs del visor nnUNet — NiiVue los carga directamente vía fetch interno.
// Ambos archivos son .nii.gz; el backend sirve el T1 cargado y la máscara
// binaria producida por el modelo de segmentación.
export const getT1URL   = (jobId) => `${API_BASE}/t1/${jobId}`;
export const getMaskURL = (jobId) => `${API_BASE}/mask/${jobId}`;

// Descarga forzada de la máscara como archivo .nii.gz (botón "Descargar máscara").
export const downloadMask = async (jobId, suggestedName) => {
  const response = await fetch(getMaskURL(jobId));
  if (!response.ok) throw new Error(await parseError(response));
  const blob = await response.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = suggestedName || `mask_${jobId}.nii.gz`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
};
