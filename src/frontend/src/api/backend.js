const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

const parseResponse = async (response) => {
  if (response.ok) return response.json();
  let detail = `Request failed with status ${response.status}`;
  try {
    const payload = await response.json();
    if (payload.detail) detail = payload.detail;
  } catch {
    // Keep the HTTP status when the backend does not return JSON.
  }
  throw new Error(detail);
};

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return parseResponse(await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData }));
};

export const getJobStatus = async (jobId) =>
  parseResponse(await fetch(`${API_BASE_URL}/status/${encodeURIComponent(jobId)}`));

export const normalizeDetection = (detection) => ({
  ...detection,
  class: detection.class_name,
});

export const getResults = async (jobId) => {
  const detections = await parseResponse(await fetch(`${API_BASE_URL}/results/${encodeURIComponent(jobId)}`));
  return detections.map(normalizeDetection);
};

export const getWaterfall = (jobId) =>
  `${API_BASE_URL}/waterfall/${encodeURIComponent(jobId)}`;
