import axios from "axios";

// Default to a same-origin (relative) base so the app works on whatever
// host/port served it (the dev server proxies /api to the backend on :8000).
// Override with VITE_API_BASE_URL only when the API lives on a different origin.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000
});

export async function detectFile(file, sourceName) {
  const formData = new FormData();
  formData.append("file", file);
  if (sourceName) {
    formData.append("source_name", sourceName);
  }

  const { data } = await api.post("/api/detect", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
}

export async function fetchDetections(status = "pending") {
  const params = {};
  if (status && status !== "all") {
    params.status = status;
  }
  const { data } = await api.get("/api/detections", { params });
  return data;
}

export async function verifyDetection(id, plateTextVerified, verifiedBy) {
  const { data } = await api.patch(`/api/detections/${id}/verify`, {
    plate_text_verified: plateTextVerified,
    verified_by: verifiedBy
  });
  return data;
}

export async function rejectDetection(id, verifiedBy, reason) {
  const { data } = await api.patch(`/api/detections/${id}/reject`, {
    verified_by: verifiedBy,
    reason
  });
  return data;
}

export async function importTruckRun(payload) {
  const { data } = await api.post("/api/truck-runs/import", payload);
  return data;
}

export async function fetchTruckRuns() {
  const { data } = await api.get("/api/truck-runs");
  return data;
}

export async function fetchTruckRecords(runId, trackId) {
  const params = {};
  if (runId) params.run_id = runId;
  if (trackId !== "" && trackId !== null && trackId !== undefined) {
    params.track_id = Number(trackId);
  }
  const { data } = await api.get("/api/trucks", { params });
  return data;
}

export async function uploadVideoForTruckRun(videoFile, analysisJsonFile) {
  const formData = new FormData();
  formData.append("video", videoFile);
  if (analysisJsonFile) {
    formData.append("analysis_json", analysisJsonFile);
  }
  const { data } = await api.post("/api/truck-runs/upload-video", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000
  });
  return data;
}

export async function startVideoTruckRun(videoFile, analysisJsonFile) {
  const formData = new FormData();
  formData.append("video", videoFile);
  if (analysisJsonFile) {
    formData.append("analysis_json", analysisJsonFile);
  }
  const { data } = await api.post("/api/truck-runs/video/start", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000
  });
  return data;
}

export async function getVideoTruckRunStatus(jobId) {
  const { data } = await api.get(`/api/truck-runs/video/${jobId}/status`);
  return data;
}

export async function finalizeVideoTruckRun(jobId) {
  const { data } = await api.post(`/api/truck-runs/video/${jobId}/finalize`);
  return data;
}

export async function startMultiCameraTruckRun(files) {
  const formData = new FormData();
  for (const [camera, file] of Object.entries(files)) {
    if (file) formData.append(camera, file);
  }
  const { data } = await api.post("/api/truck-runs/multi-camera/start", formData);
  return data;
}

export async function getMultiCameraStatus(jobId) {
  const { data } = await api.get(`/api/truck-runs/multi-camera/${jobId}/status`);
  return data;
}

export async function finalizeMultiCamera(jobId) {
  const { data } = await api.post(`/api/truck-runs/multi-camera/${jobId}/finalize`);
  return data;
}

export async function getOcrStatus() {
  const { data } = await api.get("/api/ocr/status", { timeout: 8000 });
  return data;
}

export async function setOcrToken(token) {
  const { data } = await api.post("/api/ocr/token", { token }, { timeout: 8000 });
  return data;
}

export async function fetchFieldMedia(jobId, trackId, field) {
  const { data } = await api.get(
    `/api/field-media/${encodeURIComponent(jobId)}/${encodeURIComponent(trackId)}/${encodeURIComponent(field)}`,
    { timeout: 15000 }
  );
  return data;
}

export function getVideoTruckRunFrameUrl(jobId) {
  return `${API_BASE_URL}/api/truck-runs/video/${jobId}/frame`;
}

export function getMultiCameraFrameUrl(jobId, camera) {
  return `${API_BASE_URL}/api/truck-runs/multi-camera/${jobId}/frame/${camera}`;
}

// ── Lane setup (draw lane ROIs + gate line in the browser) ──────────────────

// Upload a camera video, get back a still frame (as an object URL) to draw on.
export async function extractLaneFrame(videoFile, camera) {
  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("camera", camera);
  const { data } = await api.post("/api/lane-setup/extract-frame", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    responseType: "blob",
    timeout: 120000
  });
  return URL.createObjectURL(data);
}

// Run the engine's truck detector on sampled frames and label each truck by the
// DRAFT lanes, so the operator can confirm assignment before saving.
export async function testLanes(videoFile, camera, lanes, samples = 8) {
  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("camera", camera);
  formData.append("lanes", JSON.stringify(lanes));
  formData.append("samples", String(samples));
  const { data } = await api.post("/api/lane-setup/test-lanes", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000
  });
  return data;
}

// Persist lanes + gate line for one camera (coords in NATIVE image pixels).
export async function saveLaneRois(payload) {
  const { data } = await api.post("/api/lane-setup/save", payload);
  return data;
}

// Read back a saved ROI for re-editing (or { exists:false }).
export async function getLaneRois(camera) {
  const { data } = await api.get(`/api/lane-setup/${camera}`);
  return data;
}
