import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000
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

export function getVideoTruckRunFrameUrl(jobId) {
  return `${API_BASE_URL}/api/truck-runs/video/${jobId}/frame`;
}
