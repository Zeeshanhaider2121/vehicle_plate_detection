const state = {
  singleVideo: null,
  cameras: { front: null, right: null, back: null, left: null },
  jobId: "",
  runId: "",
  paused: false,
  approved: 0,
  rejected: 0,
};

const els = {
  message: document.getElementById("message"),
  runId: document.getElementById("runId"),
  queueState: document.getElementById("queueState"),
  statusBar: document.getElementById("statusBar"),
  detectBtn: document.getElementById("detectBtn"),
  streamImage: document.getElementById("streamImage"),
  streamEmpty: document.getElementById("streamEmpty"),
  ocrLog: document.getElementById("ocrLog"),
  truckRows: document.getElementById("truckRows"),
  detectedCount: document.getElementById("detectedCount"),
  approvedCount: document.getElementById("approvedCount"),
  rejectedCount: document.getElementById("rejectedCount"),
  fields: {
    container: document.getElementById("fContainer"),
    side: document.getElementById("fSide"),
    plate: document.getElementById("fPlate"),
    truck: document.getElementById("fTruck"),
  },
};

function setMessage(text) {
  els.message.textContent = text;
}

function setStatus(kind, text) {
  els.statusBar.className = `status ${kind}`;
  els.statusBar.querySelector("strong").textContent = text;
  els.queueState.textContent = text;
}

function fieldText(value) {
  if (!value) return "-";
  if (typeof value === "object") return value.text || "-";
  return String(value);
}

function fieldConf(value) {
  if (!value || typeof value !== "object") return "-";
  return value.confidence == null ? "-" : Number(value.confidence).toFixed(3);
}

function selectedCameraFiles() {
  return Object.fromEntries(Object.entries(state.cameras).filter(([, file]) => file));
}

function updateCameraLabel(camera, file) {
  document.getElementById(`${camera}Name`).textContent = file ? file.name : "No file";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      detail = await response.text();
    }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function appendOcr(lines = []) {
  for (const line of lines.slice(-8)) {
    const row = document.createElement("div");
    row.textContent = line;
    if (line.includes("orphan") || line.includes("no truck")) row.className = "warn";
    els.ocrLog.appendChild(row);
  }
  els.ocrLog.scrollTop = els.ocrLog.scrollHeight;
}

function normalizeTruck(row) {
  const info = row.associated_info || row.info || {};
  return {
    id: row.id || row.track_id || row.tid,
    trackId: row.track_id || row.tid || row.id,
    info,
    ocrConf: Math.max(
      ...Object.values(info).map((value) => Number(value?.confidence || 0)),
      0
    ),
    confidenceAvg: row.confidence_avg,
    durationSec: row.duration_sec,
    firstTime: row.first_seen_time_sec,
    lastTime: row.last_seen_time_sec,
    firstFrame: row.first_seen_frame,
    lastFrame: row.last_seen_frame,
    bbox: row.last_bbox,
  };
}

function renderRows(rawRows) {
  const rows = rawRows.map(normalizeTruck);
  els.detectedCount.textContent = String(rows.length);
  els.approvedCount.textContent = String(state.approved);
  els.rejectedCount.textContent = String(state.rejected);

  if (!rows.length) {
    els.truckRows.innerHTML = '<tr><td colspan="16" class="empty-row">No detections yet.</td></tr>';
    return;
  }

  const first = rows[0];
  els.fields.container.textContent = fieldText(first.info.container_number);
  els.fields.side.textContent = fieldText(first.info.container_side_no);
  els.fields.plate.textContent = fieldText(first.info.license_plate);
  els.fields.truck.textContent = fieldText(first.info.truck_number);

  els.truckRows.innerHTML = rows.map((truck) => {
    const info = truck.info;
    const bbox = Array.isArray(truck.bbox) ? `[${truck.bbox.join(",")}]` : "-";
    return `
      <tr>
        <td>${fieldText(info.container_number)}</td>
        <td>${fieldText(info.container_side_no)}</td>
        <td>${fieldText(info.driver)}</td>
        <td>${fieldText(info.license_plate)}</td>
        <td>${fieldText(info.other_container_info)}</td>
        <td>${fieldText(info.truck_company)}</td>
        <td>${fieldText(info.truck_number)}</td>
        <td>${truck.ocrConf ? truck.ocrConf.toFixed(3) : fieldConf(info.license_plate)}</td>
        <td>${truck.confidenceAvg == null ? "-" : Number(truck.confidenceAvg).toFixed(4)}</td>
        <td>${truck.durationSec ?? "-"}</td>
        <td>${truck.firstTime ?? "-"}</td>
        <td>${truck.lastTime ?? "-"}</td>
        <td>${truck.firstFrame ?? "-"}</td>
        <td>${truck.lastFrame ?? "-"}</td>
        <td>${bbox}</td>
        <td>
          <button class="action-btn approve" data-action="approve">Approve</button>
          <button class="action-btn reject" data-action="reject">Reject</button>
        </td>
      </tr>
    `;
  }).join("");
}

function renderSnapshot(snapshot) {
  const trucks = Object.values(snapshot?.trucks || {});
  renderRows(trucks);
}

function showFrame(url) {
  if (state.paused) return;
  els.streamImage.src = `${url}${url.includes("?") ? "&" : "?"}ts=${Date.now()}`;
  els.streamImage.hidden = false;
  els.streamEmpty.hidden = true;
}

async function pollSingle(jobId) {
  for (let i = 0; i < 900; i += 1) {
    const status = await api(`/api/truck-runs/video/${jobId}/status`);
    const progress = status.progress == null ? 0 : Math.round(status.progress * 100);
    setMessage(`Job ${jobId.slice(0, 8)} | ${status.state} | ${status.frame_id || 0}/${status.total_frames || "?"} | ${progress}%`);
    appendOcr(status.ocr_log || []);
    if (status.json_snapshot) renderSnapshot(JSON.parse(status.json_snapshot));
    const latest = status.latest_frame_id || status.frame_id || 0;
    if (latest > 0) showFrame(`/api/truck-runs/video/${jobId}/frame?frame=${latest}`);
    if (status.state === "failed") throw new Error(status.message || "Video processing failed.");
    if (status.state === "completed") return;
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Video processing timed out.");
}

async function pollMulti(jobId) {
  for (let i = 0; i < 900; i += 1) {
    const status = await api(`/api/truck-runs/multi-camera/${jobId}/status`);
    const progress = status.progress == null ? 0 : Math.round(status.progress * 100);
    setMessage(`Multi-camera ${jobId.slice(0, 8)} | ${status.state} | ${progress}%`);
    appendOcr(status.ocr_log || []);
    if (status.json_snapshot) renderSnapshot(JSON.parse(status.json_snapshot));
    const cam = (status.cameras || []).find((item) => (item.latest_frame_id || item.frame_id || 0) > 0);
    if (cam) {
      const latest = cam.latest_frame_id || cam.frame_id;
      showFrame(`/api/truck-runs/multi-camera/${jobId}/frame/${cam.camera}?frame=${latest}`);
    }
    if (status.state === "failed") throw new Error(status.message || "Multi-camera processing failed.");
    if (status.state === "completed") return;
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Multi-camera processing timed out.");
}

async function loadStoredRun(runId) {
  const data = await api(`/api/trucks?run_id=${encodeURIComponent(runId)}`);
  renderRows(data.items || []);
}

async function detect() {
  const cameras = selectedCameraFiles();
  const useMulti = Object.keys(cameras).length >= 2;
  if (!useMulti && !state.singleVideo) {
    setMessage("Select one video or at least two camera videos first.");
    return;
  }

  state.paused = false;
  els.detectBtn.disabled = true;
  els.ocrLog.innerHTML = "";
  els.streamImage.hidden = true;
  els.streamEmpty.hidden = false;
  setStatus("uploading", "Uploading Video...");

  try {
    const form = new FormData();
    let start;
    if (useMulti) {
      for (const [camera, file] of Object.entries(cameras)) form.append(camera, file);
      start = await api("/api/truck-runs/multi-camera/start", { method: "POST", body: form });
      state.jobId = start.job_id;
      setStatus("processing", "Processing Detection...");
      await pollMulti(start.job_id);
      const final = await api(`/api/truck-runs/multi-camera/${start.job_id}/finalize`, { method: "POST" });
      state.runId = final.run_id;
    } else {
      form.append("video", state.singleVideo);
      start = await api("/api/truck-runs/video/start", { method: "POST", body: form });
      state.jobId = start.job_id;
      setStatus("processing", "Processing Detection...");
      await pollSingle(start.job_id);
      const final = await api(`/api/truck-runs/video/${start.job_id}/finalize`, { method: "POST" });
      state.runId = final.run_id;
    }
    els.runId.textContent = state.runId;
    await loadStoredRun(state.runId);
    setStatus("complete", "Detection Complete");
    setMessage(`Detection loaded from model output. Run ID: ${state.runId}.`);
  } catch (error) {
    setStatus("error", "Processing Failed");
    setMessage(error.message || "Processing failed.");
  } finally {
    els.detectBtn.disabled = false;
  }
}

document.getElementById("singleVideo").addEventListener("change", (event) => {
  state.singleVideo = event.target.files?.[0] || null;
  if (state.singleVideo) setMessage(`Selected ${state.singleVideo.name}. Click Detect to process.`);
});

document.querySelectorAll("[data-camera]").forEach((input) => {
  input.addEventListener("change", (event) => {
    const camera = event.target.dataset.camera;
    const file = event.target.files?.[0] || null;
    state.cameras[camera] = file;
    updateCameraLabel(camera, file);
    if (file) setMessage(`Selected ${camera} camera: ${file.name}`);
  });
});

els.detectBtn.addEventListener("click", detect);

document.getElementById("pauseBtn").addEventListener("click", () => {
  state.paused = !state.paused;
  document.getElementById("pauseBtn").textContent = state.paused ? "Play" : "Pause";
});

document.getElementById("toggleInfo").addEventListener("click", () => {
  const fields = document.getElementById("infoFields");
  const hidden = fields.hidden;
  fields.hidden = !hidden;
  document.getElementById("toggleInfo").textContent = hidden ? "Hide" : "Show";
});

document.getElementById("toggleStream").addEventListener("click", () => {
  const wrap = document.getElementById("streamWrap");
  const hidden = wrap.hidden;
  wrap.hidden = !hidden;
  document.getElementById("toggleStream").textContent = hidden ? "Hide" : "Show";
});

els.truckRows.addEventListener("click", (event) => {
  const action = event.target.dataset.action;
  if (action === "approve") state.approved += 1;
  if (action === "reject") state.rejected += 1;
  els.approvedCount.textContent = String(state.approved);
  els.rejectedCount.textContent = String(state.rejected);
});
