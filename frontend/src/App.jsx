import { useMemo, useRef, useState } from "react";
import {
  fetchTruckRecords,
  finalizeMultiCamera,
  finalizeVideoTruckRun,
  getMultiCameraFrameUrl,
  getMultiCameraStatus,
  getVideoTruckRunFrameUrl,
  getVideoTruckRunStatus,
  startMultiCameraTruckRun,
  startVideoTruckRun
} from "./api";

const BLURRY_KEYWORDS = [
  "blurr", "indistinct", "unrecognizable", "no discernible",
  "no visible", "unreadable", "cannot be determined",
  "not visible", "unclear", "unable to", "no text"
];

function isBlurry(text) {
  if (!text) return false;
  const t = String(text).toLowerCase();
  return BLURRY_KEYWORDS.some(kw => t.includes(kw)) || t.length > 60;
}

function mergeInfo(existing, incoming) {
  const result = { ...(existing || {}) };
  for (const key of Object.keys(incoming || {})) {
    const inc = incoming[key];
    const ext = result[key];
    if (inc === null || inc === undefined || inc === "") continue;
    if (!ext || ext === "" || ext === null || ext === undefined) {
      result[key] = inc;
      continue;
    }
    // Prefer higher-confidence OCR result when both are structured objects
    if (typeof inc === "object" && typeof ext === "object" && inc.confidence != null && ext.confidence != null) {
      if (inc.confidence > ext.confidence) result[key] = inc;
      continue;
    }
    if (isBlurry(inc) && !isBlurry(ext)) continue;
    if (!isBlurry(inc) && isBlurry(ext)) { result[key] = inc; continue; }
    if (String(inc).length < String(ext).length) result[key] = inc;
  }
  return result;
}

const menuItems = [
  "Arrive Unit",
  "Open EIR",
  "Depart Unit",
  "Inspect Units",
  "EIR Search",
  "Gensets",
  "Stacked Chassis",
  "Pre-Arrive",
  "Pre-Arrive Admin"
];

function cleanOcrText(text) {
  if (text === null || text === undefined) return text;
  // Strip decorative '#' and '$' characters that leak in from OCR / markdown
  // (e.g. "# CA-40761" -> "CA-40761", "## REAR" -> "REAR"), then collapse spaces.
  return String(text).replace(/[#$]/g, "").replace(/\s+/g, " ").trim();
}

function readOcrFieldValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return cleanOcrText(value) || "—";
  if (typeof value === "object" && value.text) return cleanOcrText(value.text) || "—";
  return "—";
}

// The truck company is always one of these two real carriers. OCR often reads it
// wrong (e.g. "MAYA Organized Inc"), so snap whatever was read to the closest one.
const KNOWN_TRUCK_COMPANIES = ["OCEANLAND INC", "OCEANHUB INC"];

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) => [i, ...Array(n).fill(0)]);
  for (let j = 0; j <= n; j += 1) dp[0][j] = j;
  for (let i = 1; i <= m; i += 1) {
    for (let j = 1; j <= n; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

function normalizeTruckCompany(value) {
  if (value === null || value === undefined || value === "—") return value;
  const text = String(value).trim();
  if (!text) return "—";
  const lower = text.toLowerCase();
  // Distinguishing tokens win immediately.
  if (lower.includes("hub")) return "OCEANHUB INC";
  if (lower.includes("land") || lower.includes("ocean")) return "OCEANLAND INC";
  // Otherwise pick the closest canonical name by edit distance (ties -> first).
  let best = KNOWN_TRUCK_COMPANIES[0];
  let bestDist = Infinity;
  for (const cand of KNOWN_TRUCK_COMPANIES) {
    const d = levenshtein(lower, cand.toLowerCase());
    if (d < bestDist) {
      bestDist = d;
      best = cand;
    }
  }
  return best;
}

function readOcrFieldConfidence(value) {
  if (value !== null && typeof value === "object" && value.confidence != null) return value.confidence;
  return null;
}

function readOcrFieldCamera(value) {
  if (value !== null && typeof value === "object" && value.camera) return value.camera;
  return null;
}

function readTruckFields(truck) {
  const info = truck?.associated_info || {};
  const pickFirst = (...values) =>
    values.find((value) => value !== null && value !== undefined && String(value).trim() !== "") || "—";

  const ocrFields = [
    "container_company_logo", "container_number", "container_side_no",
    "driver", "license_plate", "other_container_info",
    "truck_company", "truck_number"
  ];
  const fieldTexts = {};
  const fieldConfs = {};
  const fieldCameras = {};
  for (const f of ocrFields) {
    fieldTexts[f] = readOcrFieldValue(info[f]);
    const c = readOcrFieldConfidence(info[f]);
    fieldConfs[f] = c != null ? c.toFixed(3) : "—";
    const cam = readOcrFieldCamera(info[f]);
    if (cam) fieldCameras[f] = cam;
  }
  // Constrain truck company to a known carrier name.
  fieldTexts.truck_company = normalizeTruckCompany(fieldTexts.truck_company);
  // Show max OCR confidence across filled fields — reflects the best crop selected per class
  const ocrConfs = ocrFields.map((f) => readOcrFieldConfidence(info[f])).filter((c) => c != null);
  const maxOcrConf = ocrConfs.length > 0 ? Math.max(...ocrConfs).toFixed(3) : "—";

  return {
    trackId: pickFirst(truck?.track_id),
    truckClass: pickFirst(truck?.truck_type, truck?.type),
    camera: pickFirst(truck?.camera, truck?.associated_info?._fusion?.source_cameras?.[0]),
    fieldCameras,
    ...fieldTexts,
    ...Object.fromEntries(Object.entries(fieldConfs).map(([k, v]) => [k + "_conf", v])),
    driver: fieldTexts.driver,
    license_plate: fieldTexts.license_plate,
    truck_company: fieldTexts.truck_company,
    truck_number: fieldTexts.truck_number,
    confidence: pickFirst(truck?.confidence_avg),
    ocr_confidence: maxOcrConf,
    durationSec: pickFirst(truck?.duration_sec),
    firstSeen: pickFirst(truck?.first_seen_time_sec),
    lastSeen: pickFirst(truck?.last_seen_time_sec),
    bbox: pickFirst(truck?.last_bbox ? JSON.stringify(truck.last_bbox) : null),
    firstSeenFrame: pickFirst(truck?.first_seen_frame),
    lastSeenFrame: pickFirst(truck?.last_seen_frame)
  };
}

function shortText(value, max = 22) {
  const text = String(value ?? "");
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(0, max - 3))}...`;
}

function formatTimeSec(sec) {
  if (sec === null || sec === undefined || sec === "—") return "—";
  const s = Number(sec);
  if (isNaN(s)) return "—";
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60).toString().padStart(2, "0");
  return `${m}:${ss}`;
}

function stripOcrMarkdown(value) {
  if (!value || typeof value !== "string") return value;
  // Remove markdown image ![alt](path)
  let cleaned = value.replace(/!\[.*?\]\(.*?\)\s*/g, "");
  // Remove HTML tags
  cleaned = cleaned.replace(/<[^>]+>/g, " ");
  // Remove decorative '#' and '$' signs (OCR/markdown artifacts)
  cleaned = cleaned.replace(/[#$]/g, "");
  // Collapse whitespace and trim
  cleaned = cleaned.replace(/\s+/g, " ").trim();
  // If nothing intelligible remains, return a short fallback
  return cleaned || (value.length > 80 ? value.slice(0, 77) + "..." : value);
}

function normalizeTruckFromSnapshot(trackId, truck) {
  const safeTrackId = Number(trackId ?? truck?.tid ?? truck?.track_id);

  // The inference snapshot returns { tid, type, first, last, conf, info }
  // rather than { track_id, type, first_seen_frame, ... associated_info }
  const rawInfo = truck?.info || truck?.associated_info || {};
  const cleanedInfo = {};
  for (const [k, v] of Object.entries(rawInfo)) {
    cleanedInfo[k] = stripOcrMarkdown(v);
  }

  return {
    id: `track-${safeTrackId}`,
    track_id: safeTrackId,
    truck_type: truck?.type || "—",
    camera: truck?.camera || "",
    first_seen_frame: truck?.first ?? truck?.first_seen_frame ?? null,
    last_seen_frame: truck?.last ?? truck?.last_seen_frame ?? null,
    first_seen_time_sec: truck?.first_seen_time_sec ?? null,
    last_seen_time_sec: truck?.last_seen_time_sec ?? null,
    duration_frames: truck?.duration_frames ?? null,
    duration_sec: truck?.duration_sec ?? null,
    confidence_avg: truck?.conf ?? truck?.confidence_avg ?? null,
    last_bbox: truck?.last_bbox ?? null,
    associated_info: cleanedInfo,
    review_status: "pending"
  };
}

function normalizeTruckFromDb(truck) {
  return {
    ...truck,
    id: `track-${truck?.track_id}`,
    truck_type: truck?.truck_type || truck?.type || "—",
    associated_info: truck?.associated_info || {},
    review_status: "pending"
  };
}

export default function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [analysisJsonFile, setAnalysisJsonFile] = useState(null);
  const [runId, setRunId] = useState("");
  const [detectedTrucks, setDetectedTrucks] = useState([]);
  const [reviewIndex, setReviewIndex] = useState(0);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [dashboardView, setDashboardView] = useState("table");
  const [activeJobId, setActiveJobId] = useState("");
  const [streamFrameUrl, setStreamFrameUrl] = useState("");
  const [streamWarning, setStreamWarning] = useState("");
  const [streamErrorCount, setStreamErrorCount] = useState(0);
  const [selectedTruckId, setSelectedTruckId] = useState("");
  const [processStatus, setProcessStatus] = useState("idle");
  const [videoPlaying, setVideoPlaying] = useState(true);
  const [ocrLog, setOcrLog] = useState([]);
  const [groupByField, setGroupByField] = useState(false);
  const [multiCamMode, setMultiCamMode] = useState(false);
  const [camFiles, setCamFiles] = useState({ front: null, right: null, back: null, left: null });
  const [camStreamUrls, setCamStreamUrls] = useState({ front: null, right: null, back: null, left: null });
  const [videoZoom, setVideoZoom] = useState(1);
  const videoRef = useRef(null);
  const streamFrameLoadingRef = useRef(false);
  const lastRequestedFrameRef = useRef("");
  const hasSelectedVideo = Boolean(videoFile || Object.values(camFiles).some(Boolean));

  function toggleVideo() {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setVideoPlaying(true);
    } else {
      videoRef.current.pause();
      setVideoPlaying(false);
    }
  }

  const currentTruck =
    detectedTrucks.find((truck) => truck.id === selectedTruckId) ||
    detectedTrucks.find((truck) => truck.review_status === "pending") ||
    detectedTrucks[reviewIndex] ||
    null;
  const currentFields = useMemo(() => readTruckFields(currentTruck), [currentTruck]);

  const counts = useMemo(() => {
    const summary = { detected: detectedTrucks.length, approved: 0, rejected: 0, pending: 0, withTrailer: 0, withoutTrailer: 0 };
    for (const truck of detectedTrucks) {
      if (truck.review_status === "approved") summary.approved += 1;
      else if (truck.review_status === "rejected") summary.rejected += 1;
      else summary.pending += 1;
      const type = (truck.truck_type || truck.type || "").toLowerCase();
      if (type.includes("with_container") || type.includes("with container")) summary.withTrailer += 1;
      else summary.withoutTrailer += 1;
    }
    return summary;
  }, [detectedTrucks]);

  const pendingCount = counts.pending;
  const selectedCameraCount = Object.values(camFiles).filter(Boolean).length;

  const displayRows = useMemo(() => {
    if (!groupByField) return detectedTrucks;

    const groups = new Map();
    for (const truck of detectedTrucks) {
      const fields = readTruckFields(truck);
      const key =
        (fields.license_plate !== "—" ? fields.license_plate : null) ||
        (fields.container_number !== "—" ? fields.container_number : null) ||
        (fields.truck_number !== "—" ? fields.truck_number : null) ||
        truck.id;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(truck);
    }

    return [...groups.values()].map((trucks) => {
      const best = trucks.reduce((a, b) =>
        (a.confidence_avg ?? 0) >= (b.confidence_avg ?? 0) ? a : b, trucks[0]);
      let merged = {};
      for (const t of trucks) merged = mergeInfo(merged, t.associated_info || {});
      return {
        ...best,
        associated_info: merged,
        _sourceTracks: trucks.map((t) => `T#${t.track_id}`).join(" + "),
        _groupCount: trucks.length,
      };
    });
  }, [detectedTrucks, groupByField]);

  const gateRows = useMemo(() => {
    const groups = new Map();
    for (const truck of detectedTrucks) {
      const fields = readTruckFields(truck);
      const key =
        (fields.license_plate !== "—" ? fields.license_plate : null) ||
        (fields.container_number !== "—" ? fields.container_number : null) ||
        (fields.truck_number !== "—" ? fields.truck_number : null) ||
        truck.id;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(truck);
    }
    return [...groups.values()].map((trucks) => {
      const best = trucks.reduce((a, b) =>
        (a.confidence_avg ?? 0) >= (b.confidence_avg ?? 0) ? a : b, trucks[0]);
      let merged = {};
      for (const t of trucks) merged = mergeInfo(merged, t.associated_info || {});
      // Collect camera → track_id from _fusion metadata across all trucks in group
      const cameraMap = {};
      for (const t of trucks) {
        const ft = t.associated_info?._fusion?.source_track_ids;
        if (ft && Object.keys(ft).length > 0) Object.assign(cameraMap, ft);
      }
      if (Object.keys(cameraMap).length === 0) {
        cameraMap[`T#${best.track_id}`] = best.track_id;
      }
      return {
        ...best,
        associated_info: { ...merged, _fusion: { ...(merged._fusion || {}), source_track_ids: cameraMap } },
        _sourceTracks: trucks.map((t) => `T#${t.track_id}`).join(" + "),
        _groupCount: trucks.length,
      };
    });
  }, [detectedTrucks]);

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function mergeSnapshotRows(snapshotTrucks) {
    if (!snapshotTrucks || typeof snapshotTrucks !== "object") return;
    setDetectedTrucks((prev) => {
      const byId = new Map(prev.map((truck) => [truck.id, truck]));

      for (const [trackId, payload] of Object.entries(snapshotTrucks)) {
        const incoming = normalizeTruckFromSnapshot(trackId, payload);
        const existing = byId.get(incoming.id);
        if (!existing) {
          byId.set(incoming.id, incoming);
          continue;
        }
        byId.set(incoming.id, {
          ...existing,
          ...incoming,
          associated_info: mergeInfo(existing.associated_info, incoming.associated_info),
          review_status: existing.review_status || "pending"
        });
      }

      const merged = [...byId.values()].sort((a, b) => Number(a.track_id) - Number(b.track_id));
      return merged;
    });
  }

  async function pollJobUntilComplete(jobId) {
    const maxPolls = 900;
    const maxRetries = 5;
    const maxCachedPolls = 15;
    let retries = 0;
    let cachedPolls = 0;

    for (let i = 0; i < maxPolls; i += 1) {
      let status;
      try {
        status = await getVideoTruckRunStatus(jobId);
      } catch {
        retries += 1;
        if (retries <= maxRetries) {
          await sleep(2000);
          continue;
        }
        throw new Error("Video processing backend unreachable after retries.");
      }
      retries = 0;

      const total = status.total_frames ?? "?";
      const frame = status.frame_id ?? 0;
      const latestFrame = status.latest_frame_id ?? frame;
      const pct = status.progress != null ? `${Math.round(status.progress * 100)}%` : "0%";
      setMessage(`Job ${jobId.slice(0, 8)} | ${status.state} | ${frame}/${total} | shown ${latestFrame} | ${pct}`);
      const frameKey = `single:${latestFrame}`;
      if (latestFrame > 0 && frameKey !== lastRequestedFrameRef.current && !streamFrameLoadingRef.current) {
        streamFrameLoadingRef.current = true;
        lastRequestedFrameRef.current = frameKey;
        setStreamFrameUrl(`${getVideoTruckRunFrameUrl(jobId)}?frame=${latestFrame}&ts=${Date.now()}`);
      }
      if (status.ocr_log && status.ocr_log.length > 0) {
        setOcrLog((prev) => {
          const combined = [...prev, ...status.ocr_log];
          return combined.slice(-50);
        });
      }

      if (status.json_snapshot) {
        try {
          const snap = JSON.parse(status.json_snapshot);
          mergeSnapshotRows(snap?.trucks || {});
        } catch {
          // Ignore malformed snapshots and continue polling.
        }
      }

      if (status.state === "completed") {
        return;
      }
      if (status.state === "cached") {
        cachedPolls += 1;
        if (cachedPolls > maxCachedPolls) {
          throw new Error("Local inference service is unavailable; last cached snapshot is shown above.");
        }
        await sleep(2000);
        continue;
      }
      cachedPolls = 0;
      if (status.state === "failed") {
        throw new Error(status.message || "Video processing failed.");
      }
      await sleep(2000);
    }
    throw new Error("Video processing timed out while waiting for completion.");
  }

  async function pollMultiCameraUntilComplete(jobId) {
    const maxPolls = 900;
    const maxRetries = 5;
    let retries = 0;
    for (let i = 0; i < maxPolls; i += 1) {
      let status;
      try {
        status = await getMultiCameraStatus(jobId);
      } catch {
        retries += 1;
        if (retries <= maxRetries) { await sleep(2000); continue; }
        throw new Error("Multi-camera backend unreachable after retries.");
      }
      retries = 0;
      const progress = status.progress != null ? `${Math.round(status.progress * 100)}%` : "0%";
      setMessage(`Multi-camera ${jobId.slice(0, 8)} | ${status.state} | ${progress}`);
      for (const camStatus of (status.cameras || [])) {
        const latestFrame = camStatus.latest_frame_id ?? camStatus.frame_id ?? 0;
        if (latestFrame > 0) {
          const frameKey = `${camStatus.camera}:${latestFrame}`;
          if (frameKey !== lastRequestedFrameRef.current) {
            lastRequestedFrameRef.current = frameKey;
            const url = `${getMultiCameraFrameUrl(jobId, camStatus.camera)}?frame=${latestFrame}&ts=${Date.now()}`;
            setCamStreamUrls((prev) => ({ ...prev, [camStatus.camera]: url }));
            setStreamFrameUrl(url);
          }
        }
      }
      if (status.ocr_log && status.ocr_log.length > 0) {
        setOcrLog((prev) => {
          const combined = [...prev, ...status.ocr_log];
          return combined.slice(-50);
        });
      }
      if (status.json_snapshot) {
        try {
          const snap = JSON.parse(status.json_snapshot);
          mergeSnapshotRows(snap?.trucks || {});
        } catch { /* ignore */ }
      }
      if (status.state === "completed") return;
      if (status.state === "failed") throw new Error(status.message || "Multi-camera processing failed.");
      await sleep(2000);
    }
    throw new Error("Multi-camera processing timed out.");
  }

  async function handleVideoUpload() {
    const useMultiCamera = selectedCameraCount >= 1;
    setLoading(true);
    setMessage("");
    setProcessStatus("uploading");
    try {
      setDetectedTrucks([]);
      setSelectedTruckId("");
      setReviewIndex(0);
      setActiveJobId("");
      setStreamFrameUrl("");
      setCamStreamUrls({ front: null, right: null, back: null, left: null });
      streamFrameLoadingRef.current = false;
      lastRequestedFrameRef.current = "";
      setStreamWarning("");
      setStreamErrorCount(0);

      let currentRunId;
      let storedTrucks = 0;

      if (useMultiCamera) {
        const selected = Object.fromEntries(
          Object.entries(camFiles).filter(([, f]) => f != null)
        );
        if (Object.keys(selected).length < 1) {
          throw new Error("Select at least one camera angle.");
        }
        setMessage("Starting multi-camera jobs...");
        const start = await startMultiCameraTruckRun(selected);
        const aggregateJobId = start.job_id;
        if (!aggregateJobId) throw new Error("Missing multi-camera job_id.");
        setActiveJobId(aggregateJobId);
        setProcessStatus("processing");
        await pollMultiCameraUntilComplete(aggregateJobId);
        const finalized = await finalizeMultiCamera(aggregateJobId);
        currentRunId = finalized.run_id;
        storedTrucks = finalized.stored_trucks;
      } else {
        if (!videoFile) {
          setMessage("Please select a video file.");
          setLoading(false);
          return;
        }
        setMessage(`File uploaded: ${videoFile.name}. Starting GPU processing...`);
        const start = await startVideoTruckRun(videoFile, analysisJsonFile);
        currentRunId = start.run_id;
        storedTrucks = start.stored_trucks ?? 0;
        if (start.mode !== "direct_json") {
          const jobId = start.job_id;
          if (!jobId) throw new Error("Missing job_id from backend.");
          setActiveJobId(jobId);
          setMessage(`Video job started (${jobId.slice(0, 8)}). Processing locally...`);
          setProcessStatus("processing");
          await pollJobUntilComplete(jobId);
          const finalized = await finalizeVideoTruckRun(jobId);
          currentRunId = finalized.run_id;
          storedTrucks = finalized.stored_trucks;
        }
      }

      if (!currentRunId) throw new Error("No run_id returned.");

      setRunId(currentRunId);
      const trucksResult = await fetchTruckRecords(currentRunId, "");
      const dbTrucks = (trucksResult.items || []).map((truck) => normalizeTruckFromDb(truck));
      if (dbTrucks.length > 0) {
        setDetectedTrucks((prev) => {
          const byId = new Map(prev.map((truck) => [truck.id, truck]));
          for (const dbTruck of dbTrucks) {
            const existing = byId.get(dbTruck.id);
            byId.set(dbTruck.id, {
              ...(existing || {}),
              ...dbTruck,
              associated_info: mergeInfo(existing?.associated_info, dbTruck.associated_info),
              review_status: existing?.review_status || "pending"
            });
          }
          return [...byId.values()].sort((a, b) => Number(a.track_id) - Number(b.track_id));
        });
      }

      setProcessStatus("complete");
      setMessage(`Detection loaded. Run ID: ${currentRunId}. Trucks detected: ${storedTrucks}.`);
      setStreamWarning("");
    } catch (error) {
      setProcessStatus("error");
      setMessage(error?.response?.data?.detail || error.message || "Video upload or detection import failed.");
    } finally {
      setLoading(false);
    }
  }

  function handleApprove(truckId) {
    const targetId = truckId ?? currentTruck?.id;
    if (!targetId) return;

    setSelectedTruckId(targetId);
    const updated = detectedTrucks.map((truck) =>
      truck.id === targetId ? { ...truck, review_status: "approved" } : truck
    );
    setDetectedTrucks(updated);
    const nextPendingIndex = updated.findIndex((truck) => truck.review_status === "pending");
    if (nextPendingIndex >= 0) setReviewIndex(nextPendingIndex);
    const approvedTruck = updated.find((truck) => truck.id === targetId);
    if (approvedTruck) setMessage(`Truck ID ${approvedTruck.track_id} approved and added to dashboard.`);
  }

  function handleReject(truckId) {
    const targetId = truckId ?? currentTruck?.id;
    if (!targetId) return;

    setSelectedTruckId(targetId);
    const updated = detectedTrucks.map((truck) =>
      truck.id === targetId ? { ...truck, review_status: "rejected" } : truck
    );
    setDetectedTrucks(updated);
    const nextPendingIndex = updated.findIndex((truck) => truck.review_status === "pending");
    if (nextPendingIndex >= 0) setReviewIndex(nextPendingIndex);
    const rejectedTruck = updated.find((truck) => truck.id === targetId);
    if (rejectedTruck) setMessage(`Truck ID ${rejectedTruck.track_id} rejected.`);
  }

  return (
    <div className="os-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">O</div>
          <div>
            <h2>Logistics</h2>
            <p>TERMINAL OS</p>
          </div>
        </div>

        <div className="menu-group">
          <div className="menu-head">
            <span>Gate</span>
            <span>^</span>
          </div>
          <ul>
            {menuItems.map((item, index) => (
              <li key={item} className={index === 0 ? "active" : ""}>
                {item}
              </li>
            ))}
          </ul>
        </div>

        <div className="profile-card">
          <div className="avatar">JD</div>
          <div>
            <h4>John Dispatcher</h4>
            <p>OPS DIRECTOR</p>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="top-links">
            <span>Help &amp; Support</span>
            <span>Analytics</span>
            <span>
              System Status <b>OPTIMAL</b>
            </span>
          </div>
        </header>

        <section className="content">
          <div className="content-head">
            <div>
              <p className="crumb">EXIT TERMINAL</p>
              <h1>
                Gate Arrival <span>Registry</span>
              </h1>
            </div>
            <div className="status-strip">
              <div>
                <small>RUN ID</small>
                <p>{runId ? runId.slice(0, 12) : "N/A"}</p>
              </div>
              <div>
                <small>QUEUE</small>
                <p>{pendingCount} Pending</p>
              </div>
            </div>
          </div>

          {message && <div className="message-banner">{message}</div>}

          <div className="content-grid">
            <div className="form-pane">
              <div className="form-block">
                <h3>TRUCKER INFORMATION</h3>
                <p>POPULATED FROM MODEL CLASS FIELDS
                  {currentFields.camera ? <span className="cam-source-chip" style={{marginLeft: 8}}>{currentFields.camera.toUpperCase()}</span> : null}
                </p>
                <div className="field-row">
                  <div className="field-wrap">
                    <span className="field-name">
                      truck_company
                      {currentFields.fieldCameras?.truck_company && <span className={`field-cam-tag cam-${currentFields.fieldCameras.truck_company}`}>{currentFields.fieldCameras.truck_company.toUpperCase()}</span>}
                    </span>
                    <input value={currentFields.truck_company} placeholder="truck_company" readOnly className={currentFields.fieldCameras?.truck_company ? `cam-border-${currentFields.fieldCameras.truck_company}` : ""} />
                  </div>
                  <button type="button" disabled>
                    DETECTED
                  </button>
                  <div className="field-wrap">
                    <span className="field-name">
                      driver
                      {currentFields.fieldCameras?.driver && <span className={`field-cam-tag cam-${currentFields.fieldCameras.driver}`}>{currentFields.fieldCameras.driver.toUpperCase()}</span>}
                    </span>
                    <input value={currentFields.driver} placeholder="driver" readOnly className={currentFields.fieldCameras?.driver ? `cam-border-${currentFields.fieldCameras.driver}` : ""} />
                  </div>
                </div>
                <div className="field-row two with-gap">
                  <div className="field-wrap">
                    <span className="field-name">
                      license_plate
                      {currentFields.fieldCameras?.license_plate && <span className={`field-cam-tag cam-${currentFields.fieldCameras.license_plate}`}>{currentFields.fieldCameras.license_plate.toUpperCase()}</span>}
                    </span>
                    <input value={currentFields.license_plate} placeholder="license_plate" readOnly className={currentFields.fieldCameras?.license_plate ? `cam-border-${currentFields.fieldCameras.license_plate}` : ""} />
                  </div>
                  <div className="field-wrap">
                    <span className="field-name">
                      truck_number
                      {currentFields.fieldCameras?.truck_number && <span className={`field-cam-tag cam-${currentFields.fieldCameras.truck_number}`}>{currentFields.fieldCameras.truck_number.toUpperCase()}</span>}
                    </span>
                    <input value={currentFields.truck_number} placeholder="truck_number" readOnly className={currentFields.fieldCameras?.truck_number ? `cam-border-${currentFields.fieldCameras.truck_number}` : ""} />
                  </div>
                </div>
              </div>

              <div className="form-block">
                <h3>EQUIPMENT INFORMATION</h3>
                <p>AUTO-FILLED DETECTION FIELDS (MODEL CLASSES)</p>
                <div className="field-row two">
                  <div className="pair">
                    <div className="field-wrap">
                      <span className="field-name">
                        container_number
                        {currentFields.fieldCameras?.container_number && <span className={`field-cam-tag cam-${currentFields.fieldCameras.container_number}`}>{currentFields.fieldCameras.container_number.toUpperCase()}</span>}
                      </span>
                      <input value={currentFields.container_number} placeholder="container_number" readOnly className={currentFields.fieldCameras?.container_number ? `cam-border-${currentFields.fieldCameras.container_number}` : ""} />
                    </div>
                    <div className="field-wrap">
                      <span className="field-name">
                        container_side_no
                        {currentFields.fieldCameras?.container_side_no && <span className={`field-cam-tag cam-${currentFields.fieldCameras.container_side_no}`}>{currentFields.fieldCameras.container_side_no.toUpperCase()}</span>}
                      </span>
                      <input value={currentFields.container_side_no} placeholder="container_side_no" readOnly className={currentFields.fieldCameras?.container_side_no ? `cam-border-${currentFields.fieldCameras.container_side_no}` : ""} />
                    </div>
                  </div>
                  <div className="pair">
                    <div className="field-wrap">
                      <span className="field-name">
                        container_company_logo
                        {currentFields.fieldCameras?.container_company_logo && <span className={`field-cam-tag cam-${currentFields.fieldCameras.container_company_logo}`}>{currentFields.fieldCameras.container_company_logo.toUpperCase()}</span>}
                      </span>
                      <input value={currentFields.container_company_logo} placeholder="container_company_logo" readOnly className={currentFields.fieldCameras?.container_company_logo ? `cam-border-${currentFields.fieldCameras.container_company_logo}` : ""} />
                    </div>
                    <div className="field-wrap">
                      <span className="field-name">
                        other_container_info
                        {currentFields.fieldCameras?.other_container_info && <span className={`field-cam-tag cam-${currentFields.fieldCameras.other_container_info}`}>{currentFields.fieldCameras.other_container_info.toUpperCase()}</span>}
                      </span>
                      <input value={currentFields.other_container_info} placeholder="other_container_info" readOnly className={currentFields.fieldCameras?.other_container_info ? `cam-border-${currentFields.fieldCameras.other_container_info}` : ""} />
                    </div>
                  </div>
                </div>
                <div className="field-row three with-gap">
                  <div className="field-wrap">
                    <span className="field-name">track_id</span>
                    <input value={currentFields.trackId} placeholder="track_id" readOnly />
                  </div>
                  <div className="field-wrap">
                    <span className="field-name">truck_class</span>
                    <input value={currentFields.truckClass} placeholder="truck_class" readOnly />
                  </div>
                  <div className="field-wrap">
                    <span className="field-name">confidence_avg</span>
                    <input value={currentFields.confidence} placeholder="confidence_avg" readOnly />
                  </div>
                </div>
                <div className="field-row two with-gap">
                  <div className="field-wrap">
                    <span className="field-name">duration_sec</span>
                    <input value={currentFields.durationSec} placeholder="duration_sec" readOnly />
                  </div>
                  <div className="field-wrap">
                    <span className="field-name">last_bbox</span>
                    <input value={currentFields.bbox} placeholder="last_bbox" readOnly />
                  </div>
                </div>
              </div>

              <div className="approval-row">
                <button
                  type="button"
                  className="approve-btn"
                  onClick={handleApprove}
                  disabled={!currentTruck || currentTruck.review_status === "rejected"}
                >
                  Approve
                </button>
                <button
                  type="button"
                  className="reject-btn"
                  onClick={handleReject}
                  disabled={!currentTruck || currentTruck.review_status === "rejected"}
                >
                  Reject
                </button>
              </div>
            </div>

            <aside className="stream-pane">
              <div className="stream-head">
                <span>DEMO STREAM</span>
              </div>
              <div className="stream-view">
                {selectedCameraCount >= 1 && Object.values(camStreamUrls).some(Boolean) ? (
                  <div className="multi-cam-grid">
                    {["front", "right", "back", "left"].map((cam) =>
                      camFiles[cam] ? (
                        <div key={cam} className="multi-cam-cell">
                          <span className="cam-label-badge">{cam.toUpperCase()}</span>
                          {camStreamUrls[cam] ? (
                            <img
                              src={camStreamUrls[cam]}
                              alt={`${cam} stream`}
                              className="stream-image"
                              style={{ transform: `scale(${videoZoom})`, transformOrigin: "center center" }}
                              onError={() => { streamFrameLoadingRef.current = false; }}
                              onLoad={() => { streamFrameLoadingRef.current = false; }}
                            />
                          ) : (
                            <div className="stream-empty cam-waiting">Waiting for {cam}…</div>
                          )}
                        </div>
                      ) : null
                    )}
                  </div>
                ) : activeJobId && streamFrameUrl ? (
                  <img
                    src={streamFrameUrl}
                    alt="Live detection stream"
                    className="stream-image"
                    style={{ transform: `scale(${videoZoom})`, transformOrigin: "center center" }}
                    onError={() => {
                      streamFrameLoadingRef.current = false;
                      const next = streamErrorCount + 1;
                      setStreamErrorCount(next);
                      if (next >= 3) {
                        setStreamWarning(
                          "Live frame is temporarily unavailable (network/OCR delay). Processing is still running."
                        );
                      }
                    }}
                    onLoad={() => {
                      streamFrameLoadingRef.current = false;
                      if (streamWarning) setStreamWarning("");
                      if (streamErrorCount > 0) setStreamErrorCount(0);
                    }}
                  />
                ) : hasSelectedVideo ? (
                  <div className="stream-empty">Video selected. Click Detect to process frames.</div>
                ) : (
                  <div className="stream-empty">Upload a video to process frames.</div>
                )}
              </div>
              <div className="stream-controls">
                <button type="button" className="play-pause-btn" onClick={toggleVideo}>
                  {videoPlaying ? "⏸ Pause" : "▶ Play"}
                </button>
                <button type="button" className="zoom-btn" onClick={() => setVideoZoom((z) => Math.min(z + 0.25, 3))} title="Zoom in">🔍+</button>
                <button type="button" className="zoom-btn" onClick={() => setVideoZoom((z) => Math.max(z - 0.25, 0.5))} title="Zoom out">🔍-</button>
                <button type="button" className="zoom-btn" onClick={() => setVideoZoom(1)} title="Reset zoom">⟲</button>
                <span className="zoom-label">{Math.round(videoZoom * 100)}%</span>
              </div>
              <div className={`process-status status-${processStatus}`}>
                <span className="status-dot" />
                <span className="status-label">
                  {processStatus === "idle" && "Awaiting Video Upload"}
                  {processStatus === "uploading" && "Uploading Video..."}
                  {processStatus === "processing" && "Processing Detection..."}
                  {processStatus === "complete" && "Detection Complete"}
                  {processStatus === "error" && "Processing Failed"}
                </span>
              </div>
              <div className="stream-upload-row">
                <label className="upload-label">
                  Upload Video
                  <input
                    type="file"
                    accept="video/*"
                    onChange={(event) => setVideoFile(event.target.files?.[0] || null)}
                  />
                </label>
                <label className="cam-toggle-label">
                  <input type="checkbox" checked={multiCamMode} onChange={() => setMultiCamMode((v) => !v)} />
                  {" "}Multi-Cam
                </label>
                <button type="button" className="detect-btn" onClick={handleVideoUpload} disabled={loading}>
                  {loading ? "Detecting..." : "Detect"}
                </button>
              </div>
              <div className="cam-uploads">
                {["front", "right", "back", "left"].map((cam) => (
                  <label key={cam} className={camFiles[cam] ? "cam-upload-label has-file" : "cam-upload-label"}>
                    <span>{cam}</span>
                    <small>{camFiles[cam]?.name || "No file"}</small>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={(e) => {
                        const file = e.target.files?.[0] || null;
                        setCamFiles((prev) => ({ ...prev, [cam]: file }));
                        if (file) setMultiCamMode(true);
                      }}
                    />
                  </label>
                ))}
              </div>
              <div className="multi-upload-note">
                Upload 1–4 camera angles (front / right / back / left) of the same vehicle. All streams run in parallel — OCR fires immediately on each detection.
              </div>
              {ocrLog.length > 0 && (
                <div className="ocr-log-panel">
                  <div className="ocr-log-head">LIVE OCR LOG</div>
                  <div className="ocr-log-scroll">
                    {ocrLog.slice(-10).map((line, idx) => (
                      <div key={idx} className={`ocr-log-line ${line.includes('LOCKED') ? 'log-locked' : line.includes('VOTE') ? 'log-vote' : line.includes('BEST') ? 'log-best' : line.includes('orphan') ? 'log-orphan' : ''}`}>
                        {line}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </aside>
          </div>

          <section className="dashboard-table">
            <div className="dashboard-head">
              <h3>Real-Time Trucks Dashboard</h3>
              <div className="dashboard-controls">
                <div className="view-toggle">
                  <button
                    type="button"
                    className={dashboardView === "table" ? "toggle-btn active" : "toggle-btn"}
                    onClick={() => setDashboardView("table")}
                  >
                    Table
                  </button>
                  <button
                    type="button"
                    className={dashboardView === "list" ? "toggle-btn active" : "toggle-btn"}
                    onClick={() => setDashboardView("list")}
                  >
                    List
                  </button>
                  <button
                    type="button"
                    className={dashboardView === "gate" ? "toggle-btn active" : "toggle-btn"}
                    onClick={() => setDashboardView("gate")}
                  >
                    Gate
                  </button>
                  <button
                    type="button"
                    className={groupByField ? "toggle-btn active" : "toggle-btn"}
                    onClick={() => setGroupByField((v) => !v)}
                    title="Group rows that share the same license plate / container number / truck number"
                  >
                    Group
                  </button>
                </div>
                <div className="counts">
                  <span>Detected: {counts.detected}</span>
                  <span className="count-trailer">With Trailer: {counts.withTrailer}</span>
                  <span className="count-no-trailer">No Trailer: {counts.withoutTrailer}</span>
                  <span>Approved: {counts.approved}</span>
                  <span>Rejected: {counts.rejected}</span>
                </div>
              </div>
            </div>
            {dashboardView === "table" ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Status</th>
                      <th>Camera</th>
                      <th>Truck ID</th>
                      {groupByField && <th>Source Tracks</th>}
                      <th>Truck Class</th>
                      <th>Company Logo</th>
                      <th>Container No.</th>
                      <th>Side No.</th>
                      <th>Driver</th>
                      <th>License Plate</th>
                      <th>Other Info</th>
                      <th>Truck Company</th>
                      <th>Truck No.</th>
                      <th>OCR Conf</th>
                      <th>Conf Avg</th>
                      <th>Duration</th>
                      <th>First Seen</th>
                      <th>Last Seen</th>
                      <th>First Frame</th>
                      <th>Last Frame</th>
                      <th>BBox</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayRows.length === 0 && (
                      <tr>
                          <td colSpan={groupByField ? 21 : 20} className="empty-cell">
                            No detected trucks yet.
                        </td>
                      </tr>
                    )}
                    {displayRows.map((truck) => {
                      const fields = readTruckFields(truck);
                      const status = truck.review_status || "pending";
                      const sourceTracks = truck._sourceTracks || `T#${truck.track_id}`;
                      return (
                        <tr key={truck.id} className={status === "rejected" ? "row-rejected" : ""}>
                          <td>
                            <span className={`status-chip status-${status}`}>{status}</span>
                          </td>
                          <td>
                            {fields.camera
                              ? <span className="cam-source-chip">{fields.camera.toUpperCase()}</span>
                              : <span className="cam-source-chip cam-source-unknown">—</span>}
                          </td>
                          <td>{fields.trackId}</td>
                          {groupByField && (
                            <td>
                              <span className="source-tracks-cell" title={sourceTracks}>{sourceTracks}</span>
                            </td>
                          )}
                          <td><span className="cell-truncate" title={fields.truckClass}>{shortText(fields.truckClass, 24)}</span></td>
                          <td><span className="cell-truncate" title={fields.container_company_logo}>{shortText(fields.container_company_logo, 20)}{fields.fieldCameras?.container_company_logo && <span className={`field-cam-tag cam-${fields.fieldCameras.container_company_logo}`}>{fields.fieldCameras.container_company_logo[0].toUpperCase()}</span>}</span></td>
                          <td><span className="cell-truncate" title={fields.container_number}>{shortText(fields.container_number, 20)}{fields.fieldCameras?.container_number && <span className={`field-cam-tag cam-${fields.fieldCameras.container_number}`}>{fields.fieldCameras.container_number[0].toUpperCase()}</span>}</span></td>
                          <td><span className="cell-truncate" title={fields.container_side_no}>{shortText(fields.container_side_no, 20)}{fields.fieldCameras?.container_side_no && <span className={`field-cam-tag cam-${fields.fieldCameras.container_side_no}`}>{fields.fieldCameras.container_side_no[0].toUpperCase()}</span>}</span></td>
                          <td><span className="cell-truncate" title={fields.driver}>{shortText(fields.driver, 20)}{fields.fieldCameras?.driver && <span className={`field-cam-tag cam-${fields.fieldCameras.driver}`}>{fields.fieldCameras.driver[0].toUpperCase()}</span>}</span></td>
                          <td><span className="cell-truncate" title={fields.license_plate}>{shortText(fields.license_plate, 20)}{fields.fieldCameras?.license_plate && <span className={`field-cam-tag cam-${fields.fieldCameras.license_plate}`}>{fields.fieldCameras.license_plate[0].toUpperCase()}</span>}</span></td>
                          <td><span className="cell-truncate" title={fields.other_container_info}>{shortText(fields.other_container_info, 20)}{fields.fieldCameras?.other_container_info && <span className={`field-cam-tag cam-${fields.fieldCameras.other_container_info}`}>{fields.fieldCameras.other_container_info[0].toUpperCase()}</span>}</span></td>
                          <td><span className="cell-truncate" title={fields.truck_company}>{shortText(fields.truck_company, 20)}{fields.fieldCameras?.truck_company && <span className={`field-cam-tag cam-${fields.fieldCameras.truck_company}`}>{fields.fieldCameras.truck_company[0].toUpperCase()}</span>}</span></td>
                          <td><span className="cell-truncate" title={fields.truck_number}>{shortText(fields.truck_number, 20)}{fields.fieldCameras?.truck_number && <span className={`field-cam-tag cam-${fields.fieldCameras.truck_number}`}>{fields.fieldCameras.truck_number[0].toUpperCase()}</span>}</span></td>
                          <td>{fields.ocr_confidence}</td>
                          <td>{fields.confidence}</td>
                          <td>{fields.durationSec}</td>
                          <td>{fields.firstSeen}</td>
                          <td>{fields.lastSeen}</td>
                          <td>{fields.firstSeenFrame}</td>
                          <td>{fields.lastSeenFrame}</td>
                          <td><span className="cell-truncate" title={fields.bbox}>{shortText(fields.bbox, 30)}</span></td>
                          <td>
                            <div className="row-actions">
                              <button
                                type="button"
                                className="mini-approve"
                                onClick={() => handleApprove(truck.id)}
                                disabled={status === "rejected"}
                              >
                                Approve
                              </button>
                              <button
                                type="button"
                                className="mini-reject"
                                onClick={() => handleReject(truck.id)}
                                disabled={status === "rejected"}
                              >
                                Reject
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : dashboardView === "gate" ? (
              <div className="gate-view">
                {gateRows.length === 0 && <div className="empty-cell">No detected vehicles yet.</div>}
                <div className="gate-summary-bar">
                  <span className="gate-summary-item"><b>{gateRows.length}</b> vehicle{gateRows.length !== 1 ? "s" : ""}</span>
                  <span className="gate-summary-sep">|</span>
                  <span className="gate-summary-item trailer-yes"><b>{gateRows.filter(t => (t.truck_type || "").toLowerCase().includes("with_container")).length}</b> with trailer</span>
                  <span className="gate-summary-sep">|</span>
                  <span className="gate-summary-item trailer-no"><b>{gateRows.filter(t => !(t.truck_type || "").toLowerCase().includes("with_container")).length}</b> without trailer</span>
                </div>
                <div className="gate-cards">
                  {gateRows.map((truck, idx) => {
                    const fields = readTruckFields(truck);
                    const status = truck.review_status || "pending";
                    const fusion = truck.associated_info?._fusion;
                    const cameraMap = fusion?.source_track_ids || {};
                    const cameras = Object.keys(cameraMap);
                    const hasTrailer = (truck.truck_type || "").toLowerCase().includes("with_container");
                    const timeRange = `${formatTimeSec(truck.first_seen_time_sec)} – ${formatTimeSec(truck.last_seen_time_sec)}`;
                    const duration = truck.duration_sec != null ? `${Number(truck.duration_sec).toFixed(1)}s` : "—";
                    const matchConf = fusion?.match_confidence ?? fields.confidence;
                    return (
                      <article key={truck.id} className={`gate-card ${status === "rejected" ? "gate-card-rejected" : ""}`}>
                        <div className="gate-card-header">
                          <span className="gate-vehicle-num">Vehicle {idx + 1}</span>
                          <span className={`gate-type-badge ${hasTrailer ? "badge-trailer" : "badge-no-trailer"}`}>
                            {hasTrailer ? "WITH TRAILER" : "NO TRAILER"}
                          </span>
                          <span className="gate-time">{timeRange}</span>
                          {fields.camera && <span className="cam-source-chip">{fields.camera.toUpperCase()}</span>}
                          <span className={`status-chip status-${status}`}>{status}</span>
                        </div>
                        {(cameras.length > 0 || fields.camera) && (
                          <div className="gate-cameras">
                            <span className="gate-cameras-label">Source</span>
                            {cameras.length > 0
                              ? cameras.map((cam) => (
                                  <span key={cam} className="gate-cam-badge">{cam.toUpperCase()} <span className="gate-cam-track">T#{cameraMap[cam]}</span></span>
                                ))
                              : fields.camera
                                ? <span className="gate-cam-badge">{fields.camera.toUpperCase()} <span className="gate-cam-track">T#{truck.track_id}</span></span>
                                : null}
                            {truck._groupCount > 1 && cameras.length === 0 && !fields.camera && (
                              <span className="gate-cam-badge">{truck._sourceTracks}</span>
                            )}
                          </div>
                        )}
                        <div className="gate-fields">
                          {fields.license_plate !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">License Plate</span>
                              <span className="gate-field-value strong">{fields.license_plate}</span>
                            </div>
                          )}
                          {fields.container_number !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Container No.</span>
                              <span className="gate-field-value strong">{fields.container_number}</span>
                            </div>
                          )}
                          {fields.container_side_no !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Side No.</span>
                              <span className="gate-field-value">{fields.container_side_no}</span>
                            </div>
                          )}
                          {fields.truck_number !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Truck No.</span>
                              <span className="gate-field-value">{fields.truck_number}</span>
                            </div>
                          )}
                          {fields.truck_company !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Company</span>
                              <span className="gate-field-value">{fields.truck_company}</span>
                            </div>
                          )}
                          <div className="gate-field">
                            <span className="gate-field-label">Duration</span>
                            <span className="gate-field-value">{duration}</span>
                          </div>
                          <div className="gate-field">
                            <span className="gate-field-label">Confidence</span>
                            <span className="gate-field-value">{matchConf}</span>
                          </div>
                        </div>
                        <div className="row-actions">
                          <button type="button" className="mini-approve" onClick={() => handleApprove(truck.id)} disabled={status === "rejected"}>Approve</button>
                          <button type="button" className="mini-reject" onClick={() => handleReject(truck.id)} disabled={status === "rejected"}>Reject</button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="list-view">
                {detectedTrucks.length === 0 && <div className="empty-cell">No detected trucks yet.</div>}
                {detectedTrucks.map((truck) => {
                  const fields = readTruckFields(truck);
                  const status = truck.review_status || "pending";
                  return (
                    <article key={truck.id} className={`truck-card ${status === "rejected" ? "truck-card-rejected" : ""}`}>
                      <div className="truck-card-head">
                        <strong>Truck #{fields.trackId}</strong>
                        {fields.camera && <span className="cam-source-chip">{fields.camera.toUpperCase()}</span>}
                        <span className={`status-chip status-${status}`}>{status}</span>
                      </div>
                      <div className="truck-card-grid">
                        <p><b>source_camera:</b> {fields.camera || "—"}</p>
                        <p><b>truck_class:</b> {fields.truckClass}</p>
                        <p><b>truck_company:</b> {fields.truck_company}</p>
                        <p><b>driver:</b> {fields.driver}</p>
                        <p><b>license_plate:</b> {fields.license_plate}</p>
                        <p><b>truck_number:</b> {fields.truck_number}</p>
                        <p><b>container_company_logo:</b> {fields.container_company_logo}</p>
                        <p><b>container_number:</b> {fields.container_number}</p>
                        <p><b>container_side_no:</b> {fields.container_side_no}</p>
                        <p><b>other_container_info:</b> {fields.other_container_info}</p>
                        <p><b>ocr_confidence:</b> {fields.ocr_confidence}</p>
                        <p><b>confidence_avg:</b> {fields.confidence}</p>
                        <p><b>duration_sec:</b> {fields.durationSec}</p>
                        <p><b>last_bbox:</b> {shortText(fields.bbox, 40)}</p>
                      </div>
                      <div className="row-actions">
                        <button type="button" className="mini-approve" onClick={() => handleApprove(truck.id)} disabled={status === "rejected"}>Approve</button>
                        <button type="button" className="mini-reject" onClick={() => handleReject(truck.id)} disabled={status === "rejected"}>Reject</button>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
