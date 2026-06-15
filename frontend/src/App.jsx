import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchTruckRecords,
  finalizeMultiCamera,
  fetchFieldMedia,
  finalizeVideoTruckRun,
  getMultiCameraFrameUrl,
  getMultiCameraStatus,
  getOcrStatus,
  getVideoTruckRunFrameUrl,
  getVideoTruckRunStatus,
  setOcrToken,
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
  // Strip decorative markdown/OCR artifacts:
  //  - '#' and '$' (e.g. "# CA-40761" -> "CA-40761", "$$ 4 6 $$" -> "4 6")
  //  - middle-dot '·' that OCR substitutes for '-' (e.g. "CA·40761" -> "CA-40761")
  // then collapse runs of single digits separated by spaces ("4 6" -> "46")
  // and collapse remaining whitespace.
  let out = String(text).replace(/[#$]/g, "").replace(/·/g, "-").replace(/\s+/g, " ").trim();
  // Join space-separated single digits the OCR emits as math ("4 6 7" -> "467").
  out = out.replace(/(?<=\d)\s+(?=\d)/g, "");
  return out.trim();
}

const _GARBAGE_PHRASES = [
  "abstract grayscale", "abstract gray", "grayscale curved",
  "curved shape", "simple geometric", "no text or symbols",
  "no visible text", "no text", "not visible", "no symbols",
  "close-up of", "close up of", "photograph of",
  "background with", "metallic", "cylindrical",
  "image of", "picture of", "image shows",
];
// A real plate / container / truck-number token: 4-12 alphanumerics containing a digit.
const _ID_TOKEN = /[A-Za-z0-9]{4,12}/g;
function isGarbageOcr(text) {
  if (!text) return false;
  const str = String(text);
  const lower = str.toLowerCase();
  if (_GARBAGE_PHRASES.some(p => lower.includes(p))) return true;
  // Long text with no plausible identifier token (4-12 chars containing a digit) is a caption.
  if (str.length > 30) {
    const hasIdToken = (str.match(_ID_TOKEN) || []).some(tok => /\d/.test(tok));
    if (!hasIdToken) return true;
  }
  return false;
}

function readOcrFieldValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return isGarbageOcr(value) ? "—" : (cleanOcrText(value) || "—");
  if (typeof value === "object" && value.text) return isGarbageOcr(value.text) ? "—" : (cleanOcrText(value.text) || "—");
  return "—";
}

// The truck company is one of these real carriers (read by the rear best_V2 model).
// OCR often mangles it, so snap whatever was read to the closest one — but only
// when it is actually close, otherwise the cleaned text is shown as-is.
const KNOWN_TRUCK_COMPANIES = ["SEAPORT INTERNATIONAL", "AIR AND OCEANLAND INC"];

// A truck number uniquely identifies its carrier. The back model reads the number
// reliably even when the company text itself is unreadable, so map known numbers to
// their company and let that win. Edit this map as new trucks/carriers are added.
const TRUCK_NUMBER_TO_COMPANY = {
  "801552": "SEAPORT INTERNATIONAL",
  "463": "AIR AND OCEANLAND INC"
};

function companyFromTruckNumber(truckNumber) {
  if (!truckNumber || truckNumber === "—") return null;
  const digits = String(truckNumber).replace(/[^A-Za-z0-9]/g, "");
  if (!digits) return null;
  if (TRUCK_NUMBER_TO_COMPANY[digits]) return TRUCK_NUMBER_TO_COMPANY[digits];
  // Tolerate leading-zero / partial reads, e.g. "0801552" or "801552KY" -> 801552.
  for (const [num, company] of Object.entries(TRUCK_NUMBER_TO_COMPANY)) {
    if (digits === num || digits.endsWith(num) || digits.startsWith(num)) return company;
  }
  return null;
}

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
  const cleaned = cleanOcrText(String(value).trim());
  if (!cleaned) return "—";
  const lower = cleaned.toLowerCase();
  // Distinguishing tokens win immediately.
  if (lower.includes("seaport") || lower.includes("sea port")) return "SEAPORT INTERNATIONAL";
  if (lower.includes("oceanland") || lower.includes("ocean land") || lower.includes("oceanhub")) return "AIR AND OCEANLAND INC";
  // Otherwise pick the closest canonical name by edit distance, but only accept
  // it as a match when it is reasonably close — a far-off string (or a different
  // real carrier) is left as the cleaned OCR text rather than force-snapped.
  let best = null;
  let bestDist = Infinity;
  for (const cand of KNOWN_TRUCK_COMPANIES) {
    const d = levenshtein(lower, cand.toLowerCase());
    if (d < bestDist) {
      bestDist = d;
      best = cand;
    }
  }
  if (best && bestDist <= Math.ceil(best.length * 0.45)) return best;
  return cleaned.toUpperCase();
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
  // Constrain truck company to a known carrier name, then let the truck number —
  // the strongest identifier — decide the carrier when it is one we know (the back
  // model reads the number reliably even when the company text OCR is weak/empty).
  fieldTexts.truck_company = normalizeTruckCompany(fieldTexts.truck_company);
  const companyByNumber = companyFromTruckNumber(fieldTexts.truck_number);
  if (companyByNumber) fieldTexts.truck_company = companyByNumber;
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

// Human label for an OCR field key (e.g. "truck_number" -> "Truck No.").
const FIELD_LABELS = {
  container_company_logo: "Logo",
  container_number: "Container No.",
  container_side_no: "Side No.",
  driver: "Driver",
  license_plate: "Plate",
  other_container_info: "Other",
  truck_company: "Company",
  truck_number: "Truck No."
};

// Per-merged-truck breakdown of which camera saw it and WHEN (in that camera's own
// video time), plus which fields each camera read. Collapses to one row per camera.
const _INTERNAL_CAMERAS = new Set(["", "track_fragment", "demo", "track_fragments"]);
function readCameraObservations(truck) {
  const obs = truck?.associated_info?._camera_observations;
  if (!Array.isArray(obs)) return [];
  const byCam = new Map();
  for (const o of obs) {
    const cam = (o?.camera || "").toLowerCase();
    if (_INTERNAL_CAMERAS.has(cam)) continue;
    const first = o.first_seen_time_sec;
    const last = o.last_seen_time_sec;
    const prev = byCam.get(cam);
    if (!prev) {
      byCam.set(cam, { camera: cam, first, last, fields: new Set(o.detected_fields || []) });
    } else {
      if (first != null && (prev.first == null || first < prev.first)) prev.first = first;
      if (last != null && (prev.last == null || last > prev.last)) prev.last = last;
      for (const f of (o.detected_fields || [])) prev.fields.add(f);
    }
  }
  return [...byCam.values()]
    .sort((a, b) => (a.first ?? 0) - (b.first ?? 0))
    .map((c) => ({
      camera: c.camera,
      range: `${formatTimeSec(c.first)} – ${formatTimeSec(c.last)}`,
      fields: [...c.fields].map((f) => FIELD_LABELS[f] || f)
    }));
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
  const [cameraSessions, setCameraSessions] = useState([]);
  const [videoZoom, setVideoZoom] = useState(1);
  const [ocrStatus, setOcrStatus] = useState(null);
  const [expandedCam, setExpandedCam] = useState(null);
  const [tokenInput, setTokenInput] = useState("");
  const [tokenSaving, setTokenSaving] = useState(false);
  const [tokenMsg, setTokenMsg] = useState("");
  const [showAllLogs, setShowAllLogs] = useState(false);
  const [fieldReplay, setFieldReplay] = useState(null);
  const videoRef = useRef(null);
  const streamFrameLoadingRef = useRef(false);
  const lastRequestedFrameRef = useRef("");
  const hasSelectedVideo = Boolean(videoFile || Object.values(camFiles).some(Boolean));

  // Poll MinerU OCR connectivity so the operator always sees whether reads are
  // reaching the OCR service. Polls faster while a job is actively processing.
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const status = await getOcrStatus();
        if (!cancelled) setOcrStatus(status);
      } catch {
        if (!cancelled) {
          setOcrStatus({ provider: "MinerU", status: "unreachable", reachable: false });
        }
      }
    }
    check();
    const interval = processStatus === "processing" ? 8000 : 20000;
    const id = setInterval(check, interval);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [processStatus]);

  const ocrBadge = useMemo(() => {
    const s = ocrStatus?.status;
    if (!ocrStatus) return { cls: "checking", label: "OCR: checking…" };
    if (s === "connected") {
      const reading = (ocrStatus.ok_calls ?? 0) > 0;
      return { cls: "connected", label: reading ? "MinerU OCR: connected · reading" : "MinerU OCR: connected" };
    }
    if (s === "no_token") return { cls: "error", label: "MinerU OCR: no API token" };
    if (s === "unreachable") return { cls: "error", label: "OCR status: backend unreachable" };
    return { cls: "error", label: "MinerU OCR: NOT connected" };
  }, [ocrStatus]);

  async function saveToken() {
    const token = tokenInput.trim();
    if (!token) return;
    setTokenSaving(true);
    setTokenMsg("");
    try {
      const status = await setOcrToken(token);
      setOcrStatus(status);
      setTokenInput("");
      setTokenMsg(status?.status === "connected" ? "Token saved — MinerU connected." : "Token saved.");
    } catch (error) {
      setTokenMsg(error?.response?.data?.detail || error.message || "Failed to save token.");
    } finally {
      setTokenSaving(false);
    }
  }

  async function openFieldReplay(truck, fieldKey, fieldValue, label) {
    const trackId = truck?.track_id;
    if (trackId == null) return;
    if (!activeJobId) {
      setFieldReplay({
        open: true, loading: false, label, value: fieldValue, field: fieldKey,
        frames: [], idx: 0, playing: false,
        error: "Field clips are available for the run you process in this session."
      });
      return;
    }
    setFieldReplay({ open: true, loading: true, label, value: fieldValue, field: fieldKey, frames: [], idx: 0, playing: true, error: "" });
    try {
      const data = await fetchFieldMedia(activeJobId, trackId, fieldKey);
      const frames = data?.frames || [];
      setFieldReplay((prev) => (prev && prev.open
        ? { ...prev, loading: false, frames, idx: 0, playing: frames.length > 1, error: frames.length ? "" : "No capture was recorded for this field." }
        : prev));
    } catch (e) {
      setFieldReplay((prev) => (prev && prev.open
        ? { ...prev, loading: false, error: e?.response?.data?.detail || e.message || "Failed to load field clip." }
        : prev));
    }
  }

  // Auto-advance the per-field clip frames like a looping video.
  useEffect(() => {
    if (!fieldReplay?.open || !fieldReplay.playing || (fieldReplay.frames?.length ?? 0) < 2) return undefined;
    const id = setInterval(() => {
      setFieldReplay((prev) => {
        if (!prev || !prev.open || !prev.playing || (prev.frames?.length ?? 0) < 2) return prev;
        return { ...prev, idx: (prev.idx + 1) % prev.frames.length };
      });
    }, 450);
    return () => clearInterval(id);
  }, [fieldReplay?.open, fieldReplay?.playing, fieldReplay?.frames?.length]);

  // Table cell for an OCR field: the value is a clickable trigger that replays
  // the captured crops for that (truck, field).
  function ocrTd(truck, fields, key) {
    const val = fields[key];
    const cam = fields.fieldCameras?.[key];
    const hasVal = val && val !== "—";
    return (
      <td>
        <span className="cell-truncate" title={val}>
          {hasVal ? (
            <button
              type="button"
              className="field-clip-cell"
              onClick={() => openFieldReplay(truck, key, val, FIELD_LABELS[key] || key)}
              title="Play captured clip for this field"
            >
              {shortText(val, 20)}
            </button>
          ) : shortText(val, 20)}
          {cam && <span className={`field-cam-tag cam-${cam}`}>{cam[0].toUpperCase()}</span>}
        </span>
      </td>
    );
  }

  // Gate-card field value as a clickable clip trigger.
  function clipValue(truck, key, value, extraClass = "") {
    return (
      <button
        type="button"
        className={`gate-field-value field-clip-value ${extraClass}`}
        onClick={() => openFieldReplay(truck, key, value, FIELD_LABELS[key] || key)}
        title="Play captured clip for this field"
      >
        {value}
        <span className="field-clip-icon" aria-hidden>▶</span>
      </button>
    );
  }

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
    const maxRetries = 8;
    const maxCachedPolls = 15;
    const stallPolls = 300; // ~10 min of zero progress → give up
    let retries = 0;
    let cachedPolls = 0;
    let bestFrame = -1;
    let bestProgress = -1;
    let bestLogLen = -1;
    let pollsSinceProgress = 0;

    for (let i = 0; ; i += 1) {
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
      // Backend returns the full accumulated log each poll — REPLACE, don't append.
      if (status.ocr_log) setOcrLog(status.ocr_log.slice(-1000));

      if (status.json_snapshot) {
        try {
          const snap = JSON.parse(status.json_snapshot);
          mergeSnapshotRows(snap?.trucks || {});
          const camSessions = snap?.session?.camera_sessions;
          if (camSessions && typeof camSessions === "object") {
            setCameraSessions(Object.values(camSessions));
          }
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

      const logLen = status.ocr_log?.length ?? 0;
      const prog = status.progress ?? 0;
      if (latestFrame > bestFrame || prog > bestProgress || logLen > bestLogLen) {
        bestFrame = Math.max(bestFrame, latestFrame);
        bestProgress = Math.max(bestProgress, prog);
        bestLogLen = Math.max(bestLogLen, logLen);
        pollsSinceProgress = 0;
      } else if (++pollsSinceProgress >= stallPolls) {
        throw new Error("Video processing stalled — no progress for several minutes.");
      }
      await sleep(2000);
    }
  }

  async function pollMultiCameraUntilComplete(jobId) {
    const maxRetries = 8;
    // Run to completion as long as work keeps advancing. Only give up if there is
    // NO progress (no frame advance, no new logs, no higher %) for this many
    // consecutive polls — ~10 min at 2s — a genuine stall, not just a slow job.
    const stallPolls = 300;
    let retries = 0;
    let bestProgress = -1;
    let bestFrameSum = -1;
    let bestLogLen = -1;
    let pollsSinceProgress = 0;
    for (let i = 0; ; i += 1) {
      let status;
      try {
        status = await getMultiCameraStatus(jobId);
      } catch {
        retries += 1;
        if (retries <= maxRetries) { await sleep(2000); continue; }
        throw new Error("Multi-camera backend unreachable after retries.");
      }
      retries = 0;
      const progressPct = status.progress != null ? Math.round(status.progress * 100) : 0;
      setMessage(`Multi-camera ${jobId.slice(0, 8)} | ${status.state} | ${progressPct}%`);
      let frameSum = 0;
      for (const camStatus of (status.cameras || [])) {
        const latestFrame = camStatus.latest_frame_id ?? camStatus.frame_id ?? 0;
        frameSum += latestFrame;
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
      // Backend returns the full accumulated log each poll — REPLACE, don't append.
      if (status.ocr_log) setOcrLog(status.ocr_log.slice(-1000));
      if (status.json_snapshot) {
        try {
          const snap = JSON.parse(status.json_snapshot);
          mergeSnapshotRows(snap?.trucks || {});
          const camSessions = snap?.session?.camera_sessions;
          if (camSessions && typeof camSessions === "object") {
            setCameraSessions(Object.values(camSessions));
          }
        } catch { /* ignore */ }
      }
      if (status.state === "completed") return;
      if (status.state === "failed") throw new Error(status.message || "Multi-camera processing failed.");

      const logLen = status.ocr_log?.length ?? 0;
      const prog = status.progress ?? 0;
      if (prog > bestProgress || frameSum > bestFrameSum || logLen > bestLogLen) {
        bestProgress = Math.max(bestProgress, prog);
        bestFrameSum = Math.max(bestFrameSum, frameSum);
        bestLogLen = Math.max(bestLogLen, logLen);
        pollsSinceProgress = 0;
      } else if (++pollsSinceProgress >= stallPolls) {
        throw new Error("Multi-camera processing stalled — no progress for several minutes.");
      }
      await sleep(2000);
    }
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
      setCameraSessions([]);
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

          <section className="video-stage">
            <div className="video-main">
              <div className="stream-head">
                <span className="stream-title">LIVE MULTI-CAM STREAM</span>
                <span
                  className={`ocr-status ocr-${ocrBadge.cls}`}
                  title={ocrStatus?.last_error || `${ocrStatus?.provider || "MinerU"} OCR`}
                >
                  <span className="ocr-dot" />
                  {ocrBadge.label}
                </span>
              </div>
              <div className="stream-view stream-view-large">
                {selectedCameraCount >= 1 && Object.values(camStreamUrls).some(Boolean) ? (
                  expandedCam && camFiles[expandedCam] ? (
                    <div className="cam-expanded">
                      <button type="button" className="cam-back-btn" onClick={() => setExpandedCam(null)}>
                        ← All cameras
                      </button>
                      <span className="cam-label-badge">{expandedCam.toUpperCase()}</span>
                      {camStreamUrls[expandedCam] ? (
                        <img
                          src={camStreamUrls[expandedCam]}
                          alt={`${expandedCam} stream`}
                          className="stream-image"
                          style={{ transform: `scale(${videoZoom})`, transformOrigin: "center center" }}
                          onError={() => { streamFrameLoadingRef.current = false; }}
                          onLoad={() => { streamFrameLoadingRef.current = false; }}
                        />
                      ) : (
                        <div className="stream-empty cam-waiting">Waiting for {expandedCam}…</div>
                      )}
                    </div>
                  ) : (
                    <div className="multi-cam-grid">
                      {["front", "right", "back", "left"].map((cam) =>
                        camFiles[cam] ? (
                          <div key={cam} className="multi-cam-cell" onClick={() => setExpandedCam(cam)} title={`Expand ${cam}`}>
                            <span className="cam-label-badge">{cam.toUpperCase()}</span>
                            <span className="cam-expand-hint">⤢</span>
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
                  )
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
                  <div className="stream-empty">Upload one or more camera videos to begin.</div>
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
              {streamWarning && <div className="stream-warning">{streamWarning}</div>}
            </div>
            <aside className="video-rail">
              <div className="rail-title">CONTROLS</div>
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

              <div className="token-box">
                <div className="rail-subtitle">MinerU API token</div>
                <input
                  type="password"
                  className="token-input"
                  placeholder={ocrStatus?.token_configured ? "Token set — paste to replace" : "Paste MinerU token"}
                  value={tokenInput}
                  onChange={(e) => setTokenInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") saveToken(); }}
                />
                <button
                  type="button"
                  className="token-save-btn"
                  onClick={saveToken}
                  disabled={tokenSaving || !tokenInput.trim()}
                >
                  {tokenSaving ? "Saving…" : "Save token"}
                </button>
                {tokenMsg && <div className="token-msg">{tokenMsg}</div>}
              </div>

              {ocrLog.length > 0 && (
                <div className="ocr-log-panel">
                  <div className="ocr-log-head">
                    <span>OCR LOG · {ocrLog.length}</span>
                    <button
                      type="button"
                      className="log-toggle-btn"
                      onClick={() => setShowAllLogs((v) => !v)}
                    >
                      {showAllLogs ? "Show recent" : "Show full log"}
                    </button>
                  </div>
                  <div className={showAllLogs ? "ocr-log-scroll ocr-log-scroll-full" : "ocr-log-scroll"}>
                    {(showAllLogs ? ocrLog : ocrLog.slice(-12)).map((line, idx) => (
                      <div key={idx} className={`ocr-log-line ${line.includes('LOCKED') ? 'log-locked' : line.includes('VOTE') ? 'log-vote' : line.includes('BEST') ? 'log-best' : line.includes('orphan') ? 'log-orphan' : ''}`}>
                        {line}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </aside>
          </section>

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
            {cameraSessions.length > 0 && (
              <div className="camera-timeline-bar">
                <span className="camera-timeline-label">Camera feeds</span>
                {cameraSessions.map((cs) => (
                  <span key={cs.camera} className={`camera-timeline-item cam-border-${cs.camera}`}>
                    <b>{(cs.camera || "?").toUpperCase()}</b>
                    {cs.start_time ? <> · start {cs.start_time}</> : null}
                    {cs.processed_duration_sec != null ? <> · processed {formatTimeSec(cs.processed_duration_sec)}</> : null}
                    {cs.time_offset_sec ? <span className="camera-timeline-offset"> (offset {cs.time_offset_sec > 0 ? "+" : ""}{cs.time_offset_sec}s)</span> : null}
                  </span>
                ))}
              </div>
            )}
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
                      <th>Start (m:ss)</th>
                      <th>End (m:ss)</th>
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
                      const fusionLabels = truck?.associated_info?._fusion?.source_track_labels;
                      const sourceTracks = truck._sourceTracks
                        || (fusionLabels?.length ? fusionLabels.join(" + ") : null)
                        || `T#${truck.track_id}`;
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
                          {ocrTd(truck, fields, "container_company_logo")}
                          {ocrTd(truck, fields, "container_number")}
                          {ocrTd(truck, fields, "container_side_no")}
                          {ocrTd(truck, fields, "driver")}
                          {ocrTd(truck, fields, "license_plate")}
                          {ocrTd(truck, fields, "other_container_info")}
                          {ocrTd(truck, fields, "truck_company")}
                          {ocrTd(truck, fields, "truck_number")}
                          <td>{fields.ocr_confidence}</td>
                          <td>{fields.confidence}</td>
                          <td>{fields.durationSec}</td>
                          <td title={`${fields.firstSeen}s`}>{formatTimeSec(fields.firstSeen)}</td>
                          <td title={`${fields.lastSeen}s`}>{formatTimeSec(fields.lastSeen)}</td>
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
                    const camObs = readCameraObservations(truck);
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
                        {camObs.length > 0 ? (
                          <div className="gate-cam-timeline">
                            <span className="gate-cameras-label">Seen by camera</span>
                            {camObs.map((c) => (
                              <div key={c.camera} className="gate-cam-row">
                                <span className={`cam-source-chip cam-${c.camera}`}>{c.camera.toUpperCase()}</span>
                                <span className="gate-cam-time">{c.range}</span>
                                {c.fields.length > 0 && (
                                  <span className="gate-cam-fields">read: {c.fields.join(", ")}</span>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (cameras.length > 0 || fields.camera) ? (
                          <div className="gate-cameras">
                            <span className="gate-cameras-label">Source</span>
                            {cameras.length > 0
                              ? cameras.map((cam) => (
                                  <span key={cam} className="gate-cam-badge">{cam.toUpperCase()} <span className="gate-cam-track">T#{cameraMap[cam]}</span></span>
                                ))
                              : fields.camera
                                ? <span className="gate-cam-badge">{fields.camera.toUpperCase()} <span className="gate-cam-track">T#{truck.track_id}</span></span>
                                : null}
                          </div>
                        ) : null}
                        <div className="gate-fields">
                          {fields.license_plate !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">License Plate</span>
                              {clipValue(truck, "license_plate", fields.license_plate, "strong")}
                            </div>
                          )}
                          {fields.container_number !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Container No.</span>
                              {clipValue(truck, "container_number", fields.container_number, "strong")}
                            </div>
                          )}
                          {fields.container_side_no !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Side No.</span>
                              {clipValue(truck, "container_side_no", fields.container_side_no)}
                            </div>
                          )}
                          {fields.truck_number !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Truck No.</span>
                              {clipValue(truck, "truck_number", fields.truck_number)}
                            </div>
                          )}
                          {fields.truck_company !== "—" && (
                            <div className="gate-field">
                              <span className="gate-field-label">Company</span>
                              {clipValue(truck, "truck_company", fields.truck_company)}
                            </div>
                          )}
                          <div className="gate-field">
                            <span className="gate-field-label">Start Time</span>
                            <span className="gate-field-value strong">{formatTimeSec(truck.first_seen_time_sec)}</span>
                          </div>
                          <div className="gate-field">
                            <span className="gate-field-label">End Time</span>
                            <span className="gate-field-value strong">{formatTimeSec(truck.last_seen_time_sec)}</span>
                          </div>
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

      {fieldReplay?.open && (
        <>
          <div className="replay-scrim" onClick={() => setFieldReplay(null)} />
          <aside className="replay-drawer" role="dialog" aria-label="Field capture replay">
            <div className="replay-head">
              <div className="replay-title">
                <span className="replay-eyebrow">FIELD CAPTURE</span>
                <h3>{fieldReplay.label}</h3>
              </div>
              <button type="button" className="replay-close" onClick={() => setFieldReplay(null)} aria-label="Close">✕</button>
            </div>

            {fieldReplay.value && fieldReplay.value !== "—" && (
              <div className="replay-readout">
                <span className="replay-readout-label">Reading</span>
                <span className="replay-readout-value">{fieldReplay.value}</span>
              </div>
            )}

            <div className="replay-stage">
              {fieldReplay.loading ? (
                <div className="replay-empty">Loading capture…</div>
              ) : fieldReplay.error ? (
                <div className="replay-empty replay-error">{fieldReplay.error}</div>
              ) : fieldReplay.frames.length > 0 ? (
                <img
                  className="replay-frame"
                  src={fieldReplay.frames[fieldReplay.idx]}
                  alt={`${fieldReplay.label} frame ${fieldReplay.idx + 1}`}
                />
              ) : (
                <div className="replay-empty">No frames captured.</div>
              )}
            </div>

            {fieldReplay.frames.length > 0 && !fieldReplay.loading && (
              <div className="replay-controls">
                <button
                  type="button"
                  className="replay-play"
                  onClick={() => setFieldReplay((p) => ({ ...p, playing: !p.playing }))}
                  disabled={fieldReplay.frames.length < 2}
                >
                  {fieldReplay.playing ? "⏸" : "▶"}
                </button>
                <input
                  type="range"
                  className="replay-scrub"
                  min={0}
                  max={fieldReplay.frames.length - 1}
                  value={fieldReplay.idx}
                  onChange={(e) => setFieldReplay((p) => ({ ...p, idx: Number(e.target.value), playing: false }))}
                />
                <span className="replay-counter">{fieldReplay.idx + 1}/{fieldReplay.frames.length}</span>
              </div>
            )}
            <p className="replay-hint">Frames captured each time this field was read by OCR.</p>
          </aside>
        </>
      )}
    </div>
  );
}
