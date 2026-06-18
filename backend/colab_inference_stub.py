from __future__ import annotations
import random
import io
import json
import os
import queue
import re
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from ultralytics import YOLO

# ========================= CONFIGURATION =====================================
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/content/drive/MyDrive/vehicle_plate/output/runs/yolo26x_seg_phase2/weights/best.pt",
)
OUTPUT_DIR        = "/tmp/plateflow_outputs"
CONF              = 0.23
TRUCK_CONF_THRESH = 0.92
IOU_THRESH        = 0.50
MASK_ALPHA        = 0.35
INFER_WIDTH       = 960
FRAME_SKIP        = 0
USE_HALF          = True
OCR_WORKERS       = 3
SAVE_CROPS        = False
MIN_ASSOC_SCORE   = 0.10
MIN_TRUCK_AREA    = 40_000
JSON_SNAP_EVERY   = 30
OCR_RETRY_EVERY   = 45

# ── Association locking from snippet 1 ───────────────────────────────────────
LOCK_AFTER_N_FRAMES = 5  # Lock after N consecutive frames with same truck

MINERU_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3NzgwMDYzMyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3OTExMzIyNiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNWIxM2U3YjctN2FmNi00MzdjLThhZmEtMTIxNTRiMzQyOGQxIiwiZW1haWwiOiIiLCJleHAiOjE3ODY4ODkyMjZ9.gw3idlGC_R1ulaBBXCR_FtVszMw3y7jIbFBwQcjhgCbqVNJaoXTvtUp7GdvpMF117PPbzn1xrqm8YIybpxMN_Q"

CLASS_NAMES = [
    "container_company_logo",
    "container_number",
    "container_side_no",
    "driver",
    "license_plate",
    "other_container_info",
    "truck_company",
    "truck_number",
    "truck_with_container",
    "truck_without_container",
]

TRUCK_CLASSES = {"truck_with_container", "truck_without_container"}
OCR_CLASSES = {
    "container_number",
    "container_side_no",
    "license_plate",
    "truck_number",
    "container_company_logo",
    "truck_company",
}
TRUCK_WITH_CONTAINER_FIELDS = {
    "container_number",
    "container_side_no",
    "container_company_logo",
    "other_container_info",
}
TRUCK_WITHOUT_CONTAINER_FIELDS = {
    "license_plate",
    "truck_company",
    "truck_number",
}
CHILD_CLASSES = TRUCK_WITH_CONTAINER_FIELDS | TRUCK_WITHOUT_CONTAINER_FIELDS

random.seed(42)
PALETTE: dict[int, tuple[int, int, int]] = {
    i: tuple(random.randint(60, 230) for _ in range(3))   # type: ignore[assignment]
    for i in range(len(CLASS_NAMES))
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_CROPS:
    os.makedirs(os.path.join(OUTPUT_DIR, "ocr_crops"), exist_ok=True)

# ========================= APP + MODEL =======================================
app = FastAPI(title="PlateFlow Colab GPU API", version="3.5.0")

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE   = _select_device()
USE_HALF = USE_HALF if DEVICE == "cuda" else False
print("=" * 72, flush=True)
print(f"[PlateFlow] Loading model: {MODEL_PATH}", flush=True)
print(f"[PlateFlow] Torch version: {torch.__version__}", flush=True)
if DEVICE == "cuda":
    print(f"[PlateFlow] GPU enabled: CUDA ({torch.cuda.get_device_name(0)})", flush=True)
elif DEVICE == "mps":
    print("[PlateFlow] GPU enabled: Apple Metal/MPS", flush=True)
else:
    print("[PlateFlow] GPU not available: running on CPU", flush=True)
print(f"[PlateFlow] Inference device: {DEVICE} | half precision: {USE_HALF}", flush=True)
model    = YOLO(MODEL_PATH)
model.to(DEVICE)
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(_dummy, verbose=False, half=USE_HALF, device=DEVICE)
del _dummy
print("[PlateFlow] Model loaded and warmup inference completed.", flush=True)
print("=" * 72, flush=True)

# ========================= DATA STRUCTS ======================================
_jobs_lock = threading.Lock()
_ocr_queue: queue.Queue = queue.Queue()


@dataclass
class VideoJob:
    job_id: str
    state: str
    message: str
    progress: float = 0.0
    frame_id: int = 0
    total_frames: int = 0
    fps: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    temp_video_path: str | None = None
    latest_frame_jpeg: bytes | None = None
    ocr_cache: dict[str, str] = field(default_factory=dict)
    pending_keys: set[str] = field(default_factory=set)
    submitted_keys: set[str] = field(default_factory=set)
    ocr_log: list[str] = field(default_factory=list)
    json_snapshot: str = "{}"
    session_info: dict[str, Any] = field(default_factory=dict)
    registry: Any | None = None
    sticky_assoc: dict[int, tuple[int, str, str]] = field(default_factory=dict)
    
    # ── Association locking state (from snippet 1) ───────────────────────────
    # Permanent association map (locked after LOCK_AFTER_N_FRAMES)
    # key: child track_id, value: {"truck_tid": int, "truck_type": str, "field": str}
    child_to_truck: dict[int, dict[str, Any]] = field(default_factory=dict)
    
    # Voting buffer (temporary, discarded after lock)
    # key: child track_id, value: {"truck_tid": int, "truck_type": str, "field": str, "count": int}
    child_assoc_votes: dict[int, dict[str, Any]] = field(default_factory=dict)
    
    # Best confidence tracker per (cls_name, track_id)
    ocr_best_conf: dict[tuple[str, int], float] = field(default_factory=dict)


JOBS: dict[str, VideoJob] = {}


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        job = JOBS[job_id]
        for key, value in kwargs.items():
            setattr(job, key, value)


# ========================= SPATIAL HELPERS ===================================
def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x_a, y_a = max(a[0], b[0]), max(a[1], b[1])
    x_b, y_b = min(a[2], b[2]), min(a[3], b[3])
    i_w, i_h = max(0, x_b - x_a), max(0, y_b - y_a)
    if not i_w or not i_h:
        return 0.0
    inter = i_w * i_h
    union = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0


def _containment(child: tuple[int, int, int, int], parent: tuple[int, int, int, int]) -> float:
    x_a, y_a = max(child[0], parent[0]), max(child[1], parent[1])
    x_b, y_b = min(child[2], parent[2]), min(child[3], parent[3])
    i_w, i_h = max(0, x_b - x_a), max(0, y_b - y_a)
    if not i_w or not i_h:
        return 0.0
    child_area = (child[2] - child[0]) * (child[3] - child[1])
    return (i_w * i_h) / child_area if child_area > 0 else 0.0


def _update_association_votes(
    job: VideoJob,
    track_id: int,
    cls_name: str,
    truck_tid: int,
    truck_type: str
) -> None:
    """
    Called every frame a child detection has a valid spatial association.
    Accumulates votes. Locks the association after LOCK_AFTER_N_FRAMES
    consecutive agreements on the same truck.

    Flow:
        Frame 1: first seen → start vote (count=1)
        Frame 2: same truck → count=2
        Frame 3: different truck → RESET (count=1 for new truck)
        Frame 4: same truck as frame 3 → count=2
        ...
        Frame N: count reaches LOCK_AFTER_N_FRAMES → LOCKED permanently
    """
    # Already locked → nothing to do
    if track_id in job.child_to_truck:
        return

    vote = job.child_assoc_votes.get(track_id)

    if vote is None:
        # First time we see this child → start fresh vote
        job.child_assoc_votes[track_id] = {
            "truck_tid": truck_tid,
            "truck_type": truck_type,
            "field": cls_name,
            "count": 1,
        }
        print(f"  [VOTE] child T#{track_id} ({cls_name}) "
              f"→ Truck T#{truck_tid}  count=1")

    elif vote["truck_tid"] == truck_tid:
        # Same truck wins again → increment
        vote["count"] += 1
        print(f"  [VOTE] child T#{track_id} ({cls_name}) "
              f"→ Truck T#{truck_tid}  count={vote['count']}/{LOCK_AFTER_N_FRAMES}")

        if vote["count"] >= LOCK_AFTER_N_FRAMES:
            # ── LOCK the association permanently ─────────────────────────────
            job.child_to_truck[track_id] = {
                "truck_tid": truck_tid,
                "truck_type": truck_type,
                "field": cls_name,
            }
            del job.child_assoc_votes[track_id]
            print(f"  [LOCKED] ✅ child T#{track_id} ({cls_name}) "
                  f"→ Truck T#{truck_tid} ({truck_type}) "
                  f"after {LOCK_AFTER_N_FRAMES} frames")

    else:
        # Different truck won this frame → reset vote for new truck
        print(f"  [VOTE] child T#{track_id} ({cls_name}) "
              f"RESET: was T#{vote['truck_tid']}, now T#{truck_tid}")
        job.child_assoc_votes[track_id] = {
            "truck_tid": truck_tid,
            "truck_type": truck_type,
            "field": cls_name,
            "count": 1,
        }


def _build_association_map_integrated(
    boxes_data: list[tuple],
    job: VideoJob | None = None,
) -> dict[int, tuple[int, str, str]]:
    """
    Two-stage resolution with locking from snippet 1:
    1. First check if child is already LOCKED → always use locked truck
    2. Otherwise perform spatial association and update voting
    3. Fallback to sticky cache if spatial fails
    """
    truck_list: list[tuple[tuple, int, str]] = []
    child_list: list[tuple[int, tuple, str, int | None]] = []

    for i, (x1, y1, x2, y2, cls_id, _, tid, _mask) in enumerate(boxes_data):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        box = (x1, y1, x2, y2)
        if cls_name in TRUCK_CLASSES and tid is not None:
            truck_list.append((box, tid, cls_name))
        elif cls_name in CHILD_CLASSES:
            child_list.append((i, box, cls_name, tid))

    assoc: dict[int, tuple[int, str, str]] = {}

    for child_idx, child_box, child_cls, child_tid in child_list:
        # Skip if no track_id
        if child_tid is None:
            continue

        # Stage 0: Check if permanently locked
        if job is not None and child_tid in job.child_to_truck:
            locked = job.child_to_truck[child_tid]
            assoc[child_idx] = (locked["truck_tid"], locked["truck_type"], locked["field"])
            continue

        # Stage 1: Spatial association (only for unlocked children)
        best_score = 0.0
        best_match: tuple[int, str, str] | None = None
        for truck_box, truck_tid, truck_type in truck_list:
            score = max(_containment(child_box, truck_box), _iou(child_box, truck_box))
            if score > best_score:
                best_score = score
                best_match = (truck_tid, truck_type, child_cls)

        if best_match and best_score >= MIN_ASSOC_SCORE:
            assoc[child_idx] = best_match
            # Update voting for unlocked children
            if job is not None:
                _update_association_votes(job, child_tid, child_cls, best_match[0], best_match[1])

        # Stage 2: Sticky fallback (only if not locked and spatial failed)
        elif job is not None and child_tid in job.sticky_assoc:
            sticky = job.sticky_assoc[child_tid]
            assoc[child_idx] = sticky

    return assoc


# ========================= TEXT QUALITY ======================================
def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for c in text if c.isalnum() or c in " -/.")
    return good / len(text)


def _extract_mineru_image(md_text: str) -> str | None:
    """Extract the image path from Mineru markdown output."""
    m = re.search(r"!\[\]\(([^)]+)\)", md_text)
    return m.group(1) if m else None


# ========================= TRUCK REGISTRY ====================================
class TruckRegistry:
    _WITH_CONTAINER_TEMPLATE = {
        "container_number": None,
        "container_side_no": None,
        "container_company_logo": None,
        "other_container_info": None,
    }
    _WITHOUT_CONTAINER_TEMPLATE = {
        "license_plate": None,
        "truck_company": None,
        "truck_number": None,
    }

    def __init__(self, fps: float) -> None:
        self.fps = max(fps, 1.0)
        self.trucks: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _new_record(
        self,
        tid: int,
        cls_name: str,
        bbox: tuple[int, int, int, int],
        frame_id: int,
        conf: float,
    ) -> dict[str, Any]:
        template = (
            self._WITH_CONTAINER_TEMPLATE
            if cls_name == "truck_with_container"
            else self._WITHOUT_CONTAINER_TEMPLATE
        )
        return {
            "track_id": tid,
            "type": cls_name,
            "first_seen_frame": frame_id,
            "last_seen_frame": frame_id,
            "first_seen_time_sec": round(frame_id / self.fps, 3),
            "last_seen_time_sec": round(frame_id / self.fps, 3),
            "duration_frames": 1,
            "duration_sec": round(1.0 / self.fps, 3),
            "confidence_avg": round(conf, 4),
            "_conf_n": 1,
            "last_bbox": list(bbox),
            "associated_info": {k: v for k, v in template.items()},
        }

    def update(
        self,
        track_id: int,
        cls_name: str,
        bbox: tuple[int, int, int, int],
        frame_id: int,
        conf: float,
    ) -> None:
        with self._lock:
            if track_id not in self.trucks:
                self.trucks[track_id] = self._new_record(
                    track_id, cls_name, bbox, frame_id, conf
                )
                return
            rec = self.trucks[track_id]
            rec["last_seen_frame"] = frame_id
            rec["last_seen_time_sec"] = round(frame_id / self.fps, 3)
            rec["duration_frames"] = frame_id - rec["first_seen_frame"] + 1
            rec["duration_sec"] = round(rec["duration_frames"] / self.fps, 3)
            rec["last_bbox"] = list(bbox)
            n = rec["_conf_n"]
            rec["confidence_avg"] = round((rec["confidence_avg"] * n + conf) / (n + 1), 4)
            rec["_conf_n"] = n + 1

    def attach_ocr(
        self,
        truck_track_id: int,
        truck_type: str,
        field: str,
        text: str,
        conf: float = 0.0,
        image: str | None = None,
    ) -> None:
        if not text:
            return
        with self._lock:
            rec = self.trucks.get(truck_track_id)
            if rec is None:
                return
            info = rec["associated_info"]
            if field not in info:
                return
            existing = info.get(field)
            # Handle both old string format and new structured dict
            if existing is None or existing == "":
                info[field] = {"text": text, "confidence": round(conf, 4)}
                if image:
                    info[field]["image"] = image
                return
            if isinstance(existing, dict):
                existing_text = existing.get("text", "")
                existing_conf = existing.get("confidence", 0.0)
            else:
                existing_text = existing
                existing_conf = 0.0
            # Prefer higher confidence; tiebreak with text quality
            if conf > existing_conf:
                info[field] = {"text": text, "confidence": round(conf, 4)}
                if image:
                    info[field]["image"] = image
            elif conf == existing_conf:
                new_q = _text_quality(text)
                old_q = _text_quality(existing_text)
                if new_q > old_q or (new_q == old_q and len(text) < len(existing_text)):
                    info[field] = {"text": text, "confidence": round(conf, 4)}
                    if image:
                        info[field]["image"] = image

    def field_is_empty(self, truck_track_id: int, field: str) -> bool:
        """Return True if the truck exists and the field hasn't been filled yet."""
        with self._lock:
            rec = self.trucks.get(truck_track_id)
            if rec is None:
                return False
            val = rec["associated_info"].get(field)
            if val is None:
                return True
            if isinstance(val, dict):
                return not val.get("text")
            return not val

    def find_truck_for_field(self, field: str) -> int | None:
        with self._lock:
            best_tid: int | None = None
            best_frame: int = -1
            for tid, rec in self.trucks.items():
                info = rec["associated_info"]
                if field not in info:
                    continue
                existing = info.get(field)
                if isinstance(existing, dict):
                    existing_text = existing.get("text", "") or ""
                else:
                    existing_text = existing or ""
                last_f = rec.get("last_seen_frame", 0)
                if not existing_text and last_f > best_frame:
                    best_frame = last_f
                    best_tid = tid
            if best_tid is None:
                for tid, rec in self.trucks.items():
                    info   = rec["associated_info"]
                    last_f = rec.get("last_seen_frame", 0)
                    if field in info and last_f > best_frame:
                        best_frame = last_f
                        best_tid   = tid
            return best_tid

    def to_dict(self, session_info: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            trucks_out: dict[str, dict[str, Any]] = {}
            for tid, rec in self.trucks.items():
                clean = {k: v for k, v in rec.items() if not k.startswith("_")}
                clean["associated_info"] = dict(rec["associated_info"])
                trucks_out[str(tid)] = clean
        twc  = sum(1 for t in trucks_out.values() if t["type"] == "truck_with_container")
        twoc = sum(1 for t in trucks_out.values() if t["type"] == "truck_without_container")
        return {
            "session": session_info or {},
            "summary": {
                "total_trucks_tracked":    len(trucks_out),
                "trucks_with_container":   twc,
                "trucks_without_container": twoc,
            },
            "trucks": trucks_out,
        }


# ========================= OCR HELPERS =======================================
def _parse_mineru_markdown(md_text: str) -> str:
    details = re.findall(
        r"<details>.*?<summary>[^<]*</summary>\s*(.*?)\s*</details>",
        md_text,
        flags=re.DOTALL,
    )
    if details:
        combined = " | ".join(d.strip() for d in details if d.strip())
        if combined:
            return combined
    lines = [
        line.strip()
        for line in md_text.splitlines()
        if line.strip()
        and not line.strip().startswith("![")
        and not re.match(r"^<[^>]+>$", line.strip())
    ]
    return " ".join(lines).strip()


def _extract_md_from_zip(zip_url: str) -> str:
    try:
        resp = requests.get(zip_url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if name.endswith("full.md"):
                    return _parse_mineru_markdown(
                        zf.read(name).decode("utf-8", errors="ignore")
                    )
    except Exception:
        return ""
    return ""


def _call_mineru_ocr(crop_rgb: np.ndarray, max_retries: int = 2) -> str:
    if not MINERU_TOKEN:
        return ""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}",
    }
    _, enc = cv2.imencode(".png", crop_rgb)
    filename = f"crop_{uuid.uuid4().hex[:8]}.png"
    data_id  = uuid.uuid4().hex

    for attempt in range(max_retries):
        try:
            step1 = requests.post(
                "https://mineru.net/api/v4/file-urls/batch",
                json={
                    "files": [{"name": filename, "data_id": data_id}],
                    "model_version": "vlm",
                },
                headers=headers,
                timeout=15,
            )
            step1.raise_for_status()
            payload = step1.json()
            if payload.get("code") != 0:
                print(f"  ⚠️ MinerU: {payload.get('msg')}")
                continue
            batch_id   = payload["data"]["batch_id"]
            upload_url = payload["data"]["file_urls"][0]

            put_resp = requests.put(upload_url, data=enc.tobytes(), timeout=30)
            if put_resp.status_code not in (200, 201):
                print(f"  ⚠️ Upload failed: {put_resp.status_code}")
                continue

            poll_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
            for poll_i in range(20):
                time.sleep(3)
                try:
                    pr = requests.get(poll_url, headers=headers, timeout=15)
                    pr.raise_for_status()
                    results = pr.json().get("data", {}).get("extract_result", [])
                    if not results:
                        continue
                    state = results[0].get("state")
                    if state == "done":
                        return _extract_md_from_zip(results[0].get("full_zip_url", ""))
                    if state == "failed":
                        print(f"  ⚠️ MinerU failed: {results[0].get('err_msg')}")
                        return ""
                except requests.exceptions.Timeout:
                    print(f"  ⚠️ Poll #{poll_i} timed out — retrying…")
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"  ⚠️ Poll #{poll_i} error: {e}")
                    continue
            print(f"  ⚠️ MinerU: max polls reached (attempt {attempt + 1})")

        except requests.exceptions.Timeout:
            print(f"  ⚠️ MinerU S1/S2 timeout (attempt {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ MinerU request error: {e} (attempt {attempt + 1}/{max_retries})")

    return ""


# ========================= VISUAL HELPERS ====================================
def _resize_for_inference(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= INFER_WIDTH:
        return frame, 1.0
    scale = INFER_WIDTH / w
    return cv2.resize(frame, (INFER_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR), scale


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if h < 64:
        crop_bgr = cv2.resize(
            crop_bgr,
            (max(1, int(w * 64 / h)), 64),
            interpolation=cv2.INTER_CUBIC,
        )
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def _annotate_frame(
    frame: np.ndarray,
    boxes_data: list[tuple],
    assoc_map: dict[int, tuple[int, str, str]],
    ocr_map: dict[int, str],
    pending_track_ids: set[int],
    job: VideoJob | None = None,
) -> np.ndarray:
    H, W = frame.shape[:2]

    for i, (x1, y1, x2, y2, cls_id, conf_val, tid_raw, mask) in enumerate(boxes_data):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        colour   = PALETTE.get(cls_id, (88, 170, 255))
        tid_str  = f"#{tid_raw} " if tid_raw is not None else ""

        if mask is not None:
            m  = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            ov = frame.copy()
            ov[m > 0.5] = colour
            cv2.addWeighted(ov, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
            ctrs, _ = cv2.findContours(
                (m > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(frame, ctrs, -1, colour, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # ── Label: show lock status (from snippet 1) ───────────────────────────
        label = f"{tid_str}{cls_name} {conf_val:.2f}"
        if tid_raw is not None and cls_name in CHILD_CLASSES and job is not None:
            if tid_raw in job.child_to_truck:
                link = job.child_to_truck[tid_raw]
                label += f"  🔒→T#{link['truck_tid']}"
            elif tid_raw in job.child_assoc_votes:
                vote = job.child_assoc_votes[tid_raw]
                label += f"  ⏳{vote['count']}/{LOCK_AFTER_N_FRAMES}→T#{vote['truck_tid']}"
        elif i in assoc_map:
            truck_tid, _truck_type, _field = assoc_map[i]
            label += f"  →T#{truck_tid}"

        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - lh - bl - 4)), (x1 + lw, y1), colour, -1)
        cv2.putText(
            frame, label,
            (x1, max(lh, y1 - bl - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA,
        )

        if cls_name in OCR_CLASSES:
            if i in ocr_map and ocr_map[i]:
                txt   = ocr_map[i][:60]
                color = colour
            elif tid_raw in pending_track_ids:
                txt   = "scanning…"
                color = (200, 200, 50)
            else:
                continue

            txt_y = y2 + 22 if y2 + 42 < H else max(20, y1 - 10)
            cv2.putText(
                frame, txt, (x1 + 1, txt_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                frame, txt, (x1, txt_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
            )

    return frame


def _det_key(
    track_id: int | None,
    frame_id: int,
    det_idx: int,
    truck_tid: int | None = None,
    field: str | None = None,
) -> str:
    """
    Truck-field-scoped OCR key.
    When we know which truck and which field this crop belongs to, scope the
    key to (truck_tid, field) rather than the child's track_id.
    """
    if truck_tid is not None and field is not None:
        return f"truck_{truck_tid}_{field}"
    return f"t{track_id}" if track_id is not None else f"f{frame_id}_i{det_idx}"


def _enqueue_ocr(
    job_id: str,
    key: str,
    crop_rgb: np.ndarray,
    field: str,
    frame_id: int,
    truck_tid: int | None,
    truck_type: str | None,
    child_track_id: int | None = None,
    conf: float = 0.0,
) -> None:
    with _jobs_lock:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.submitted_keys.add(key)
        job.pending_keys.add(key)
    _ocr_queue.put((job_id, key, crop_rgb, field, frame_id, truck_tid, truck_type, child_track_id, conf))


def _ocr_worker() -> None:
    while True:
        item = _ocr_queue.get()
        if item is None:
            break
        job_id, key, crop_rgb, field, frame_id, truck_tid, truck_type, child_track_id, conf = item
        
        with _jobs_lock:
            job = JOBS.get(job_id)
            if job is None:
                _ocr_queue.task_done()
                continue
            
            # ── Look up permanent locked association if available ──────────────────
            if child_track_id is not None and child_track_id in job.child_to_truck:
                lock = job.child_to_truck[child_track_id]
                truck_tid = lock["truck_tid"]
                truck_type = lock["truck_type"]
                field = lock["field"]
                print(f"  [OCR] using LOCKED association: child T#{child_track_id} → T#{truck_tid}")
        
        # Perform OCR outside lock
        md_text = _call_mineru_ocr(crop_rgb)
        plain_text = _parse_mineru_markdown(md_text) if md_text else None
        image_path = _extract_mineru_image(md_text) if md_text else None
        
        with _jobs_lock:
            job = JOBS.get(job_id)
            if job is None:
                _ocr_queue.task_done()
                continue
            
            job.pending_keys.discard(key)
            if plain_text:
                job.ocr_cache[key] = plain_text
                log = f"[f{frame_id}] {field} → '{plain_text}'"

                resolved_tid = truck_tid
                resolved_type = truck_type or ""

                if job.registry is not None:
                    if resolved_tid is not None:
                        job.registry.attach_ocr(resolved_tid, resolved_type, field, plain_text, conf, image_path)
                        log += f"  (T#{resolved_tid})"
                    else:
                        candidate = job.registry.find_truck_for_field(field)
                        if candidate is not None:
                            resolved_tid = candidate
                            rec = job.registry.trucks.get(candidate)
                            resolved_type = rec["type"] if rec else ""
                            job.registry.attach_ocr(resolved_tid, resolved_type, field, plain_text, conf, image_path)
                            log += f"  (orphan → T#{resolved_tid})"
                            print(f"  [OCR] ↩ orphan '{field}' rescued → T#{resolved_tid}")
                        else:
                            log += "  (orphan — no compatible truck yet)"
                            print(f"  [OCR] ⚠ orphan '{field}' — no compatible truck, dropping")
                
                job.ocr_log.append(log)
                job.ocr_log = job.ocr_log[-60:]
                print(f"  [OCR] ✓ {key} → '{plain_text}'")
            else:
                # Empty result — remove from submitted so retry is possible
                job.submitted_keys.discard(key)
                print(f"  [OCR] ✗ {key} — will retry")
        
        _ocr_queue.task_done()


for _ in range(OCR_WORKERS):
    threading.Thread(target=_ocr_worker, daemon=True).start()


# ========================= PROCESSING ========================================
def _run_video_analysis(video_path: str, job: VideoJob | None = None) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    session_info: dict[str, Any] = {
        "video_path":   video_path,
        "total_frames": total_frames,
        "video_fps":    round(fps, 2),
        "resolution":   f"{w_orig}x{h_orig}",
        "started_at":   datetime.now().isoformat(timespec="seconds"),
        "device":       DEVICE,
        "model":        MODEL_PATH,
        "lock_after_n_frames": LOCK_AFTER_N_FRAMES,  # Record locking config
    }

    registry = TruckRegistry(fps=fps)
    if job is not None:
        _set_job(job.job_id, registry=registry, session_info=session_info, total_frames=total_frames)

    frame_id   = 0
    t_prev     = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            break

        # ── Frame-skip ────────────────────────────────────────────────────────
        if FRAME_SKIP > 0 and frame_id % (FRAME_SKIP + 1) != 0:
            small, _ = _resize_for_inference(frame_orig)
            try:
                model.track(
                    small, conf=CONF, iou=IOU_THRESH,
                    tracker="bytetrack.yaml", persist=True,
                    verbose=False, half=USE_HALF, device=DEVICE,
                )
            except Exception:
                pass
            frame_id += 1
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        frame_small, scale = _resize_for_inference(frame_orig)
        t0      = time.time()
        try:
            results = model.track(
                frame_small,
                conf=CONF,
                iou=IOU_THRESH,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                half=USE_HALF,
                device=DEVICE,
            )
        except Exception:
            results = None
        infer_ms = (time.time() - t0) * 1000

        boxes_data: list[tuple] = []
        ocr_map:    dict[int, str] = {}

        boxes_raw = results[0].boxes if results else None
        masks_raw = results[0].masks if results else None

        if boxes_raw is not None and len(boxes_raw) > 0:
            for i, box in enumerate(boxes_raw):
                sx1, sy1, sx2, sy2 = map(int, box.xyxy[0].tolist())
                x1 = max(0,      int(sx1 / scale))
                y1 = max(0,      int(sy1 / scale))
                x2 = min(w_orig, int(sx2 / scale))
                y2 = min(h_orig, int(sy2 / scale))
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                track_id = int(box.id[0]) if box.id is not None else None

                # Per-class confidence gates
                if cls_name in TRUCK_CLASSES:
                    if conf_val < TRUCK_CONF_THRESH:
                        continue
                    truck_area = (x2 - x1) * (y2 - y1)
                    if truck_area < MIN_TRUCK_AREA:
                        print(f"  [AREA] drop {cls_name} area={truck_area} < {MIN_TRUCK_AREA}")
                        continue

                if cls_name in TRUCK_CLASSES and masks_raw is not None and i < len(masks_raw.data):
                    mask = masks_raw.data[i].cpu().numpy()
                else:
                    mask = None

                boxes_data.append((x1, y1, x2, y2, cls_id, conf_val, track_id, mask))

                # Register truck in registry
                if cls_name in TRUCK_CLASSES and track_id is not None:
                    registry.update(track_id, cls_name, (x1, y1, x2, y2), frame_id, conf_val)

        # ── Spatial association with locking (from snippet 1) ───────────────────
        assoc_map = _build_association_map_integrated(boxes_data, job if job is not None else None)

        # ── Periodic retry for empty truck fields ───────────────────────────────
        if job is not None and frame_id % OCR_RETRY_EVERY == 0 and frame_id > 0:
            with _jobs_lock:
                for i, (x1, y1, x2, y2, cls_id, _conf, track_id, _mask) in enumerate(boxes_data):
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    if cls_name not in OCR_CLASSES or i not in assoc_map:
                        continue
                    truck_tid, truck_type, fld = assoc_map[i]
                    if registry.field_is_empty(truck_tid, fld):
                        retry_key = _det_key(track_id, frame_id, i, truck_tid, fld)
                        if retry_key in job.submitted_keys:
                            job.submitted_keys.discard(retry_key)
                            job.pending_keys.discard(retry_key)
                            print(f"  [RETRY] f{frame_id} T#{truck_tid} field='{fld}' unlocked for re-submission")

        # ── Snapshot current OCR state from the job ───────────────────────────
        if job is not None:
            with _jobs_lock:
                submitted = set(job.submitted_keys)
                cached    = dict(job.ocr_cache)
                pending   = set(job.pending_keys)
        else:
            submitted, cached, pending = set(), {}, set()

        # ── Build ocr_map + enqueue new crops with best-confidence tracking ────
        for i, (x1, y1, x2, y2, cls_id, conf_val, track_id, _mask) in enumerate(boxes_data):
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            if cls_name not in OCR_CLASSES:
                continue

            crop_bgr = frame_orig[y1:y2, x1:x2].copy()
            if crop_bgr.size == 0:
                continue

            if SAVE_CROPS:
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, "ocr_crops", f"f{frame_id:05d}_i{i}_{cls_name}.jpg"),
                    crop_bgr,
                )

            # ── Resolve truck association for this detection ───────────────────
            truck_tid, truck_type, field = None, None, cls_name
            if i in assoc_map:
                truck_tid, truck_type, field = assoc_map[i]

            # Use truck-field-scoped key when we have a truck association
            key = _det_key(track_id, frame_id, i, truck_tid, field)

            if key in cached:
                ocr_map[i] = cached[key]

            # Best-confidence tracking: always use the highest-confidence crop per (cls, track_id).
            # We check even when the key is already submitted so a better crop supersedes the old one.
            should_submit = False
            if job is not None:
                best_key = (cls_name, track_id) if track_id is not None else None
                if best_key:
                    with _jobs_lock:
                        prev_best = job.ocr_best_conf.get(best_key, 0.0)
                        if conf_val > prev_best:
                            job.ocr_best_conf[best_key] = conf_val
                            job.submitted_keys.discard(key)
                            job.pending_keys.discard(key)
                            should_submit = True
                            print(f"  [OCR-BEST] {cls_name} T#{track_id} conf={conf_val:.3f} > prev={prev_best:.3f}")
                elif key not in pending and key not in submitted:
                    should_submit = True

            if should_submit and job is not None:
                _enqueue_ocr(
                    job.job_id, key,
                    _preprocess_crop(crop_bgr),
                    field, frame_id, truck_tid, truck_type,
                    track_id,  # Pass child track_id for lock lookup
                    conf=conf_val,
                )

        # ── pending_track_ids for "scanning…" overlay ─────────────────────────
        pending_track_ids: set[int] = {
            int(k[1:]) for k in pending if k.startswith("t") and k[1:].isdigit()
        }

        # ── Gather HUD numbers including locking stats ────────────────────────
        n_cache   = len(cached)
        n_pending = len(pending)
        q_size    = _ocr_queue.qsize()
        n_trucks  = len(registry.trucks)
        pct       = int(frame_id / total_frames * 100) if total_frames > 0 else 0
        gpu_txt   = (
            f"GPU: {torch.cuda.memory_allocated() // 1024**2} MB"
            if DEVICE == "cuda" else "CPU mode"
        )
        
        # Get locking stats from job
        n_locked = len(job.child_to_truck) if job else 0
        n_voting = len(job.child_assoc_votes) if job else 0

        # ── Annotate ──────────────────────────────────────────────────────────
        annotated = _annotate_frame(
            frame_orig.copy(), boxes_data, assoc_map, ocr_map, pending_track_ids, job
        )

        cv2.putText(
            annotated,
            f"Frame {frame_id}/{total_frames} ({pct}%)  |  {fps_smooth:.1f} FPS  |  {infer_ms:.0f} ms",
            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Trucks tracked: {n_trucks}  |  OCR done: {n_cache}  pending: {n_pending}  "
            f"queue: {q_size}  |  {gpu_txt}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 1, cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Assoc locked: {n_locked}  voting: {n_voting}",
            (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA,
        )

        bar_w = int(w_orig * frame_id / total_frames) if total_frames > 0 else 0
        cv2.rectangle(annotated, (0, h_orig - 8), (w_orig, h_orig), (40, 40, 40), -1)
        cv2.rectangle(annotated, (0, h_orig - 8), (bar_w,  h_orig), (0, 220, 100), -1)

        ok, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok and job is not None:
            _set_job(job.job_id, latest_frame_jpeg=jpeg.tobytes())

        t_now      = time.time()
        fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev     = t_now

        frame_id += 1
        if job is not None:
            progress = (frame_id / total_frames) if total_frames > 0 else 0.0
            _set_job(
                job.job_id,
                progress=round(progress, 4),
                frame_id=frame_id,
                fps=round(fps_smooth, 2),
                message=(
                    f"Processing frame {frame_id}/{total_frames}  |  "
                    f"OCR done: {n_cache}  pending: {n_pending}  queue: {q_size}  |  "
                    f"Locked: {n_locked}  Voting: {n_voting}"
                ),
            )
            if frame_id % JSON_SNAP_EVERY == 0:
                _set_job(
                    job.job_id,
                    json_snapshot=json.dumps(registry.to_dict(session_info), indent=2),
                )

    cap.release()
    session_info["finished_at"]      = datetime.now().isoformat(timespec="seconds")
    session_info["frames_processed"] = frame_id
    final_data = registry.to_dict(session_info)
    if job is not None:
        _set_job(job.job_id, json_snapshot=json.dumps(final_data, indent=2))
    return final_data


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job        = JOBS[job_id]
        video_path = job.temp_video_path

    if not video_path:
        _set_job(job_id, state="failed", error="Missing temp video path", message="Job failed.")
        return

    try:
        _set_job(job_id, state="running", message="Video processing started.")
        result = _run_video_analysis(video_path, job=job)
        _set_job(job_id, state="completed", progress=1.0, result=result, message="Video processing completed.")
    except Exception as exc:
        _set_job(job_id, state="failed", error=str(exc), message=f"Video processing failed: {exc}")
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


# ========================= ENDPOINTS =========================================
@app.post("/infer")
async def infer(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / (file.filename or "upload.jpg")
        image_path.write_bytes(raw)
        results = model.predict(str(image_path), verbose=False, conf=CONF, iou=IOU_THRESH)
        detections = []
        if results:
            names = results[0].names or {}
            for box in results[0].boxes:
                xyxy     = box.xyxy[0].tolist()
                cls_id   = int(box.cls[0].item()) if box.cls is not None else -1
                conf_val = float(box.conf[0].item()) if box.conf is not None else 0.0
                cls_name = names.get(cls_id, f"class_{cls_id}")

                if cls_name in TRUCK_CLASSES and conf_val < TRUCK_CONF_THRESH:
                    continue

                detections.append({
                    "plate_text": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": {
                        "x1": round(xyxy[0], 2), "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2), "y2": round(xyxy[3], 2),
                    },
                })
    return {"detections": detections, "filename": file.filename}


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty video upload.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / (file.filename or "upload.mp4")
        path.write_bytes(raw)
        return _run_video_analysis(str(path), job=None)


@app.post("/analyze-video/start")
async def analyze_video_start(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty video upload.")

    job_id     = uuid.uuid4().hex
    tmp_dir    = tempfile.mkdtemp(prefix="plateflow_job_")
    video_path = Path(tmp_dir) / (file.filename or "upload.mp4")
    video_path.write_bytes(raw)

    with _jobs_lock:
        JOBS[job_id] = VideoJob(
            job_id=job_id,
            state="queued",
            message="Job queued.",
            temp_video_path=str(video_path),
        )
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "state": "queued", "message": "Video job created."}


@app.get("/analyze-video/jobs/{job_id}")
def analyze_video_job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id":        job.job_id,
        "state":         job.state,
        "progress":      job.progress,
        "frame_id":      job.frame_id,
        "total_frames":  job.total_frames,
        "fps":           job.fps,
        "message":       job.message,
        "error":         job.error,
        "ocr_log":       job.ocr_log[-20:],
        "json_snapshot": job.json_snapshot,
    }


@app.get("/analyze-video/jobs/{job_id}/result")
def analyze_video_job_result(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.state == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Job failed.")
    if job.state != "completed" or job.result is None:
        raise HTTPException(status_code=409, detail="Job is not completed yet.")
    return {"job_id": job.job_id, "state": job.state, "result": job.result}


@app.get("/analyze-video/jobs/{job_id}/frame")
def analyze_video_job_frame(job_id: str) -> Response:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.latest_frame_jpeg:
        raise HTTPException(status_code=404, detail="No frame available yet.")
    return Response(content=job.latest_frame_jpeg, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
