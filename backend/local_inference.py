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
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# ========================= CONFIGURATION =====================================
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)

def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, minimum)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent / "model" / "V4.pt"),
)
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "plateflow_outputs"),
)
MODEL_PATH_BACK = os.getenv(
    "MODEL_PATH_BACK",
    str(Path(__file__).resolve().parent / "model" / "V4.pt"),
)
OUTPUT_DIR_BACK = os.getenv(
    "OUTPUT_DIR_BACK",
    str(Path(__file__).resolve().parent / "plateflow_outputs_back"),
)
BACK_CAMERA_LABELS = {"back", "rear"}
CONF = _env_float("CONF", 0.73)
TRUCK_CONF_THRESH = _env_float("TRUCK_CONF_THRESH", 0.80)
IOU_THRESH = 0.50
IOU_MERGE_THRESH = _env_float("IOU_MERGE_THRESH", 0.50)        # ← raised from 0.10
MASK_ALPHA = 0.65
INFER_WIDTH = _env_int("INFER_WIDTH", 1280)
ENABLE_FRAME_SKIP = _env_bool("ENABLE_FRAME_SKIP", False)
PROCESS_EVERY_N_FRAMES = _env_int("PROCESS_EVERY_N_FRAMES", 1)
FRAME_SKIP = PROCESS_EVERY_N_FRAMES - 1 if ENABLE_FRAME_SKIP else 0
OCR_EVERY_N_FRAMES = _env_int("OCR_EVERY_N_FRAMES", 1)
# Only crops whose detection confidence clears this bar are sent to OCR. This is
# a SEPARATE, stricter gate than the global detection CONF — a box can be drawn /
# tracked at CONF but is only read by OCR when it is at least this confident.
OCR_CONF_THRESH = _env_float("OCR_CONF_THRESH", 0.75)
USE_HALF = True
OCR_WORKERS = 6
SAVE_CROPS = False
SAVE_DETECTION_CROPS = _env_bool("SAVE_DETECTION_CROPS", True)
MIN_ASSOC_SCORE = 0.60

# ================ FIX: Lowered area thresholds ===============================
MIN_TRUCK_AREA = _env_int("MIN_TRUCK_AREA", 15000)           # was 40000
MIN_TRUCK_TRACK_FRAMES = _env_int("MIN_TRUCK_TRACK_FRAMES", 10)
MIN_TRUCK_MASK_AREA = _env_float("MIN_TRUCK_MASK_AREA", 10000)  # lowered accordingly

JSON_SNAP_EVERY = 10
OCR_RETRY_EVERY = 45
LOCK_AFTER_N_FRAMES = 5

# ================ FIX: Greatly reduced active window and raised IoU thresholds
ACTIVE_WINDOW_FRAMES = _env_int("ACTIVE_WINDOW_FRAMES", 60)      # was 600
HIGH_IOU_MERGE_THRESH = _env_float("HIGH_IOU_MERGE_THRESH", 0.95) # was 0.50
# Two truck records that occupy the same spatial region (high union-bbox overlap)
# but are separated by more than this many frames are DIFFERENT physical trucks
# (e.g. a 2-min truck and a 7-min truck both driving through the same gate lane).
# Default: 150 frames = 5 s at 30 fps.  Set MIN_INTER_TRUCK_GAP_FRAMES=0 to disable.
MIN_INTER_TRUCK_GAP_FRAMES = _env_int("MIN_INTER_TRUCK_GAP_FRAMES", 150)

# ========================= PER-CAMERA DETECTION STATS =======================
@dataclass
class CameraDetectionStats:
    camera: str
    yolo_with_container: int = 0
    yolo_without_container: int = 0
    bytetrack_ids_with_container: set = field(default_factory=set)
    bytetrack_ids_without_container: set = field(default_factory=set)

    def to_dict(self) -> "dict[str, Any]":
        return {
            "camera": self.camera,
            "yolo_raw_detections": {
                "truck_with_container": self.yolo_with_container,
                "truck_without_container": self.yolo_without_container,
                "total": self.yolo_with_container + self.yolo_without_container,
            },
            "bytetrack_unique_ids": {
                "truck_with_container": len(self.bytetrack_ids_with_container),
                "truck_without_container": len(self.bytetrack_ids_without_container),
                "total": len(self.bytetrack_ids_with_container | self.bytetrack_ids_without_container),
            },
        }


MINERU_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3NzgwMDYzMyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3OTExMzIyNiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiNWIxM2U3YjctN2FmNi00MzdjLThhZmEtMTIxNTRiMzQyOGQxIiwiZW1haWwiOiIiLCJleHAiOjE3ODY4ODkyMjZ9.gw3idlGC_R1ulaBBXCR_FtVszMw3y7jIbFBwQcjhgCbqVNJaoXTvtUp7GdvpMF117PPbzn1xrqm8YIybpxMN_Q"

# Operators can paste a fresh MinerU token from the UI; it is persisted here and
# reloaded on startup so it survives restarts (env var MINERU_TOKEN still wins on
# first boot if set). Kept out of git via .gitignore.
_MINERU_TOKEN_FILE = str(Path(__file__).resolve().parent / ".mineru_token")

if os.getenv("MINERU_TOKEN"):
    MINERU_TOKEN = os.getenv("MINERU_TOKEN")
else:
    try:
        if os.path.exists(_MINERU_TOKEN_FILE):
            with open(_MINERU_TOKEN_FILE, "r", encoding="utf-8") as _tf:
                _saved = _tf.read().strip()
            if _saved:
                MINERU_TOKEN = _saved
                print("[PlateFlow] Loaded MinerU token from .mineru_token")
    except Exception as _exc:  # noqa: BLE001
        print(f"[PlateFlow] Could not read persisted MinerU token: {_exc}")


def set_mineru_token(token: str) -> bool:
    """Set the MinerU token at runtime (used by the OCR engine immediately) and
    persist it so it survives a restart. Returns True on success."""
    global MINERU_TOKEN
    token = (token or "").strip()
    if not token:
        return False
    MINERU_TOKEN = token
    try:
        with open(_MINERU_TOKEN_FILE, "w", encoding="utf-8") as tf:
            tf.write(token)
    except Exception as exc:  # noqa: BLE001
        print(f"[PlateFlow] Could not persist MinerU token: {exc}")
    return True


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
    i: tuple(random.randint(60, 230) for _ in range(3))
    for i in range(len(CLASS_NAMES))
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_CROPS:
    os.makedirs(os.path.join(OUTPUT_DIR, "ocr_crops"), exist_ok=True)
if SAVE_DETECTION_CROPS:
    os.makedirs(os.path.join(OUTPUT_DIR, "detections"), exist_ok=True)

# ================ FIX: ByteTrack config – more permissive matching, longer buffer
_TRACKER_CFG_PATH = os.path.join(OUTPUT_DIR, "custom_bytetrack.yaml")
with open(_TRACKER_CFG_PATH, "w") as _f:
    _f.write(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.60\n"
        "track_low_thresh: 0.33\n"
        "new_track_thresh: 0.50\n"
        "track_buffer: 120\n"           # was 90 – 4 seconds at 30 fps
        "match_thresh: 0.55\n"          # was 0.60 – easier re‑association
        "fuse_score: True\n"
    )
YOLO_TRACKER = _TRACKER_CFG_PATH

# ========================= APP + MODEL =======================================
app = FastAPI(title="PlateFlow Colab GPU API", version="3.6.0")

def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()
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
print(
    f"[PlateFlow] Detection config: conf={CONF} truck_conf={TRUCK_CONF_THRESH} "
    f"infer_width={INFER_WIDTH} tracker={YOLO_TRACKER} iou_merge={IOU_MERGE_THRESH}",
    flush=True,
)
print(
    f"[PlateFlow] Tracking stability: active_window={ACTIVE_WINDOW_FRAMES} frames "
    f"high_iou_merge={HIGH_IOU_MERGE_THRESH} min_track_frames={MIN_TRUCK_TRACK_FRAMES}",
    flush=True,
)
print(f"[PlateFlow] Minimum truck area: {MIN_TRUCK_AREA} pixels", flush=True)
if ENABLE_FRAME_SKIP:
    print(
        f"[PlateFlow] Frame skip enabled: processing every {PROCESS_EVERY_N_FRAMES} frame(s)",
        flush=True,
    )
else:
    print("[PlateFlow] Frame skip disabled: processing every frame", flush=True)
print(f"[PlateFlow] OCR throttle: reading text every {OCR_EVERY_N_FRAMES} frame(s)", flush=True)
print(f"[PlateFlow] Save detection crops: {SAVE_DETECTION_CROPS} → {OUTPUT_DIR}/detections", flush=True)
print(f"[PlateFlow] Ghost-track filter: min {MIN_TRUCK_TRACK_FRAMES} frames before truck appears in output", flush=True)

model = YOLO(MODEL_PATH)
model.to(DEVICE)
try:
    CLASS_NAMES = [model.names[i] for i in range(len(model.names))]
    print(f"[PlateFlow] Class names from model: {CLASS_NAMES}", flush=True)
except Exception as _exc:
    print(f"[PlateFlow] Could not read model.names ({_exc}); using fallback CLASS_NAMES", flush=True)

os.makedirs(OUTPUT_DIR_BACK, exist_ok=True)
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(_dummy, verbose=False, half=USE_HALF, device=DEVICE)
del _dummy
print("[PlateFlow] Model loaded and warmup inference completed.", flush=True)
print("=" * 72, flush=True)

# ========================= DATA STRUCTS ======================================
_jobs_lock = threading.Lock()
_ocr_queue: queue.Queue = queue.Queue()

def _load_tracking_model(model_path: str = MODEL_PATH) -> tuple[YOLO, list[str]]:
    tracking_model = YOLO(model_path)
    tracking_model.to(DEVICE)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    tracking_model.predict(dummy, verbose=False, half=USE_HALF, device=DEVICE)
    try:
        class_names = [tracking_model.names[i] for i in range(len(tracking_model.names))]
    except Exception:
        class_names = list(CLASS_NAMES)
    return tracking_model, class_names

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
    child_to_truck: dict[int, dict[str, Any]] = field(default_factory=dict)
    child_assoc_votes: dict[int, dict[tuple, int]] = field(default_factory=dict)
    raw_to_perm_id: dict[int, int] = field(default_factory=dict)
    _next_perm_id: int = 1
    ocr_best_conf: dict[tuple[str, int], float] = field(default_factory=dict)
    orphan_buffer: list[dict[str, Any]] = field(default_factory=list)
    camera_source: str = ""

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

def _mask_area(mask: Any, box: tuple[int, int, int, int]) -> float:
    if mask is not None:
        try:
            return float((mask > 0.5).sum())
        except Exception:
            pass
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))

def _dedupe_truck_detections(boxes_data: list[tuple], class_names: list[str] | None = None) -> list[tuple]:
    class_names = class_names or CLASS_NAMES
    truck_indices: list[int] = []
    keep = [True] * len(boxes_data)
    for i, (x1, y1, x2, y2, cls_id, _conf, _tid, _mask) in enumerate(boxes_data):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        if cls_name in TRUCK_CLASSES:
            truck_indices.append(i)

    for pos, i in enumerate(truck_indices):
        if not keep[i]:
            continue
        a = boxes_data[i]
        a_box = (a[0], a[1], a[2], a[3])
        a_area = _mask_area(a[7], a_box)
        for j in truck_indices[pos + 1:]:
            if not keep[j]:
                continue
            b = boxes_data[j]
            b_box = (b[0], b[1], b[2], b[3])
            overlap = max(_iou(a_box, b_box), _containment(a_box, b_box), _containment(b_box, a_box))
            if overlap < IOU_MERGE_THRESH:
                continue
            b_area = _mask_area(b[7], b_box)
            drop_idx, drop_area = (j, b_area) if a_area >= b_area else (i, a_area)
            keep[drop_idx] = False
            dropped = boxes_data[drop_idx]
            print(
                f"  [TRUCK-DEDUPE] drop T#{dropped[6]} area={drop_area:.0f} "
                f"overlap={overlap:.2f}; kept larger truck segment"
            )
            if drop_idx == i:
                break
    return [box for idx, box in enumerate(boxes_data) if keep[idx]]

def _update_association_votes(
    job: VideoJob,
    track_id: int,
    cls_name: str,
    truck_tid: int,
    truck_type: str,
) -> None:
    if track_id in job.child_to_truck:
        return
    votes = job.child_assoc_votes.setdefault(track_id, {})
    key = (truck_tid, truck_type)
    votes[key] = votes.get(key, 0) + 1
    best_key = max(votes, key=lambda k: votes[k])
    best_count = votes[best_key]
    print(f"  [VOTE] child T#{track_id} ({cls_name}) → Truck T#{best_key[0]}  votes={best_count}/{LOCK_AFTER_N_FRAMES}")
    if best_count >= LOCK_AFTER_N_FRAMES:
        job.child_to_truck[track_id] = {
            "truck_tid": best_key[0],
            "truck_type": best_key[1],
            "field": cls_name,
        }
        del job.child_assoc_votes[track_id]
        print(f"  [LOCKED] ✅ child T#{track_id} ({cls_name}) → Truck T#{best_key[0]} ({best_key[1]}) after majority vote")

def _build_association_map_integrated(
    boxes_data: list[tuple],
    job: VideoJob | None = None,
    class_names: list[str] | None = None,
) -> dict[int, tuple[int, str, str]]:
    class_names = class_names or CLASS_NAMES
    truck_list: list[tuple[tuple, tuple, int, str]] = []
    child_list: list[tuple[int, tuple, str, int | None]] = []

    for i, (x1, y1, x2, y2, cls_id, _, tid, _mask) in enumerate(boxes_data):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        box = (x1, y1, x2, y2)
        if cls_name in TRUCK_CLASSES and tid is not None:
            union_box = box
            if job is not None and job.registry is not None:
                rec = job.registry.trucks.get(tid)
                if rec:
                    ub = rec.get("union_bbox")
                    if ub:
                        union_box = tuple(ub)
            truck_list.append((box, union_box, tid, cls_name))
        elif cls_name in CHILD_CLASSES:
            child_list.append((i, box, cls_name, tid))

    assoc: dict[int, tuple[int, str, str]] = {}

    for child_idx, child_box, child_cls, child_tid in child_list:
        if child_tid is None:
            continue
        if job is not None and child_tid in job.child_to_truck:
            locked = job.child_to_truck[child_tid]
            assoc[child_idx] = (locked["truck_tid"], locked["truck_type"], locked["field"])
            continue

        best_score = 0.0
        best_match: tuple[int, str, str] | None = None
        for truck_box, union_box, truck_tid, truck_type in truck_list:
            score = max(
                _containment(child_box, truck_box), _iou(child_box, truck_box),
                _containment(child_box, union_box), _iou(child_box, union_box),
            )
            if score > best_score:
                best_score = score
                best_match = (truck_tid, truck_type, child_cls)

        if best_match and best_score >= MIN_ASSOC_SCORE:
            assoc[child_idx] = best_match
            if job is not None:
                _update_association_votes(job, child_tid, child_cls, best_match[0], best_match[1])
        elif job is not None and child_tid in job.sticky_assoc:
            sticky = job.sticky_assoc[child_tid]
            assoc[child_idx] = sticky

    return assoc

# ========================= PERMANENT TRUCK ID (FIXED) =========================
def _get_perm_truck_id(
    job: VideoJob,
    raw_id: int,
    bbox: tuple[int, int, int, int],
    registry: "TruckRegistry",
    frame_id: int,
) -> int:
    """
    Map a raw ByteTrack ID to a stable permanent ID.

    FIXES:
    - Hard merge (Path B) now requires IoU >= HIGH_IOU_MERGE_THRESH (0.95)
      AND the existing permanent truck must have been seen within the last
      30 frames (1 second at 30 fps).  This prevents merging a truck that
      left long ago with a newly arriving truck.
    - Soft merge (Path C) uses the (reduced) ACTIVE_WINDOW_FRAMES (60)
      and the raised IOU_MERGE_THRESH (0.65).
    """
    if raw_id in job.raw_to_perm_id:
        return job.raw_to_perm_id[raw_id]

    best_perm_hard: int | None = None
    best_score_hard = 0.0
    best_age_hard = None

    best_perm_soft: int | None = None
    best_score_soft = 0.0

    with registry._lock:
        for perm_id, rec in registry.trucks.items():
            lb = tuple(rec["last_bbox"])
            ub = tuple(rec["union_bbox"])
            score = max(
                _iou(bbox, lb), _containment(bbox, lb), _containment(lb, bbox),
                _iou(bbox, ub), _containment(bbox, ub), _containment(ub, bbox),
            )
            age = frame_id - rec.get("last_seen_frame", 0)

            # Path B – hard merge: extremely high IoU AND very recent
            if score >= HIGH_IOU_MERGE_THRESH:
                if best_score_hard < score:
                    best_score_hard = score
                    best_perm_hard = perm_id
                    best_age_hard = age

            # Path C – soft merge: within active window, lower IoU
            if age <= ACTIVE_WINDOW_FRAMES and score >= IOU_MERGE_THRESH:
                if score > best_score_soft:
                    best_score_soft = score
                    best_perm_soft = perm_id

    # Hard merge only if age is ≤ 30 frames (1 second) – prevents merging sequential trucks
    if best_perm_hard is not None and best_score_hard >= HIGH_IOU_MERGE_THRESH and best_age_hard <= 100:
        job.raw_to_perm_id[raw_id] = best_perm_hard
        print(
            f"  [PERM-ID] raw T#{raw_id} HARD-merged → perm T#{best_perm_hard} "
            f"(score={best_score_hard:.2f}, age={best_age_hard} frames)"
        )
        return best_perm_hard
    elif best_perm_hard is not None:
        print(
            f"  [PERM-ID] raw T#{raw_id} HARD merge rejected – existing truck T#{best_perm_hard} "
            f"age={best_age_hard} > 30 frames (score={best_score_hard:.2f})"
        )

    # Soft merge (time‑gated)
    if best_perm_soft is not None:
        job.raw_to_perm_id[raw_id] = best_perm_soft
        print(
            f"  [PERM-ID] raw T#{raw_id} soft-merged → perm T#{best_perm_soft} "
            f"(score={best_score_soft:.2f}, within {ACTIVE_WINDOW_FRAMES} frames)"
        )
        return best_perm_soft

    # New permanent ID
    perm_id = job._next_perm_id
    job._next_perm_id += 1
    job.raw_to_perm_id[raw_id] = perm_id
    print(f"  [PERM-ID] raw T#{raw_id} → NEW perm T#{perm_id}")
    return perm_id

# ========================= POST-PROCESSING CONSOLIDATION =====================
def _consolidate_perm_ids(job: VideoJob, registry: "TruckRegistry") -> None:
    """Merge perm IDs that belong to the same physical truck (safety net)."""
    with registry._lock:
        truck_ids = list(registry.trucks.keys())
    if len(truck_ids) <= 1:
        print("  [CONSOLIDATE] Only one truck — nothing to merge.")
        return

    merged_into: dict[int, int] = {}
    records = []
    with registry._lock:
        for tid in truck_ids:
            rec = registry.trucks[tid]
            records.append({
                "tid": tid,
                "first_frame": rec["first_seen_frame"],
                "last_frame": rec["last_seen_frame"],
                "union_bbox": tuple(rec["union_bbox"]),
                "duration": rec["duration_frames"],
            })
    records.sort(key=lambda r: r["first_frame"])

    for i, ra in enumerate(records):
        if ra["tid"] in merged_into:
            continue
        for rb in records[i + 1:]:
            if rb["tid"] in merged_into:
                continue
            overlap = max(
                _iou(ra["union_bbox"], rb["union_bbox"]),
                _containment(ra["union_bbox"], rb["union_bbox"]),
                _containment(rb["union_bbox"], ra["union_bbox"]),
            )
            if overlap < HIGH_IOU_MERGE_THRESH:
                continue
            time_overlap = max(
                0,
                min(ra["last_frame"], rb["last_frame"]) - max(ra["first_frame"], rb["first_frame"])
            )
            if time_overlap > 30:
                print(
                    f"  [CONSOLIDATE] T#{ra['tid']} and T#{rb['tid']} overlap spatially "
                    f"({overlap:.2f}) but are CONCURRENT ({time_overlap} frames) — kept separate"
                )
                continue
            # Sequential trucks in the same lane will have a large time gap even
            # though their union bboxes overlap (both drive through the same gate
            # position).  Never merge two records separated by more than
            # MIN_INTER_TRUCK_GAP_FRAMES — they are different physical trucks.
            time_gap = max(
                0,
                max(ra["first_frame"], rb["first_frame"])
                - min(ra["last_frame"], rb["last_frame"]),
            )
            if MIN_INTER_TRUCK_GAP_FRAMES > 0 and time_gap > MIN_INTER_TRUCK_GAP_FRAMES:
                print(
                    f"  [CONSOLIDATE] T#{ra['tid']} and T#{rb['tid']} spatial={overlap:.2f} "
                    f"but gap={time_gap} frames > {MIN_INTER_TRUCK_GAP_FRAMES} — different trucks, kept separate"
                )
                continue
            if ra["duration"] >= rb["duration"]:
                primary, secondary = ra["tid"], rb["tid"]
            else:
                primary, secondary = rb["tid"], ra["tid"]
            merged_into[secondary] = primary
            print(
                f"  [CONSOLIDATE] perm T#{secondary} → perm T#{primary} "
                f"(spatial={overlap:.2f}, time_overlap={time_overlap} frames)"
            )

    if not merged_into:
        print("  [CONSOLIDATE] No perm IDs to merge — tracking was already clean.")
        return

    with registry._lock:
        for secondary, primary in merged_into.items():
            sec_rec = registry.trucks.get(secondary)
            pri_rec = registry.trucks.get(primary)
            if sec_rec is None or pri_rec is None:
                continue
            pri_rec["first_seen_frame"] = min(pri_rec["first_seen_frame"], sec_rec["first_seen_frame"])
            pri_rec["last_seen_frame"] = max(pri_rec["last_seen_frame"], sec_rec["last_seen_frame"])
            pri_rec["first_seen_time_sec"] = round(pri_rec["first_seen_frame"] / registry.fps, 3)
            pri_rec["last_seen_time_sec"] = round(pri_rec["last_seen_frame"] / registry.fps, 3)
            pri_rec["duration_frames"] = pri_rec["last_seen_frame"] - pri_rec["first_seen_frame"] + 1
            pri_rec["duration_sec"] = round(pri_rec["duration_frames"] / registry.fps, 3)
            ua = pri_rec["union_bbox"]
            ub_ = sec_rec["union_bbox"]
            pri_rec["union_bbox"] = [
                min(ua[0], ub_[0]), min(ua[1], ub_[1]),
                max(ua[2], ub_[2]), max(ua[3], ub_[3]),
            ]
            n_pri = pri_rec.get("_conf_n", 1)
            n_sec = sec_rec.get("_conf_n", 1)
            pri_rec["confidence_avg"] = round(
                (pri_rec["confidence_avg"] * n_pri + sec_rec["confidence_avg"] * n_sec) / (n_pri + n_sec), 4
            )
            pri_rec["_conf_n"] = n_pri + n_sec

            pri_info = pri_rec["associated_info"]
            sec_info = sec_rec["associated_info"]
            for field, val in sec_info.items():
                if val is None:
                    continue
                existing = pri_info.get(field)
                if existing is None:
                    pri_info[field] = val
                    print(f"    [CONSOLIDATE-OCR] T#{primary}.{field} ← '{val}' (from T#{secondary})")
                elif isinstance(existing, dict) and isinstance(val, dict):
                    if val.get("confidence", 0.0) > existing.get("confidence", 0.0):
                        pri_info[field] = val
                        print(f"    [CONSOLIDATE-OCR] T#{primary}.{field} upgraded conf")
            del registry.trucks[secondary]
            print(f"  [CONSOLIDATE] ✅ Removed T#{secondary}, merged into T#{primary}")

    for raw_id, perm_id in list(job.raw_to_perm_id.items()):
        if perm_id in merged_into:
            job.raw_to_perm_id[raw_id] = merged_into[perm_id]
    for child_tid, lock in job.child_to_truck.items():
        if lock["truck_tid"] in merged_into:
            lock["truck_tid"] = merged_into[lock["truck_tid"]]
    print(f"  [CONSOLIDATE] Done. Merged {len(merged_into)} secondary ID(s).")

# ========================= TEXT QUALITY ======================================
def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for c in text if c.isalnum() or c in " -/.")
    return good / len(text)

def _extract_mineru_image(md_text: str) -> str | None:
    m = re.search(r"!\[\]\(([^)]+)\)", md_text)
    return m.group(1) if m else None


def _pick_reading_by_count(field_hist: dict) -> dict | None:
    """Pick the winning reading for a field by MAJORITY VOTE.

    The same field is OCR'd across many frames; the value the operator should see
    is the one read MOST OFTEN (max `count`), not merely the single highest-
    confidence frame — e.g. "SUDU8227936" read 6× beats a one-off "SUOU8227936".
    Ties break by confidence, then text quality, then shorter length.
    """
    best = None
    best_key = None
    for entry in (field_hist or {}).values():
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", "")
        if not text or _is_garbage_ocr(text):
            continue
        key = (
            int(entry.get("count", 0)),
            float(entry.get("confidence", 0.0)),
            _text_quality(text),
            -len(text),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = entry
    return best

# ========================= TRUCK REGISTRY ====================================
class TruckRegistry:
    _UNIFIED_TEMPLATE = {
        "container_number": None,
        "container_side_no": None,
        "container_company_logo": None,
        "other_container_info": None,
        "license_plate": None,
        "truck_company": None,
        "truck_number": None,
    }

    def __init__(self, fps: float, camera_source: str = "") -> None:
        self.fps = max(fps, 1.0)
        self.camera_source = camera_source
        self.trucks: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _new_record(self, tid: int, cls_name: str, bbox: tuple[int, int, int, int], frame_id: int, conf: float) -> dict[str, Any]:
        return {
            "track_id": tid,
            "type": cls_name,
            "camera": self.camera_source,
            "first_seen_frame": frame_id,
            "last_seen_frame": frame_id,
            "first_seen_time_sec": round(frame_id / self.fps, 3),
            "last_seen_time_sec": round(frame_id / self.fps, 3),
            "duration_frames": 1,
            "duration_sec": round(1.0 / self.fps, 3),
            "confidence_avg": round(conf, 4),
            "_conf_n": 1,
            "last_bbox": list(bbox),
            "union_bbox": list(bbox),
            "associated_info": {k: v for k, v in self._UNIFIED_TEMPLATE.items()},
        }

    def update(self, track_id: int, cls_name: str, bbox: tuple[int, int, int, int], frame_id: int, conf: float) -> None:
        with self._lock:
            if track_id not in self.trucks:
                rec = self._new_record(track_id, cls_name, bbox, frame_id, conf)
                self.trucks[track_id] = rec
                return
            rec = self.trucks[track_id]
            rec["last_seen_frame"] = frame_id
            rec["last_seen_time_sec"] = round(frame_id / self.fps, 3)
            rec["duration_frames"] = frame_id - rec["first_seen_frame"] + 1
            rec["duration_sec"] = round(rec["duration_frames"] / self.fps, 3)
            rec["last_bbox"] = list(bbox)
            ub = rec["union_bbox"]
            rec["union_bbox"] = [
                min(ub[0], bbox[0]), min(ub[1], bbox[1]),
                max(ub[2], bbox[2]), max(ub[3], bbox[3]),
            ]
            n = rec["_conf_n"]
            rec["confidence_avg"] = round((rec["confidence_avg"] * n + conf) / (n + 1), 4)
            rec["_conf_n"] = n + 1

    def attach_ocr(self, truck_track_id: int, truck_type: str, field: str, text: str, conf: float = 0.0, image: str | None = None, frame_id: int | None = None) -> None:
        if not text or _is_garbage_ocr(text):
            return
        with self._lock:
            rec = self.trucks.get(truck_track_id)
            if rec is None:
                return
            info = rec["associated_info"]
            if field not in info:
                return
            # Record a timestamped history of every distinct reading per field so the
            # merge layer can attribute a value to the correct physical truck by the
            # VIDEO time it was seen (e.g. a rear camera that reads truck-1's number
            # early and truck-2's number later on the same continuous track).
            if frame_id is not None:
                t_sec = round(frame_id / self.fps, 3)
                hist = info.setdefault("_ocr_history", {})
                field_hist = hist.setdefault(field, {})
                entry = field_hist.get(text)
                if entry is None:
                    field_hist[text] = {
                        "text": text, "confidence": round(conf, 4),
                        "camera": self.camera_source,
                        "time_first_sec": t_sec, "time_last_sec": t_sec, "count": 1,
                    }
                else:
                    entry["time_last_sec"] = t_sec
                    entry["count"] += 1
                    if conf > entry["confidence"]:
                        entry["confidence"] = round(conf, 4)
                # Final value = the reading seen MOST OFTEN across frames (majority
                # vote), not just the single most-confident frame.
                chosen = _pick_reading_by_count(field_hist)
                if chosen is not None:
                    info[field] = {
                        "text": chosen["text"],
                        "confidence": chosen.get("confidence", 0.0),
                        "camera": chosen.get("camera", self.camera_source),
                    }
                    if image:
                        info[field]["image"] = image
                return
            existing = info.get(field)
            if existing is None or existing == "":
                info[field] = {"text": text, "confidence": round(conf, 4), "camera": self.camera_source}
                if image:
                    info[field]["image"] = image
                return
            if isinstance(existing, dict):
                existing_text = existing.get("text", "")
                existing_conf = existing.get("confidence", 0.0)
            else:
                existing_text = existing
                existing_conf = 0.0
            if conf > existing_conf:
                info[field] = {"text": text, "confidence": round(conf, 4), "camera": self.camera_source}
                if image:
                    info[field]["image"] = image
            elif conf == existing_conf:
                new_q = _text_quality(text)
                old_q = _text_quality(existing_text)
                if new_q > old_q or (new_q == old_q and len(text) < len(existing_text)):
                    info[field] = {"text": text, "confidence": round(conf, 4), "camera": self.camera_source}
                    if image:
                        info[field]["image"] = image

    def field_is_empty(self, truck_track_id: int, field: str) -> bool:
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

    def find_truck_for_field(self, field: str, current_frame: int = 0) -> int | None:
        with self._lock:
            min_frame = current_frame - ACTIVE_WINDOW_FRAMES
            best_tid: int | None = None
            best_frame: int = -1
            for tid, rec in self.trucks.items():
                if rec.get("last_seen_frame", 0) < min_frame:
                    continue
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
                    if rec.get("last_seen_frame", 0) < min_frame:
                        continue
                    info = rec["associated_info"]
                    last_f = rec.get("last_seen_frame", 0)
                    if field in info and last_f > best_frame:
                        best_frame = last_f
                        best_tid = tid
            return best_tid

    def to_dict(self, session_info: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            trucks_out: dict[str, dict[str, Any]] = {}
            for tid, rec in self.trucks.items():
                if rec.get("duration_frames", 1) < MIN_TRUCK_TRACK_FRAMES:
                    print(f"  [GHOST-FILTER] skip T#{tid} duration_frames={rec.get('duration_frames',1)} < {MIN_TRUCK_TRACK_FRAMES}")
                    continue
                clean = {k: v for k, v in rec.items() if not k.startswith("_")}
                clean["associated_info"] = dict(rec["associated_info"])
                trucks_out[str(tid)] = clean
        twc = sum(1 for t in trucks_out.values() if t["type"] == "truck_with_container")
        twoc = sum(1 for t in trucks_out.values() if t["type"] == "truck_without_container")
        return {
            "session": session_info or {},
            "summary": {
                "total_trucks_tracked": len(trucks_out),
                "trucks_with_container": twc,
                "trucks_without_container": twoc,
            },
            "trucks": trucks_out,
        }

# ========================= OCR HELPERS =======================================
def _parse_mineru_markdown(md_text: str) -> str:
    details = re.findall(r"<details>.*?<summary>[^<]*</summary>\s*(.*?)\s*</details>", md_text, flags=re.DOTALL)
    if details:
        combined = " | ".join(d.strip() for d in details if d.strip())
        if combined:
            return combined
    lines = [line.strip() for line in md_text.splitlines() if line.strip() and not line.strip().startswith("![") and not re.match(r"^<[^>]+>$", line.strip())]
    return " ".join(lines).strip()

_GARBAGE_OCR_PHRASES = (
    "abstract grayscale", "abstract gray", "grayscale curved",
    "curved shape", "simple geometric", "no text or symbols",
    "no visible text", "no text", "not visible", "no symbols",
    "close-up of", "close up of", "photograph of",
    "background with", "metallic", "cylindrical",
    "image of", "picture of", "image shows",
)

# A real plate / container / truck-number token: 4–12 alphanumerics containing a digit.
_ID_PATTERN = re.compile(r"[A-Za-z0-9]{4,12}")

def _is_garbage_ocr(text: str) -> bool:
    """Return True when the OCR result is an image-caption description, not actual text.

    Two triggers:
      1. The text contains a known caption phrase ("abstract grayscale", "close-up of", ...).
      2. The text is longer than 30 chars and contains NO plausible identifier token
         (a 4–12 char alphanumeric run that includes at least one digit).
    """
    if not text:
        return False
    lower = text.lower()
    if any(phrase in lower for phrase in _GARBAGE_OCR_PHRASES):
        return True
    if len(text) > 30:
        has_id_token = any(
            any(ch.isdigit() for ch in tok)
            for tok in _ID_PATTERN.findall(text)
        )
        if not has_id_token:
            return True
    return False


def _extract_md_from_zip(zip_url: str) -> str:
    try:
        resp = requests.get(zip_url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if name.endswith("full.md"):
                    return _parse_mineru_markdown(zf.read(name).decode("utf-8", errors="ignore"))
    except Exception:
        return ""
    return ""

# Runtime record of MinerU OCR connectivity, surfaced to the UI via the
# /api/ocr/status endpoint so the operator can see whether reads are actually
# reaching the MinerU service (it is the only OCR backend — on failure crops
# simply return no text).
_mineru_stats = {
    "ok_calls": 0,
    "failed_calls": 0,
    "last_ok_ts": None,
    "last_error": None,
    "last_error_ts": None,
}


def _mineru_mark_ok() -> None:
    _mineru_stats["ok_calls"] += 1
    _mineru_stats["last_ok_ts"] = time.time()


def _mineru_mark_fail(err: object) -> None:
    _mineru_stats["failed_calls"] += 1
    _mineru_stats["last_error"] = str(err)
    _mineru_stats["last_error_ts"] = time.time()


def mineru_status(probe: bool = True) -> dict:
    """Report MinerU OCR connectivity for the UI status indicator.

    `reachable` reflects a live lightweight probe of the MinerU host (when
    `probe` is True); the `*_calls` counters reflect real OCR traffic so the
    operator can tell connected-but-idle from actively-reading.
    """
    token_configured = bool(MINERU_TOKEN)
    reachable: bool | None = None
    probe_error: str | None = None
    if probe and token_configured:
        try:
            resp = requests.get("https://mineru.net", timeout=4)
            reachable = resp.status_code < 500
        except requests.exceptions.RequestException as exc:
            reachable = False
            probe_error = str(exc)
    if token_configured and reachable is not False:
        status = "connected"
    elif not token_configured:
        status = "no_token"
    else:
        status = "disconnected"
    return {
        "provider": "MinerU",
        "status": status,
        "token_configured": token_configured,
        "reachable": reachable,
        "ok_calls": _mineru_stats["ok_calls"],
        "failed_calls": _mineru_stats["failed_calls"],
        "last_ok_ts": _mineru_stats["last_ok_ts"],
        "last_error": probe_error or _mineru_stats["last_error"],
    }


def _call_mineru_ocr(crop_rgb: np.ndarray, max_retries: int = 2) -> str:
    if not MINERU_TOKEN:
        return ""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {MINERU_TOKEN}"}
    _, enc = cv2.imencode(".png", crop_rgb)
    filename = f"crop_{uuid.uuid4().hex[:8]}.png"
    data_id = uuid.uuid4().hex
    for attempt in range(max_retries):
        try:
            step1 = requests.post("https://mineru.net/api/v4/file-urls/batch", json={"files": [{"name": filename, "data_id": data_id}], "model_version": "vlm"}, headers=headers, timeout=15)
            step1.raise_for_status()
            payload = step1.json()
            if payload.get("code") != 0:
                print(f"  ⚠️ MinerU: {payload.get('msg')}")
                _mineru_mark_fail(payload.get("msg") or "non-zero response code")
                continue
            _mineru_mark_ok()
            batch_id = payload["data"]["batch_id"]
            upload_url = payload["data"]["file_urls"][0]
            put_resp = requests.put(upload_url, data=enc.tobytes(), timeout=30)
            if put_resp.status_code not in (200, 201):
                print(f"  ⚠️ Upload failed: {put_resp.status_code}")
                continue
            print(f"  [MinerU] crop uploaded ({len(enc.tobytes())} bytes) → batch {batch_id[:10]}… polling for result", flush=True)
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
            _mineru_mark_fail(f"timeout (attempt {attempt + 1})")
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ MinerU request error: {e} (attempt {attempt + 1}/{max_retries})")
            _mineru_mark_fail(e)
    return ""

# ========================= VISUAL HELPERS ====================================
def _resize_for_inference(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= INFER_WIDTH:
        return frame, 1.0
    scale = INFER_WIDTH / w
    return cv2.resize(frame, (INFER_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR), scale

def _save_detection_crop(frame: np.ndarray, cls_name: str, frame_id: int, det_index: int, conf: float, track_id: int | None, box: tuple[int, int, int, int], detections_dir: str | None = None) -> None:
    if not SAVE_DETECTION_CROPS:
        return
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return
    base_dir = detections_dir or os.path.join(OUTPUT_DIR, "detections")
    safe_class = re.sub(r"[^A-Za-z0-9_.-]", "_", cls_name)
    class_dir = os.path.join(base_dir, safe_class)
    os.makedirs(class_dir, exist_ok=True)
    tid = f"t{track_id}" if track_id is not None else "no_track"
    stem = f"f{frame_id:06d}_i{det_index:03d}_{tid}_conf{conf:.3f}"
    image_path = os.path.join(class_dir, f"{stem}.jpg")
    meta_path = os.path.join(class_dir, f"{stem}.json")
    cv2.imwrite(image_path, crop)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"class_name": cls_name, "confidence": round(conf, 6), "track_id": track_id, "frame_id": frame_id, "detection_index": det_index, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "crop_path": image_path}, f, indent=2)

def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if h < 64:
        crop_bgr = cv2.resize(crop_bgr, (max(1, int(w * 64 / h)), 64), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

def _annotate_frame(frame: np.ndarray, boxes_data: list[tuple], assoc_map: dict[int, tuple[int, str, str]], ocr_map: dict[int, str], pending_track_ids: set[int], job: VideoJob | None = None, class_names: list[str] | None = None) -> np.ndarray:
    class_names = class_names or CLASS_NAMES
    H, W = frame.shape[:2]
    for i, (x1, y1, x2, y2, cls_id, conf_val, tid_raw, mask) in enumerate(boxes_data):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        colour = PALETTE.get(cls_id, (88, 170, 255))
        tid_str = f"#{tid_raw} " if tid_raw is not None else ""
        if mask is not None:
            m = mask
            ov = frame.copy()
            ov[m > 0.5] = colour
            cv2.addWeighted(ov, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
            ctrs, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, ctrs, -1, colour, 2)
        elif cls_name in TRUCK_CLASSES:
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
            cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{tid_str}{cls_name} {conf_val:.2f}"
        if tid_raw is not None and cls_name in CHILD_CLASSES and job is not None:
            if tid_raw in job.child_to_truck:
                link = job.child_to_truck[tid_raw]
                label += f"  🔒→T#{link['truck_tid']}"
            elif tid_raw in job.child_assoc_votes:
                votes = job.child_assoc_votes[tid_raw]
                best_k = max(votes, key=lambda k: votes[k])
                label += f"  ⏳{votes[best_k]}/{LOCK_AFTER_N_FRAMES}→T#{best_k[0]}"
        elif i in assoc_map:
            truck_tid, _truck_type, _field = assoc_map[i]
            label += f"  →T#{truck_tid}"
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - lh - bl - 4)), (x1 + lw, y1), colour, -1)
        cv2.putText(frame, label, (x1, max(lh, y1 - bl - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
        if cls_name in OCR_CLASSES:
            if i in ocr_map and ocr_map[i]:
                txt = ocr_map[i][:60]
                color = colour
            elif tid_raw in pending_track_ids:
                txt = "scanning…"
                color = (200, 200, 50)
            else:
                continue
            txt_y = y2 + 22 if y2 + 42 < H else max(20, y1 - 10)
            cv2.putText(frame, txt, (x1 + 1, txt_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (x1, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame

def _det_key(track_id: int | None, frame_id: int, det_idx: int, truck_tid: int | None = None, field: str | None = None) -> str:
    if truck_tid is not None and field is not None:
        return f"truck_{truck_tid}_{field}"
    return f"t{track_id}" if track_id is not None else f"f{frame_id}_i{det_idx}"

FIELD_MEDIA_DIR = os.path.join(OUTPUT_DIR, "field_media")


def _save_field_media(job_id: str, truck_tid: int, field: str, frame_id: int, crop_rgb: np.ndarray) -> None:
    """Persist the exact crop we OCR'd for a (truck, field), so the UI can replay
    the per-field captures as a short clip when the operator clicks that field."""
    try:
        if crop_rgb is None or getattr(crop_rgb, "size", 0) == 0:
            return
        safe_field = re.sub(r"[^A-Za-z0-9_.-]", "_", str(field))
        out_dir = os.path.join(FIELD_MEDIA_DIR, str(job_id), str(truck_tid), safe_field)
        os.makedirs(out_dir, exist_ok=True)
        bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"f{frame_id:06d}.jpg"), bgr)
    except Exception as exc:  # noqa: BLE001 — media capture must never break OCR
        print(f"  ⚠️ field media save failed: {exc}")


def _enqueue_ocr(job_id: str, key: str, crop_rgb: np.ndarray, field: str, frame_id: int, truck_tid: int | None, truck_type: str | None, child_track_id: int | None = None, conf: float = 0.0) -> None:
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
            if child_track_id is not None and child_track_id in job.child_to_truck:
                lock = job.child_to_truck[child_track_id]
                truck_tid = lock["truck_tid"]
                truck_type = lock["truck_type"]
                field = lock["field"]
                print(f"  [OCR] using LOCKED association: child T#{child_track_id} → T#{truck_tid}")
        crop_h, crop_w = (crop_rgb.shape[0], crop_rgb.shape[1]) if crop_rgb is not None and crop_rgb.size else (0, 0)
        tid_label = f"T#{truck_tid}" if truck_tid is not None else "T#?"
        send_log = f"[f{frame_id}] → SEND→OCR  field='{field}'  truck={tid_label}  conf={conf:.2f}  crop={crop_w}x{crop_h}"
        print(f"  [OCR→SEND] {send_log}", flush=True)
        with _jobs_lock:
            _sj = JOBS.get(job_id)
            if _sj is not None:
                _sj.ocr_log.append(send_log)
                _sj.ocr_log = _sj.ocr_log[-5000:]
        md_text = _call_mineru_ocr(crop_rgb)
        plain_text = _parse_mineru_markdown(md_text) if md_text else None
        image_path = _extract_mineru_image(md_text) if md_text else None
        print(
            f"  [OCR←RECV] [f{frame_id}] field='{field}'  raw_chars={len(md_text or '')}  "
            f"parsed='{plain_text if plain_text else ''}'",
            flush=True,
        )
        with _jobs_lock:
            job = JOBS.get(job_id)
            if job is None:
                _ocr_queue.task_done()
                continue
            job.pending_keys.discard(key)
            if plain_text and _is_garbage_ocr(plain_text):
                print(f"  [OCR] 🗑 garbage OCR for '{field}', dropping: '{plain_text[:80]}'")
                _ocr_queue.task_done()
                continue
            if plain_text:
                job.ocr_cache[key] = plain_text
                cam_prefix = f"[{job.camera_source}] " if job.camera_source else ""
                log = f"[f{frame_id}] {cam_prefix}← RECV←OCR  {field} → '{plain_text}'"
                resolved_tid = truck_tid
                resolved_type = truck_type or ""
                if job.registry is not None:
                    if resolved_tid is not None:
                        job.registry.attach_ocr(resolved_tid, resolved_type, field, plain_text, conf, image_path, frame_id=frame_id)
                        log += f"  (T#{resolved_tid})"
                    else:
                        candidate = job.registry.find_truck_for_field(field, current_frame=frame_id)
                        if candidate is not None:
                            resolved_tid = candidate
                            rec = job.registry.trucks.get(candidate)
                            resolved_type = rec["type"] if rec else ""
                            job.registry.attach_ocr(resolved_tid, resolved_type, field, plain_text, conf, image_path, frame_id=frame_id)
                            log += f"  (recovered → attached to T#{resolved_tid})"
                            print(f"  [OCR] ↩ '{field}' had no in-frame truck box; recovered → attached to T#{resolved_tid}")
                        else:
                            if not _is_garbage_ocr(plain_text) and len(plain_text) <= 60 and len(job.orphan_buffer) < 30:
                                job.orphan_buffer.append({"field": field, "text": plain_text, "conf": conf, "image": image_path})
                                log += "  (orphan — buffered for retry)"
                                print(f"  [OCR] ⏳ orphan '{field}' buffered → '{plain_text}'")
                            else:
                                log += "  (orphan — no compatible truck yet)"
                                print(f"  [OCR] ⚠ orphan '{field}' — no compatible truck, dropping")
                if resolved_tid is not None:
                    _save_field_media(job_id, resolved_tid, field, frame_id, crop_rgb)
                job.ocr_log.append(log)
                job.ocr_log = job.ocr_log[-5000:]
                print(f"  [OCR] ✓ {key} → '{plain_text}'", flush=True)
            else:
                job.submitted_keys.discard(key)
                print(f"  [OCR] ✗ {key} — will retry")
        _ocr_queue.task_done()

for _ in range(OCR_WORKERS):
    threading.Thread(target=_ocr_worker, daemon=True).start()

# ========================= PROCESSING ========================================
def _run_video_analysis(video_path: str, job: VideoJob | None = None) -> dict[str, Any]:
    camera = (job.camera_source if job is not None else "").strip().lower()
    is_back = camera in BACK_CAMERA_LABELS
    model_path = MODEL_PATH_BACK if is_back else MODEL_PATH
    detections_dir = os.path.join(OUTPUT_DIR_BACK if is_back else OUTPUT_DIR, "detections")
    os.makedirs(detections_dir, exist_ok=True)
    tracking_model, class_names = _load_tracking_model(model_path)
    has_truck_class = bool(TRUCK_CLASSES & set(class_names))
    virtual_truck_mode = not has_truck_class
    VIRTUAL_TRUCK_ID = 1
    VIRTUAL_TRUCK_TYPE = "truck_without_container"
    print(f"[PlateFlow] Video job tracker isolated: camera='{camera or 'default'}' model={model_path} virtual_truck_mode={virtual_truck_mode} path={video_path}", flush=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    session_info: dict[str, Any] = {
        "video_path": video_path, "total_frames": total_frames, "video_fps": round(fps, 2),
        "resolution": f"{w_orig}x{h_orig}", "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE, "model": model_path, "confidence": CONF, "truck_confidence": TRUCK_CONF_THRESH,
        "infer_width": INFER_WIDTH, "tracker": YOLO_TRACKER, "save_detection_crops": SAVE_DETECTION_CROPS,
        "lock_after_n_frames": LOCK_AFTER_N_FRAMES, "active_window_frames": ACTIVE_WINDOW_FRAMES,
        "high_iou_merge_thresh": HIGH_IOU_MERGE_THRESH, "enable_frame_skip": ENABLE_FRAME_SKIP,
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES if ENABLE_FRAME_SKIP else 1,
        "ocr_every_n_frames": OCR_EVERY_N_FRAMES, "camera": job.camera_source if job is not None else "",
    }
    camera_src = job.camera_source if job is not None else ""
    camera_stats = CameraDetectionStats(camera=camera_src or "default")
    registry = TruckRegistry(fps=fps, camera_source=camera_src)
    if job is not None:
        _set_job(job.job_id, registry=registry, session_info=session_info, total_frames=total_frames)
    frame_id = 0
    t_prev = time.time()
    fps_smooth = 0.0
    while True:
        ret, frame_orig = cap.read()
        if not ret:
            break
        if FRAME_SKIP > 0 and frame_id % (FRAME_SKIP + 1) != 0:
            frame_id += 1
            if job is not None:
                progress = (frame_id / total_frames) if total_frames > 0 else 0.0
                _set_job(job.job_id, progress=progress, frame_id=frame_id, message=f"Skipping frame {frame_id}/{total_frames}; processing every {PROCESS_EVERY_N_FRAMES} frame(s).")
            continue
        frame_small, scale = _resize_for_inference(frame_orig)
        t0 = time.time()
        try:
            results = tracking_model.track(frame_small, conf=CONF, iou=IOU_THRESH, tracker=YOLO_TRACKER, persist=True, verbose=False, half=USE_HALF, device=DEVICE, retina_masks=True)
        except Exception:
            results = None
        infer_ms = (time.time() - t0) * 1000
        boxes_data: list[tuple] = []
        ocr_map: dict[int, str] = {}
        boxes_raw = results[0].boxes if results else None
        masks_raw = results[0].masks if results else None
        if boxes_raw is not None and len(boxes_raw) > 0:
            for i, box in enumerate(boxes_raw):
                sx1, sy1, sx2, sy2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, int(sx1 / scale))
                y1 = max(0, int(sy1 / scale))
                x2 = min(w_orig, int(sx2 / scale))
                y2 = min(h_orig, int(sy2 / scale))
                if x2 <= x1 or y2 <= y1:
                    continue
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                track_id = int(box.id[0]) if box.id is not None else None
                if cls_name in TRUCK_CLASSES:
                    if conf_val < TRUCK_CONF_THRESH:
                        continue
                    truck_area = (x2 - x1) * (y2 - y1)
                    if truck_area < MIN_TRUCK_AREA:
                        print(f"  [AREA] drop {cls_name} area={truck_area} < {MIN_TRUCK_AREA}")
                        continue
                if masks_raw is not None and i < len(masks_raw.data):
                    raw_mask = masks_raw.data[i].cpu().numpy()
                    mask = cv2.resize(raw_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = None
                mask_area = _mask_area(mask, (x1, y1, x2, y2))
                if cls_name in TRUCK_CLASSES and mask_area < MIN_TRUCK_MASK_AREA:
                    print(f"  [MASK-AREA] drop {cls_name} T#{track_id} mask_area={mask_area:.0f} < {MIN_TRUCK_MASK_AREA}")
                    continue
                boxes_data.append((x1, y1, x2, y2, cls_id, conf_val, track_id, mask))
                # YOLO raw detection counter (one instance per detected box per frame)
                if cls_name == "truck_with_container":
                    camera_stats.yolo_with_container += 1
                elif cls_name == "truck_without_container":
                    camera_stats.yolo_without_container += 1
                _save_detection_crop(frame_orig, cls_name, frame_id, i, conf_val, track_id, (x1, y1, x2, y2), detections_dir=detections_dir)
        boxes_data = _dedupe_truck_detections(boxes_data, class_names)
        if job is not None:
            remapped: list[tuple] = []
            for x1, y1, x2, y2, cls_id, conf_val, track_id, mask in boxes_data:
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                if cls_name in TRUCK_CLASSES and track_id is not None:
                    perm_id = _get_perm_truck_id(job, track_id, (x1, y1, x2, y2), registry, frame_id)
                    remapped.append((x1, y1, x2, y2, cls_id, conf_val, perm_id, mask))
                    # ByteTrack raw unique-ID counter (raw ByteTrack IDs before perm-ID consolidation)
                    if cls_name == "truck_with_container":
                        camera_stats.bytetrack_ids_with_container.add(track_id)
                    else:
                        camera_stats.bytetrack_ids_without_container.add(track_id)
                else:
                    remapped.append((x1, y1, x2, y2, cls_id, conf_val, track_id, mask))
            boxes_data = remapped
        for x1, y1, x2, y2, cls_id, conf_val, track_id, _mask in boxes_data:
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            if cls_name in TRUCK_CLASSES and track_id is not None:
                registry.update(track_id, cls_name, (x1, y1, x2, y2), frame_id, conf_val)
        if virtual_truck_mode and boxes_data:
            vx1 = min(b[0] for b in boxes_data)
            vy1 = min(b[1] for b in boxes_data)
            vx2 = max(b[2] for b in boxes_data)
            vy2 = max(b[3] for b in boxes_data)
            vconf = max((b[5] for b in boxes_data), default=0.0)
            registry.update(VIRTUAL_TRUCK_ID, VIRTUAL_TRUCK_TYPE, (vx1, vy1, vx2, vy2), frame_id, vconf)
        if job is not None and frame_id % 5 == 0 and job.orphan_buffer:
            with _jobs_lock:
                remaining = []
                for item in job.orphan_buffer:
                    candidate = registry.find_truck_for_field(item["field"], current_frame=frame_id)
                    if candidate is not None:
                        rec = registry.trucks.get(candidate)
                        registry.attach_ocr(candidate, rec["type"] if rec else "", item["field"], item["text"], item["conf"], item.get("image"))
                        rescued_log = f"[f{frame_id}] {item['field']} rescued → T#{candidate} '{item['text']}'"
                        job.ocr_log.append(rescued_log)
                        job.ocr_log = job.ocr_log[-5000:]
                        print(f"  [OCR] ✅ orphan rescued: {rescued_log}")
                    else:
                        remaining.append(item)
                job.orphan_buffer = remaining
        if virtual_truck_mode:
            assoc_map = {}
            for i, (x1, y1, x2, y2, cls_id, conf_val, track_id, _mask) in enumerate(boxes_data):
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                if cls_name in OCR_CLASSES:
                    assoc_map[i] = (VIRTUAL_TRUCK_ID, VIRTUAL_TRUCK_TYPE, cls_name)
        else:
            assoc_map = _build_association_map_integrated(boxes_data, job if job is not None else None, class_names)
        if job is not None and frame_id % OCR_RETRY_EVERY == 0 and frame_id > 0:
            with _jobs_lock:
                for i, (x1, y1, x2, y2, cls_id, _conf, track_id, _mask) in enumerate(boxes_data):
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    if cls_name not in OCR_CLASSES or i not in assoc_map:
                        continue
                    truck_tid, truck_type, fld = assoc_map[i]
                    if registry.field_is_empty(truck_tid, fld):
                        retry_key = _det_key(track_id, frame_id, i, truck_tid, fld)
                        if retry_key in job.submitted_keys:
                            job.submitted_keys.discard(retry_key)
                            job.pending_keys.discard(retry_key)
                            print(f"  [RETRY] f{frame_id} T#{truck_tid} field='{fld}' unlocked for re-submission")
        if job is not None:
            with _jobs_lock:
                submitted = set(job.submitted_keys)
                cached = dict(job.ocr_cache)
                pending = set(job.pending_keys)
        else:
            submitted, cached, pending = set(), {}, set()
        for i, (x1, y1, x2, y2, cls_id, conf_val, track_id, _mask) in enumerate(boxes_data):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            if cls_name not in OCR_CLASSES:
                continue
            if conf_val < OCR_CONF_THRESH:
                continue
            if frame_id % OCR_EVERY_N_FRAMES != 0:
                continue
            crop_bgr = frame_orig[y1:y2, x1:x2].copy()
            if crop_bgr.size == 0:
                continue
            if SAVE_CROPS:
                cv2.imwrite(os.path.join(OUTPUT_DIR, "ocr_crops", f"f{frame_id:05d}_i{i}_{cls_name}.jpg"), crop_bgr)
            truck_tid, truck_type, field = None, None, cls_name
            if i in assoc_map:
                truck_tid, truck_type, field = assoc_map[i]
            key = _det_key(track_id, frame_id, i, truck_tid, field)
            if key in cached:
                ocr_map[i] = cached[key]
            should_submit = False
            if job is not None:
                owner_id = truck_tid if truck_tid is not None else track_id
                best_key = (field, owner_id) if owner_id is not None else None
                if best_key:
                    with _jobs_lock:
                        prev_best = job.ocr_best_conf.get(best_key, 0.0)
                        if conf_val > prev_best:
                            job.ocr_best_conf[best_key] = conf_val
                            job.submitted_keys.discard(key)
                            job.pending_keys.discard(key)
                            should_submit = True
                            print(f"  [OCR-BEST] {field} owner#{owner_id} conf={conf_val:.3f} > prev={prev_best:.3f}")
                elif key not in pending and key not in submitted:
                    should_submit = True
            if should_submit and job is not None:
                _enqueue_ocr(job.job_id, key, _preprocess_crop(crop_bgr), field, frame_id, truck_tid, truck_type, track_id, conf=conf_val)
        n_cache = len(cached)
        n_pending = len(pending)
        q_size = _ocr_queue.qsize()
        n_trucks = len(registry.trucks)
        pct = int(frame_id / total_frames * 100) if total_frames > 0 else 0
        gpu_txt = f"GPU: {torch.cuda.memory_allocated() // 1024**2} MB" if DEVICE == "cuda" else "CPU mode"
        n_locked = len(job.child_to_truck) if job else 0
        n_voting = len(job.child_assoc_votes) if job else 0
        pending_track_ids: set[int] = {int(k[1:]) for k in pending if k.startswith("t") and k[1:].isdigit()}
        annotated = _annotate_frame(frame_orig.copy(), boxes_data, assoc_map, ocr_map, pending_track_ids, job, class_names)
        cv2.putText(annotated, f"Frame {frame_id}/{total_frames} ({pct}%)  |  {fps_smooth:.1f} FPS  |  {infer_ms:.0f} ms", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Trucks tracked: {n_trucks}  |  OCR done: {n_cache}  pending: {n_pending}  queue: {q_size}  |  {gpu_txt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Assoc locked: {n_locked}  voting: {n_voting}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)
        bar_w = int(w_orig * frame_id / total_frames) if total_frames > 0 else 0
        cv2.rectangle(annotated, (0, h_orig - 8), (w_orig, h_orig), (40, 40, 40), -1)
        cv2.rectangle(annotated, (0, h_orig - 8), (bar_w, h_orig), (0, 220, 100), -1)
        ok, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok and job is not None:
            _set_job(job.job_id, latest_frame_jpeg=jpeg.tobytes())
        t_now = time.time()
        fps_smooth = 0.8 * fps_smooth + 0.2 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        frame_id += 1
        if job is not None:
            progress = (frame_id / total_frames) if total_frames > 0 else 0.0
            _set_job(job.job_id, progress=round(progress, 4), frame_id=frame_id, fps=round(fps_smooth, 2), message=f"Processing frame {frame_id}/{total_frames}  |  OCR done: {n_cache}  pending: {n_pending}  queue: {q_size}  |  Locked: {n_locked}  Voting: {n_voting}")
            if frame_id % JSON_SNAP_EVERY == 0:
                _set_job(job.job_id, json_snapshot=json.dumps(registry.to_dict(session_info), indent=2))
    cap.release()
    session_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    session_info["frames_processed"] = frame_id
    session_info["camera_detection_stats"] = camera_stats.to_dict()
    final_data = registry.to_dict(session_info)
    if job is not None:
        _set_job(job.job_id, json_snapshot=json.dumps(final_data, indent=2))
    return final_data

def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = JOBS[job_id]
        video_path = job.temp_video_path
    if not video_path:
        _set_job(job_id, state="failed", error="Missing temp video path", message="Job failed.")
        return
    try:
        _set_job(job_id, state="running", message="Video processing started.")
        _run_video_analysis(video_path, job=job)
        # Drain remaining OCR work. MinerU can be slow, so wait as long as the
        # backlog keeps SHRINKING — only give up if the pending count stops
        # decreasing for OCR_DRAIN_STALL_SECONDS (a genuine stall), with a high
        # absolute ceiling. This lets long videos finish instead of timing out.
        stall_limit = _env_float("OCR_DRAIN_STALL_SECONDS", 420.0)
        hard_cap = _env_float("OCR_DRAIN_MAX_SECONDS", 7200.0)
        start_ts = time.time()
        last_pending = None
        last_progress_ts = start_ts
        while time.time() - start_ts < hard_cap:
            with _jobs_lock:
                n_pending = len(JOBS[job_id].pending_keys)
            if n_pending == 0:
                break
            now = time.time()
            if last_pending is None or n_pending < last_pending:
                last_pending = n_pending
                last_progress_ts = now
            elif now - last_progress_ts > stall_limit:
                print(f"  [OCR-WAIT] stalled at {n_pending} pending for >{stall_limit:.0f}s — finalizing anyway")
                break
            waited = int(now - start_ts)
            _set_job(job_id, message=f"Frames processed — waiting for {n_pending} OCR task(s) to complete… ({waited}s)")
            print(f"  [OCR-WAIT] {n_pending} task(s) still pending… ({waited}s elapsed)")
            time.sleep(2)
        with _jobs_lock:
            job = JOBS[job_id]
        if job.registry is not None:
            print("[PlateFlow] Running post-processing ID consolidation…", flush=True)
            _consolidate_perm_ids(job, job.registry)
        final_result = job.registry.to_dict(job.session_info) if job.registry is not None else {}
        _set_job(job_id, state="completed", progress=1.0, result=final_result, json_snapshot=json.dumps(final_result, indent=2), message="Video processing completed.")
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
        frame = cv2.imread(str(image_path))
        results = model.predict(str(image_path), verbose=False, conf=CONF, iou=IOU_THRESH)
        detections = []
        if results:
            names = results[0].names or {}
            for det_index, box in enumerate(results[0].boxes):
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf_val = float(box.conf[0].item()) if box.conf is not None else 0.0
                cls_name = names.get(cls_id, f"class_{cls_id}")
                if cls_name in TRUCK_CLASSES and conf_val < TRUCK_CONF_THRESH:
                    continue
                if frame is not None:
                    h, w = frame.shape[:2]
                    x1 = max(0, min(w, int(xyxy[0])))
                    y1 = max(0, min(h, int(xyxy[1])))
                    x2 = max(0, min(w, int(xyxy[2])))
                    y2 = max(0, min(h, int(xyxy[3])))
                    if x2 > x1 and y2 > y1:
                        _save_detection_crop(frame, cls_name, frame_id=0, det_index=det_index, conf=conf_val, track_id=None, box=(x1, y1, x2, y2))
                detections.append({"plate_text": cls_name, "confidence": round(conf_val, 4), "bbox": {"x1": round(xyxy[0], 2), "y1": round(xyxy[1], 2), "x2": round(xyxy[2], 2), "y2": round(xyxy[3], 2)}})
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
    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix="plateflow_job_")
    video_path = Path(tmp_dir) / (file.filename or "upload.mp4")
    video_path.write_bytes(raw)
    with _jobs_lock:
        JOBS[job_id] = VideoJob(job_id=job_id, state="queued", message="Job queued.", temp_video_path=str(video_path))
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "state": "queued", "message": "Video job created."}

@app.get("/analyze-video/jobs/{job_id}")
def analyze_video_job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job.job_id, "state": job.state, "progress": job.progress, "frame_id": job.frame_id, "total_frames": job.total_frames, "fps": job.fps, "message": job.message, "error": job.error, "ocr_log": list(job.ocr_log), "json_snapshot": job.json_snapshot}

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