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
    "/content/drive/MyDrive/vehicle_plate/output/runs/yolo26x_seg_phase2/weights/best.pt",
)
OUTPUT_DIR        = os.getenv(
    "OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "plateflow_outputs"),
)
CONF              = _env_float("CONF", 0.45)
TRUCK_CONF_THRESH = _env_float("TRUCK_CONF_THRESH", 0.70)
IOU_THRESH        = 0.50
IOU_MERGE_THRESH  = _env_float("IOU_MERGE_THRESH", 0.10)
MASK_ALPHA        = 0.65
INFER_WIDTH       = _env_int("INFER_WIDTH", 1280)
ENABLE_FRAME_SKIP = _env_bool("ENABLE_FRAME_SKIP", False)
PROCESS_EVERY_N_FRAMES = _env_int("PROCESS_EVERY_N_FRAMES", 1)
FRAME_SKIP        = PROCESS_EVERY_N_FRAMES - 1 if ENABLE_FRAME_SKIP else 0
OCR_EVERY_N_FRAMES = _env_int("OCR_EVERY_N_FRAMES", 1)
USE_HALF          = True
OCR_WORKERS       = 3
SAVE_CROPS        = False
SAVE_DETECTION_CROPS = _env_bool("SAVE_DETECTION_CROPS", True)
MIN_ASSOC_SCORE   = 0.10
MIN_TRUCK_AREA    = 40_000
MIN_TRUCK_TRACK_FRAMES = _env_int("MIN_TRUCK_TRACK_FRAMES", 3)
MIN_TRUCK_MASK_AREA = _env_float("MIN_TRUCK_MASK_AREA", 20_000)
JSON_SNAP_EVERY   = 30
OCR_RETRY_EVERY   = 45

# ── Association locking from snippet 1 ───────────────────────────────────────
LOCK_AFTER_N_FRAMES = 5  # Lock after N consecutive frames with same truck
ACTIVE_WINDOW_FRAMES = _env_int("ACTIVE_WINDOW_FRAMES", 90)

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
if SAVE_DETECTION_CROPS:
    os.makedirs(os.path.join(OUTPUT_DIR, "detections"), exist_ok=True)

# Write custom ByteTrack config for better tracking continuity
_TRACKER_CFG_PATH = os.path.join(OUTPUT_DIR, "custom_bytetrack.yaml")
with open(_TRACKER_CFG_PATH, "w") as _f:
    _f.write(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.05\n"
        "new_track_thresh: 0.25\n"
        "track_buffer: 60\n"
        "match_thresh: 0.85\n"
        "fuse_score: True\n"
    )
YOLO_TRACKER = _TRACKER_CFG_PATH

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
print(
    f"[PlateFlow] Detection config: conf={CONF} truck_conf={TRUCK_CONF_THRESH} "
    f"infer_width={INFER_WIDTH} tracker={YOLO_TRACKER} iou_merge={IOU_MERGE_THRESH}",
    flush=True,
)
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


def _load_tracking_model() -> YOLO:
    tracking_model = YOLO(MODEL_PATH)
    tracking_model.to(DEVICE)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    tracking_model.predict(dummy, verbose=False, half=USE_HALF, device=DEVICE)
    return tracking_model


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
    # key: child track_id, value: dict mapping (truck_tid, truck_type) → vote count
    child_assoc_votes: dict[int, dict[tuple, int]] = field(default_factory=dict)

    raw_to_perm_id: dict[int, int] = field(default_factory=dict)   # raw tracker ID → permanent ID
    _next_perm_id: int = 1
    
    # Best confidence tracker per (cls_name, track_id)
    ocr_best_conf: dict[tuple[str, int], float] = field(default_factory=dict)

    # Buffered OCR results that had no compatible truck at completion time.
    # Flushed to the registry every few frames once a truck appears.
    orphan_buffer: list[dict[str, Any]] = field(default_factory=list)


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


def _dedupe_truck_detections(boxes_data: list[tuple]) -> list[tuple]:
    truck_indices: list[int] = []
    keep = [True] * len(boxes_data)
    for i, (x1, y1, x2, y2, cls_id, _conf, _tid, _mask) in enumerate(boxes_data):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
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
    """
    Majority-vote locking: accumulate votes per (truck_tid, truck_type) candidate.
    Locks permanently when any candidate reaches LOCK_AFTER_N_FRAMES total votes.
    A single different frame no longer resets the counter.
    """
    if track_id in job.child_to_truck:
        return

    votes = job.child_assoc_votes.setdefault(track_id, {})
    key = (truck_tid, truck_type)
    votes[key] = votes.get(key, 0) + 1
    best_key = max(votes, key=lambda k: votes[k])
    best_count = votes[best_key]

    print(f"  [VOTE] child T#{track_id} ({cls_name}) "
          f"→ Truck T#{best_key[0]}  votes={best_count}/{LOCK_AFTER_N_FRAMES}")

    if best_count >= LOCK_AFTER_N_FRAMES:
        job.child_to_truck[track_id] = {
            "truck_tid": best_key[0],
            "truck_type": best_key[1],
            "field": cls_name,
        }
        del job.child_assoc_votes[track_id]
        print(f"  [LOCKED] ✅ child T#{track_id} ({cls_name}) "
              f"→ Truck T#{best_key[0]} ({best_key[1]}) "
              f"after majority vote ({best_count} frames)")


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
    # Build truck_list with union_bbox
    truck_list: list[tuple[tuple, tuple, int, str]] = []  # (last_box, union_box, tid, type)
    child_list: list[tuple[int, tuple, str, int | None]] = []

    for i, (x1, y1, x2, y2, cls_id, _, tid, _mask) in enumerate(boxes_data):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        box = (x1, y1, x2, y2)
        if cls_name in TRUCK_CLASSES and tid is not None:
            union_box = box  # fallback
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
            # Update voting for unlocked children
            if job is not None:
                _update_association_votes(job, child_tid, child_cls, best_match[0], best_match[1])

        # Stage 2: Sticky fallback (only if not locked and spatial failed)
        elif job is not None and child_tid in job.sticky_assoc:
            sticky = job.sticky_assoc[child_tid]
            assoc[child_idx] = sticky

    return assoc


# ========================= PERMANENT TRUCK ID ================================
def _get_perm_truck_id(
    job: VideoJob,
    raw_id: int,
    bbox: tuple[int, int, int, int],
    registry: TruckRegistry,
    frame_id: int,
) -> int:
    """
    Map a raw ByteTrack ID to a stable permanent ID.
    If the raw ID was seen before, returns the existing mapping.
    If the bbox overlaps a recent truck in the registry (score >= IOU_MERGE_THRESH),
    merges into that truck's perm ID.
    Otherwise assigns a fresh perm ID.
    """
    if raw_id in job.raw_to_perm_id:
        return job.raw_to_perm_id[raw_id]

    best_perm: int | None = None
    best_score = 0.0
    with registry._lock:
        for perm_id, rec in registry.trucks.items():
            if frame_id - rec.get("last_seen_frame", 0) > ACTIVE_WINDOW_FRAMES:
                continue
            lb = tuple(rec["last_bbox"])
            ub = tuple(rec["union_bbox"])
            score = max(
                _iou(bbox, lb),       _containment(bbox, lb),  _containment(lb, bbox),
                _iou(bbox, ub),       _containment(bbox, ub),  _containment(ub, bbox),
            )
            if score > best_score:
                best_score = score
                best_perm = perm_id

    if best_perm is not None and best_score >= IOU_MERGE_THRESH:
        job.raw_to_perm_id[raw_id] = best_perm
        print(f"  [PERM-ID] raw T#{raw_id} merged → perm T#{best_perm} (score={best_score:.2f})")
        return best_perm

    perm_id = job._next_perm_id
    job._next_perm_id += 1
    job.raw_to_perm_id[raw_id] = perm_id
    print(f"  [PERM-ID] raw T#{raw_id} → new perm T#{perm_id}")
    return perm_id


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
    _UNIFIED_TEMPLATE = {
        "container_number": None,
        "container_side_no": None,
        "container_company_logo": None,
        "other_container_info": None,
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
        template = self._UNIFIED_TEMPLATE
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
            "union_bbox": list(bbox),
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
            ub = rec["union_bbox"]
            rec["union_bbox"] = [
                min(ub[0], bbox[0]), min(ub[1], bbox[1]),
                max(ub[2], bbox[2]), max(ub[3], bbox[3]),
            ]
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
                    print(
                        f"  [GHOST-FILTER] skip T#{tid} duration_frames="
                        f"{rec.get('duration_frames', 1)} < {MIN_TRUCK_TRACK_FRAMES}"
                    )
                    continue
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


def _save_detection_crop(
    frame: np.ndarray,
    cls_name: str,
    frame_id: int,
    det_index: int,
    conf: float,
    track_id: int | None,
    box: tuple[int, int, int, int],
) -> None:
    if not SAVE_DETECTION_CROPS:
        return
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return
    safe_class = re.sub(r"[^A-Za-z0-9_.-]", "_", cls_name)
    class_dir = os.path.join(OUTPUT_DIR, "detections", safe_class)
    os.makedirs(class_dir, exist_ok=True)
    tid = f"t{track_id}" if track_id is not None else "no_track"
    stem = f"f{frame_id:06d}_i{det_index:03d}_{tid}_conf{conf:.3f}"
    image_path = os.path.join(class_dir, f"{stem}.jpg")
    meta_path = os.path.join(class_dir, f"{stem}.json")
    cv2.imwrite(image_path, crop)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_name": cls_name,
                "confidence": round(conf, 6),
                "track_id": track_id,
                "frame_id": frame_id,
                "detection_index": det_index,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop_path": image_path,
            },
            f,
            indent=2,
        )


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
            m = mask
            ov = frame.copy()
            ov[m > 0.5] = colour
            cv2.addWeighted(ov, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
            ctrs, _ = cv2.findContours(
                (m > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(frame, ctrs, -1, colour, 2)
        elif cls_name in TRUCK_CLASSES:
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), colour, -1)
            cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # ── Label: show lock status (from snippet 1) ───────────────────────────
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
                        candidate = job.registry.find_truck_for_field(field, current_frame=frame_id)
                        if candidate is not None:
                            resolved_tid = candidate
                            rec = job.registry.trucks.get(candidate)
                            resolved_type = rec["type"] if rec else ""
                            job.registry.attach_ocr(resolved_tid, resolved_type, field, plain_text, conf, image_path)
                            log += f"  (orphan → T#{resolved_tid})"
                            print(f"  [OCR] ↩ orphan '{field}' rescued → T#{resolved_tid}")
                        else:
                            _bad_desc = any(
                                t in plain_text.lower()
                                for t in ("close-up", "no visible", "no text", "not visible", "metallic", "cylindrical")
                            )
                            if not _bad_desc and len(plain_text) <= 60 and len(job.orphan_buffer) < 30:
                                job.orphan_buffer.append({
                                    "field": field,
                                    "text": plain_text,
                                    "conf": conf,
                                    "image": image_path,
                                })
                                log += "  (orphan — buffered for retry)"
                                print(f"  [OCR] ⏳ orphan '{field}' buffered → '{plain_text}'")
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
    tracking_model = _load_tracking_model()
    print(
        f"[PlateFlow] Video job tracker isolated: model_id={id(tracking_model)} path={video_path}",
        flush=True,
    )
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
        "confidence": CONF,
        "truck_confidence": TRUCK_CONF_THRESH,
        "infer_width": INFER_WIDTH,
        "tracker": YOLO_TRACKER,
        "save_detection_crops": SAVE_DETECTION_CROPS,
        "lock_after_n_frames": LOCK_AFTER_N_FRAMES,  # Record locking config
        "enable_frame_skip": ENABLE_FRAME_SKIP,
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES if ENABLE_FRAME_SKIP else 1,
        "ocr_every_n_frames": OCR_EVERY_N_FRAMES,
    }

    registry = TruckRegistry(fps=fps)
    if job is not None:
        _set_job(job.job_id, registry=registry, session_info=session_info, total_frames=total_frames)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_DETECTION_CROPS:
        os.makedirs(os.path.join(OUTPUT_DIR, "detections"), exist_ok=True)

    frame_id   = 0
    t_prev     = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            break

        # ── Frame-skip ────────────────────────────────────────────────────────
        if FRAME_SKIP > 0 and frame_id % (FRAME_SKIP + 1) != 0:
            frame_id += 1
            if job is not None:
                progress = (frame_id / total_frames) if total_frames > 0 else 0.0
                _set_job(
                    job.job_id,
                    progress=progress,
                    frame_id=frame_id,
                    message=(
                        f"Skipping frame {frame_id}/{total_frames}; "
                        f"processing every {PROCESS_EVERY_N_FRAMES} frame(s)."
                    ),
                )
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        frame_small, scale = _resize_for_inference(frame_orig)
        t0      = time.time()
        try:
            results = tracking_model.track(
                frame_small,
                conf=CONF,
                iou=IOU_THRESH,
                tracker=YOLO_TRACKER,
                persist=True,
                verbose=False,
                half=USE_HALF,
                device=DEVICE,
                retina_masks=True,
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
                _save_detection_crop(
                    frame_orig,
                    cls_name,
                    frame_id,
                    i,
                    conf_val,
                    track_id,
                    (x1, y1, x2, y2),
                )

        boxes_data = _dedupe_truck_detections(boxes_data)

        # Remap raw ByteTrack truck IDs → stable permanent IDs before any
        # downstream code (registry, association, OCR) sees them.
        if job is not None:
            remapped: list[tuple] = []
            for x1, y1, x2, y2, cls_id, conf_val, track_id, mask in boxes_data:
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                if cls_name in TRUCK_CLASSES and track_id is not None:
                    perm_id = _get_perm_truck_id(job, track_id, (x1, y1, x2, y2), registry, frame_id)
                    remapped.append((x1, y1, x2, y2, cls_id, conf_val, perm_id, mask))
                else:
                    remapped.append((x1, y1, x2, y2, cls_id, conf_val, track_id, mask))
            boxes_data = remapped

        for x1, y1, x2, y2, cls_id, conf_val, track_id, _mask in boxes_data:
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            if cls_name in TRUCK_CLASSES and track_id is not None:
                registry.update(track_id, cls_name, (x1, y1, x2, y2), frame_id, conf_val)

        # ── Flush buffered orphan OCR results every 5 frames ────────────────────
        if job is not None and frame_id % 5 == 0 and job.orphan_buffer:
            with _jobs_lock:
                remaining = []
                for item in job.orphan_buffer:
                    candidate = registry.find_truck_for_field(item["field"], current_frame=frame_id)
                    if candidate is not None:
                        rec = registry.trucks.get(candidate)
                        registry.attach_ocr(
                            candidate, rec["type"] if rec else "", item["field"],
                            item["text"], item["conf"], item.get("image"),
                        )
                        rescued_log = f"[f{frame_id}] {item['field']} rescued → T#{candidate} '{item['text']}'"
                        job.ocr_log.append(rescued_log)
                        job.ocr_log = job.ocr_log[-60:]
                        print(f"  [OCR] ✅ orphan rescued: {rescued_log}")
                    else:
                        remaining.append(item)
                job.orphan_buffer = remaining

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
            if frame_id % OCR_EVERY_N_FRAMES != 0:
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

            # Best-confidence tracking: always use the highest-confidence crop per truck field.
            # We check even when the key is already submitted so a better crop supersedes the old one.
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
                            print(
                                f"  [OCR-BEST] {field} owner#{owner_id} "
                                f"conf={conf_val:.3f} > prev={prev_best:.3f}"
                            )
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

        # ── pending_track_ids for "scanning…" overlay ─────────────────────────
        pending_track_ids: set[int] = {
            int(k[1:]) for k in pending if k.startswith("t") and k[1:].isdigit()
        }

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
        _run_video_analysis(video_path, job=job)

        # Wait for all async OCR tasks to finish before returning the result.
        # MinerU calls take 20-60 s each; we poll until pending_keys is empty.
        deadline = time.time() + 600  # 10-minute hard cap
        while time.time() < deadline:
            with _jobs_lock:
                n_pending = len(JOBS[job_id].pending_keys)
            if n_pending == 0:
                break
            _set_job(
                job_id,
                message=f"Frames processed — waiting for {n_pending} OCR task(s) to complete…",
            )
            print(f"  [OCR-WAIT] {n_pending} task(s) still pending…")
            time.sleep(2)

        with _jobs_lock:
            job = JOBS[job_id]
        final_result = (
            job.registry.to_dict(job.session_info)
            if job.registry is not None
            else {}
        )
        _set_job(job_id, state="completed", progress=1.0, result=final_result,
                 json_snapshot=json.dumps(final_result, indent=2),
                 message="Video processing completed.")
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
                xyxy     = box.xyxy[0].tolist()
                cls_id   = int(box.cls[0].item()) if box.cls is not None else -1
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
                        _save_detection_crop(
                            frame,
                            cls_name,
                            frame_id=0,
                            det_index=det_index,
                            conf=conf_val,
                            track_id=None,
                            box=(x1, y1, x2, y2),
                        )

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