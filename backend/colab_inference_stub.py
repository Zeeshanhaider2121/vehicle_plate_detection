"""
Colab GPU FastAPI service with live video-job progress and optional MinerU OCR.

Run in Colab:
1) pip install fastapi uvicorn python-multipart ultralytics opencv-python-headless requests
2) export MODEL_PATH=/content/drive/.../best.pt
3) export MINERU_TOKEN=<your_token>   (optional but required for OCR text)
4) python colab_inference_stub.py
5) expose port 8001 via ngrok/cloudflared
"""

from __future__ import annotations

import io
import os
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
import requests
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "/content/drive/MyDrive/vehicle_plate/output/runs/yolo26x_seg_phase2/weights/best.pt")
MINERU_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI5NTYwMDA2MiIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3ODc1MTM1MywiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiODA1Yzc0MTYtZmM1Zi00MmQ0LTljZjctZDA5YzM1MjNhM2I5IiwiZW1haWwiOiIiLCJleHAiOjE3ODY1MjczNTN9.0Fwe0jwakra4gmjjxzyzIKjTAtFubWdhJwlrzU5rBINwM5aqmgPb508Vgv-Z0P9XZYAMP6t-SlK4dMt5_cytxQ"


CONF = float(os.getenv("CONF", "0.23"))
TRUCK_CONF_THRESH = float(os.getenv("TRUCK_CONF_THRESH", "0.92"))
IOU_THRESH = float(os.getenv("IOU_THRESH", "0.50"))
INFER_WIDTH = int(os.getenv("INFER_WIDTH", "960"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "0"))
MIN_ASSOC_SCORE = float(os.getenv("MIN_ASSOC_SCORE", "0.10"))
MAX_OCR_PER_FRAME = int(os.getenv("MAX_OCR_PER_FRAME", "2"))

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
    "driver",
}

app = FastAPI(title="PlateFlow Colab GPU API", version="2.0.0")
model = YOLO(MODEL_PATH)
_jobs_lock = threading.Lock()


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


JOBS: dict[str, VideoJob] = {}


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


def _resize_for_inference(frame: Any) -> tuple[Any, float]:
    h, w = frame.shape[:2]
    if w <= INFER_WIDTH:
        return frame, 1.0
    scale = INFER_WIDTH / w
    return cv2.resize(frame, (INFER_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR), scale


def _parse_markdown(md_text: str) -> str:
    details = re.findall(r"<details>.*?<summary>[^<]*</summary>\s*(.*?)\s*</details>", md_text, flags=re.DOTALL)
    if details:
        text = " | ".join(chunk.strip() for chunk in details if chunk.strip())
        if text:
            return text
    lines = [line.strip() for line in md_text.splitlines() if line.strip() and not line.strip().startswith("![")]
    return " ".join(lines).strip()


def _extract_text_from_zip(zip_url: str) -> str:
    try:
        response = requests.get(zip_url, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for name in archive.namelist():
                if name.endswith("full.md"):
                    return _parse_markdown(archive.read(name).decode("utf-8", errors="ignore"))
    except Exception:
        return ""
    return ""


def _call_mineru_ocr(crop_bgr: Any) -> str:
    if not MINERU_TOKEN:
        return ""

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    _, encoded = cv2.imencode(".png", crop_rgb)
    filename = f"crop_{uuid.uuid4().hex[:8]}.png"
    data_id = uuid.uuid4().hex
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}",
    }
    try:
        step1 = requests.post(
            "https://mineru.net/api/v4/file-urls/batch",
            json={"files": [{"name": filename, "data_id": data_id}], "model_version": "vlm"},
            headers=headers,
            timeout=15,
        )
        step1.raise_for_status()
        data = step1.json()
        if data.get("code") != 0:
            return ""
        batch_id = data["data"]["batch_id"]
        upload_url = data["data"]["file_urls"][0]
        put = requests.put(upload_url, data=encoded.tobytes(), timeout=30)
        if put.status_code not in (200, 201):
            return ""

        poll_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        for _ in range(20):
            time.sleep(3)
            poll = requests.get(poll_url, headers=headers, timeout=15)
            poll.raise_for_status()
            results = poll.json().get("data", {}).get("extract_result", [])
            if not results:
                continue
            state = results[0].get("state")
            if state == "done":
                return _extract_text_from_zip(results[0].get("full_zip_url", ""))
            if state == "failed":
                return ""
    except Exception:
        return ""
    return ""


class TruckRegistry:
    def __init__(self, fps: float) -> None:
        self.fps = max(fps, 1.0)
        self.trucks: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _template(self, truck_type: str) -> dict[str, Any]:
        if truck_type == "truck_with_container":
            return {
                "container_number": None,
                "container_side_no": None,
                "container_company_logo": None,
                "other_container_info": None,
                "driver": None,
            }
        return {
            "license_plate": None,
            "truck_company": None,
            "truck_number": None,
            "driver": None,
        }

    def update_truck(self, track_id: int, truck_type: str, bbox: tuple[int, int, int, int], frame_id: int, conf: float) -> None:
        with self._lock:
            if track_id not in self.trucks:
                self.trucks[track_id] = {
                    "track_id": track_id,
                    "type": truck_type,
                    "first_seen_frame": frame_id,
                    "last_seen_frame": frame_id,
                    "first_seen_time_sec": round(frame_id / self.fps, 3),
                    "last_seen_time_sec": round(frame_id / self.fps, 3),
                    "duration_frames": 1,
                    "duration_sec": round(1.0 / self.fps, 3),
                    "confidence_avg": round(conf, 4),
                    "_conf_n": 1,
                    "last_bbox": list(bbox),
                    "associated_info": self._template(truck_type),
                }
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

    def attach_field(self, track_id: int, field: str, value: str) -> None:
        if not value:
            return
        with self._lock:
            rec = self.trucks.get(track_id)
            if not rec:
                return
            info = rec["associated_info"]
            if field not in info:
                return
            existing = info.get(field) or ""
            if len(value) >= len(existing):
                info[field] = value

    def to_dict(self, session_info: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            trucks_out: dict[str, Any] = {}
            for tid, rec in self.trucks.items():
                clean = {k: v for k, v in rec.items() if not k.startswith("_")}
                clean["associated_info"] = dict(rec["associated_info"])
                trucks_out[str(tid)] = clean

        twc = sum(1 for item in trucks_out.values() if item["type"] == "truck_with_container")
        twoc = sum(1 for item in trucks_out.values() if item["type"] == "truck_without_container")
        return {
            "session": session_info,
            "summary": {
                "total_trucks_tracked": len(trucks_out),
                "trucks_with_container": twc,
                "trucks_without_container": twoc,
            },
            "trucks": trucks_out,
        }


def _set_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        job = JOBS[job_id]
        for key, value in kwargs.items():
            setattr(job, key, value)


def _draw_preview(frame_bgr: Any, detections: list[dict[str, Any]]) -> bytes | None:
    view = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class_name"]
        conf = det["conf"]
        cv2.rectangle(view, (x1, y1), (x2, y2), (88, 170, 255), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(view, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 245, 255), 2)
    ok, encoded = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    return encoded.tobytes()


def _build_association_map(boxes_data: list[dict[str, Any]]) -> dict[int, tuple[int, str, str]]:
    truck_list: list[tuple[tuple[int, int, int, int], int, str]] = []
    child_list: list[tuple[int, tuple[int, int, int, int], str]] = []

    for idx, det in enumerate(boxes_data):
        cls_name = det["class_name"]
        bbox = det["bbox"]
        if cls_name in TRUCK_CLASSES and det["track_id"] is not None:
            truck_list.append((bbox, det["track_id"], cls_name))
        elif cls_name in OCR_CLASSES:
            child_list.append((idx, bbox, cls_name))

    assoc: dict[int, tuple[int, str, str]] = {}
    for child_idx, child_box, child_class in child_list:
        best_score = 0.0
        best_match: tuple[int, str, str] | None = None
        for truck_box, truck_track_id, truck_type in truck_list:
            score = max(_containment(child_box, truck_box), _iou(child_box, truck_box))
            if score > best_score:
                best_score = score
                best_match = (truck_track_id, truck_type, child_class)
        if best_match and best_score >= MIN_ASSOC_SCORE:
            assoc[child_idx] = best_match
    return assoc


def _run_video_analysis(video_path: str, progress_cb: callable | None = None, job: VideoJob | None = None) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    registry = TruckRegistry(fps=fps)
    frame_id = 0
    started_at = datetime.now().isoformat(timespec="seconds")
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if FRAME_SKIP > 0 and frame_id % (FRAME_SKIP + 1) != 0:
            frame_id += 1
            continue

        resized, scale = _resize_for_inference(frame)
        tracked = model.track(
            resized,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF,
            iou=IOU_THRESH,
            verbose=False,
        )
        boxes = tracked[0].boxes if tracked else None
        boxes_data: list[dict[str, Any]] = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                sx1, sy1, sx2, sy2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, int(sx1 / scale))
                y1 = max(0, int(sy1 / scale))
                x2 = min(width, int(sx2 / scale))
                y2 = min(height, int(sy2 / scale))
                if x2 <= x1 or y2 <= y1:
                    continue
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                track_id = int(box.id[0]) if box.id is not None else None
                if class_name in TRUCK_CLASSES and conf_val < TRUCK_CONF_THRESH:
                    continue
                det = {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "conf": conf_val,
                    "track_id": track_id,
                }
                boxes_data.append(det)
                if class_name in TRUCK_CLASSES and track_id is not None:
                    registry.update_truck(track_id, class_name, det["bbox"], frame_id, conf_val)

        assoc = _build_association_map(boxes_data)
        ocr_calls_this_frame = 0
        for idx, det in enumerate(boxes_data):
            cls_name = det["class_name"]
            if cls_name not in OCR_CLASSES:
                continue
            if idx not in assoc:
                continue
            truck_track_id, _, field_name = assoc[idx]
            if truck_track_id is None:
                continue

            key = f"{truck_track_id}:{field_name}"
            cached_text = job.ocr_cache.get(key) if job else None
            if cached_text:
                registry.attach_field(truck_track_id, field_name, cached_text)
                continue

            if ocr_calls_this_frame >= MAX_OCR_PER_FRAME:
                continue
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            text = _call_mineru_ocr(crop)
            if text:
                registry.attach_field(truck_track_id, field_name, text)
                if job is not None:
                    job.ocr_cache[key] = text
            ocr_calls_this_frame += 1

        preview = _draw_preview(frame, boxes_data)
        if preview is not None and job is not None:
            _set_job(job.job_id, latest_frame_jpeg=preview)

        frame_id += 1
        t_now = time.time()
        fps_runtime = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        if progress_cb:
            progress_cb(frame_id, total_frames, fps_runtime)

    cap.release()
    session_info = {
        "video_path": video_path,
        "total_frames": total_frames,
        "video_fps": round(fps, 2),
        "resolution": f"{width}x{height}",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "frames_processed": frame_id,
        "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") else "cpu",
        "model": MODEL_PATH,
    }
    return registry.to_dict(session_info)


def _to_image_detections(image_path: str) -> list[dict[str, Any]]:
    results = model.predict(image_path, verbose=False, conf=CONF, iou=IOU_THRESH)
    if not results:
        return []
    result = results[0]
    names = result.names or {}
    items: list[dict[str, Any]] = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0
        items.append(
            {
                "plate_text": names.get(cls_id, f"class_{cls_id}"),
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                },
            }
        )
    return items


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = JOBS[job_id]
        video_path = job.temp_video_path
    if not video_path:
        _set_job(job_id, state="failed", error="Missing temp video path", message="Job failed.")
        return
    try:
        _set_job(job_id, state="running", message="Video processing started.")

        def progress_cb(frame_id: int, total_frames: int, fps: float) -> None:
            progress = (frame_id / total_frames) if total_frames > 0 else 0.0
            _set_job(
                job_id,
                progress=round(progress, 4),
                frame_id=frame_id,
                total_frames=total_frames,
                fps=round(fps, 2),
                message=f"Processing frame {frame_id}/{total_frames}",
            )

        result = _run_video_analysis(video_path, progress_cb=progress_cb, job=job)
        _set_job(job_id, state="completed", progress=1.0, result=result, message="Video processing completed.")
    except Exception as exc:  # pragma: no cover
        _set_job(job_id, state="failed", error=str(exc), message=f"Video processing failed: {exc}")
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


@app.post("/infer")
async def infer(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / (file.filename or "upload.jpg")
        image_path.write_bytes(raw)
        detections = _to_image_detections(str(image_path))
    return {"detections": detections, "filename": file.filename}


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty video upload.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = Path(tmp_dir) / (file.filename or "upload.mp4")
        video_path.write_bytes(raw)
        return _run_video_analysis(str(video_path))


@app.post("/analyze-video/start")
async def analyze_video_start(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty video upload.")
    job_id = uuid.uuid4().hex
    temp_dir = tempfile.mkdtemp(prefix="plateflow_job_")
    video_path = Path(temp_dir) / (file.filename or "upload.mp4")
    video_path.write_bytes(raw)

    with _jobs_lock:
        JOBS[job_id] = VideoJob(
            job_id=job_id,
            state="queued",
            message="Job queued.",
            temp_video_path=str(video_path),
        )
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id, "state": "queued", "message": "Video job created."}


@app.get("/analyze-video/jobs/{job_id}")
def analyze_video_job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "state": job.state,
        "progress": job.progress,
        "frame_id": job.frame_id,
        "total_frames": job.total_frames,
        "fps": job.fps,
        "message": job.message,
        "error": job.error,
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
