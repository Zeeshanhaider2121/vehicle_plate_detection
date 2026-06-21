"""
Lane-setup router — draw lane ROIs + a physical gate line in the browser.

The operator draws, once, on a real frame from each gate camera (front/left/right):
  * two lane polygons  -> per-detection lane_id (the cross-lane fix), and
  * one horizontal gate line -> the crossing tripwire (Subsystem 2, engine side).

We persist them as ``{camera}_lane_rois.json`` in the SAME directory the engine's
``local_inference._load_lane_rois`` reads from, in its existing format:

    {"camera": "right", "image_width": 1280, "image_height": 720,
     "lanes": {"1": [[x,y],...], "2": [[x,y],...]},
     "gate_line": [[x1,y1],[x2,y2]]}

The engine reads only ``lanes`` today and ignores ``gate_line`` and the size hints,
so this file is backward compatible. Coordinates are NATIVE image pixels (the
frontend maps display coords -> native before sending).

This module does NOT import or modify main.py or the engine. It is wired with a single
``include_router`` line, exactly like the aggregator.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# The engine resolves ROI files relative to local_inference.py, which lives in backend/.
# This module is app/lane_setup.py, so backend/ is two parents up.
_ENGINE_ROI_DIR = Path(__file__).resolve().parents[1]

# A gate camera's lanes mean nothing for the interior camera; mirror the spec's roles.
GATE_CAMERAS = ("front", "left", "right")


def roi_path_for(camera: str) -> Path:
    """Where the engine looks first: backend/{camera}_lane_rois.json."""
    return _ENGINE_ROI_DIR / f"{camera.lower()}_lane_rois.json"


def _write_temp_clip(video_bytes: bytes, suffix: str = ".mp4") -> Path:
    """Persist upload to a temp file KEEPING the extension — OpenCV's decoder backend
    is far more reliable when the container is identifiable by suffix."""
    safe = suffix if (suffix and suffix.startswith(".") and len(suffix) <= 6) else ".mp4"
    tmp = Path(tempfile.mkdtemp(prefix="lane_setup_")) / f"clip{safe}"
    tmp.write_bytes(video_bytes)
    return tmp


def _cleanup_temp(tmp: Path) -> None:
    try:
        tmp.unlink()
        tmp.parent.rmdir()
    except OSError:
        pass


def _is_clean_frame(frame) -> bool:
    """Reject black frames AND macroblock-garbage frames.

    A frame decoded from a non-keyframe without its reference comes out as flat grey
    with blocky residuals: brightness sits in a mid band with very LOW spatial variance.
    A real scene has meaningful structure (higher std). So require: not near-black, and
    enough detail (std) to be a genuine image rather than decode mush.
    """
    if float(frame.mean()) < 12.0:
        return False  # black / warm-up
    if float(frame.std()) < 18.0:
        return False  # flat grey or garbled decode mush
    return True


def _extract_representative_frame(video_bytes: bytes, suffix: str = ".mp4") -> bytes:
    """
    Grab one clean still frame from a video to draw on. We decode SEQUENTIALLY (not by
    seeking) so every frame is reconstructed from its keyframe in order — seeking to a
    non-keyframe is what produced the grey macroblock-garbage frames. We skip the first
    couple seconds (warm-up) and return the first frame that is neither black nor garbled.
    """
    import cv2  # local import: keeps module importable where cv2 is absent

    tmp = _write_temp_clip(video_bytes, suffix)
    cap = cv2.VideoCapture(str(tmp))
    try:
        if not cap.isOpened():
            raise ValueError("could not open the uploaded video (unsupported codec?)")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_at = int(fps * 2)                       # skip ~2s of warm-up
        # bound how far we'll scan so a long clip can't hang the request
        max_scan = total if total > 0 else int(fps * 120)
        max_scan = min(max_scan, start_at + int(fps * 120))

        best = None
        best_score = -1.0
        idx = 0
        while idx < max_scan:
            ok, frame = cap.read()                    # sequential decode -> clean frames
            if not ok or frame is None:
                break
            score = float(frame.std())                # detail as a quality proxy
            if score > best_score:
                best_score, best = score, frame
            if idx >= start_at and _is_clean_frame(frame):
                ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok2:
                    return buf.tobytes()
            idx += 1

        # nothing passed the clean test — fall back to the most-detailed frame we saw
        if best is None:
            raise ValueError("could not read any frame from the uploaded video")
        ok, buf = cv2.imencode(".jpg", best, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("could not encode the extracted frame")
        return buf.tobytes()
    finally:
        cap.release()
        _cleanup_temp(tmp)


def test_lanes_on_video(camera: str, video_bytes: bytes,
                        lanes: Dict[str, List[List[float]]],
                        samples: int = 8, suffix: str = ".mp4") -> dict:
    """
    Run the engine's truck detector on a handful of sampled frames and label each
    detected truck by which DRAFT lane polygon its bottom-centre falls in. Returns
    the single frame with the most trucks (base64 JPEG) plus the truck boxes, so the
    operator can confirm "lane 1 truck -> lane 1" BEFORE saving. Reuses the engine's
    model + thresholds; does not modify it.
    """
    import base64

    import cv2
    import numpy as np

    import local_inference as li  # lazy: loads the YOLO model on first call

    rois = {int(k): np.array(v, np.int32) for k, v in lanes.items() if len(v) >= 3}

    tmp = _write_temp_clip(video_bytes, suffix)
    cap = cv2.VideoCapture(str(tmp))
    try:
        if not cap.isOpened():
            raise ValueError("could not open the uploaded video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Decode SEQUENTIALLY (seeking to non-keyframes yields garbled frames) and run
        # detection at even strides on CLEAN frames only.
        scan_cap = total if total > 0 else int(fps * 120)
        step = max(1, scan_cap // (samples + 1))
        sample_idxs = {step * (i + 1) for i in range(samples)}

        best = {"frame": None, "trucks": [], "w": 0, "h": 0}
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            idx += 1
            if idx > scan_cap:
                break
            if idx not in sample_idxs or not _is_clean_frame(frame):
                continue
            results = li.model.predict(
                frame, conf=li.TRUCK_CONF_THRESH, verbose=False,
                half=li.USE_HALF, device=li.DEVICE,
            )
            boxes = results[0].boxes if results else None
            trucks = []
            for b in (boxes or []):
                cls_id = int(b.cls[0])
                cls_name = li.CLASS_NAMES[cls_id] if cls_id < len(li.CLASS_NAMES) else str(cls_id)
                if cls_name not in li.TRUCK_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                lane = li._get_detection_lane((x1, y1, x2, y2), rois)
                trucks.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "type": cls_name, "confidence": round(float(b.conf[0]), 4),
                    "lane": lane,
                })
            if len(trucks) >= len(best["trucks"]):
                h, w = frame.shape[:2]
                best = {"frame": frame, "trucks": trucks, "w": w, "h": h}

        if best["frame"] is None:
            raise ValueError("could not read any frame to test")

        ok, buf = cv2.imencode(".jpg", best["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        return {
            "camera": camera.lower(),
            "image_width": best["w"],
            "image_height": best["h"],
            "frame_jpeg_base64": jpeg_b64,
            "trucks": best["trucks"],
            "truck_count": len(best["trucks"]),
        }
    finally:
        cap.release()
        _cleanup_temp(tmp)


def save_rois(camera: str, image_width: int, image_height: int,
              lanes: Dict[str, List[List[float]]],
              gate_line: Optional[List[List[float]]]) -> Path:
    """Write the engine-format ROI file (+ gate_line) for one camera. Returns its path."""
    # round to int pixels — pointPolygonTest wants integer-ish polygons anyway
    clean_lanes = {
        str(k): [[int(round(x)), int(round(y))] for x, y in pts]
        for k, pts in lanes.items()
    }
    payload: Dict[str, object] = {
        "camera": camera.lower(),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "lanes": clean_lanes,
    }
    if gate_line:
        payload["gate_line"] = [[int(round(x)), int(round(y))] for x, y in gate_line]

    out = roi_path_for(camera)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_rois(camera: str) -> Optional[dict]:
    """Read back a saved ROI file for re-editing, or None if not set yet."""
    path = roi_path_for(camera)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# FastAPI router                                                              #
# --------------------------------------------------------------------------- #

try:
    from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
    from pydantic import BaseModel, Field

    router = APIRouter(prefix="/api/lane-setup", tags=["lane-setup"])

    class SaveRoisRequest(BaseModel):
        camera: str
        image_width: int = Field(gt=0)
        image_height: int = Field(gt=0)
        # {"1": [[x,y],...], "2": [[x,y],...]}
        lanes: Dict[str, List[List[float]]]
        # [[x1,y1],[x2,y2]] — optional until the operator draws it
        gate_line: Optional[List[List[float]]] = None

    def _require_gate_camera(camera: str) -> str:
        cam = camera.lower()
        if cam not in GATE_CAMERAS:
            raise HTTPException(
                status_code=400,
                detail=f"Lane setup applies to gate cameras {GATE_CAMERAS}, not '{camera}'.",
            )
        return cam

    @router.post("/extract-frame")
    async def extract_frame(video: UploadFile = File(...),
                            camera: str = Form(...)) -> Response:
        """Return one still JPEG from the uploaded camera video to draw on."""
        _require_gate_camera(camera)
        data = await video.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty video upload.")
        suffix = Path(video.filename or "").suffix or ".mp4"
        try:
            jpeg = _extract_representative_frame(data, suffix=suffix)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ImportError as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"OpenCV unavailable: {exc}") from exc
        return Response(content=jpeg, media_type="image/jpeg")

    @router.post("/test-lanes")
    async def test_lanes_endpoint(video: UploadFile = File(...),
                                  camera: str = Form(...),
                                  lanes: str = Form(...),
                                  samples: int = Form(8)) -> dict:
        """Detect trucks on sampled frames and label each by the draft lane polygons."""
        _require_gate_camera(camera)
        data = await video.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty video upload.")
        try:
            parsed = json.loads(lanes)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"lanes must be JSON: {exc}") from exc
        if not isinstance(parsed, dict) or not parsed:
            raise HTTPException(status_code=400, detail="lanes must be a non-empty object.")
        suffix = Path(video.filename or "").suffix or ".mp4"
        try:
            return test_lanes_on_video(camera, data, parsed,
                                       samples=max(1, min(samples, 30)), suffix=suffix)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ImportError as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Engine/OpenCV unavailable: {exc}") from exc

    @router.post("/save")
    def save_endpoint(req: SaveRoisRequest) -> dict:
        cam = _require_gate_camera(req.camera)
        if not req.lanes:
            raise HTTPException(status_code=400, detail="At least one lane polygon required.")
        for name, pts in req.lanes.items():
            if len(pts) < 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Lane '{name}' needs at least 3 points (got {len(pts)}).",
                )
        if req.gate_line is not None and len(req.gate_line) != 2:
            raise HTTPException(status_code=400, detail="gate_line must be exactly two points.")
        path = save_rois(cam, req.image_width, req.image_height, req.lanes, req.gate_line)
        return {"saved": True, "camera": cam, "path": str(path),
                "lanes": list(req.lanes.keys()),
                "has_gate_line": req.gate_line is not None}

    @router.get("/{camera}")
    def get_endpoint(camera: str) -> dict:
        cam = _require_gate_camera(camera)
        data = load_rois(cam)
        return {"camera": cam, "exists": data is not None, "roi": data}

except ImportError:  # FastAPI not installed — module still importable for unit tests
    router = None
