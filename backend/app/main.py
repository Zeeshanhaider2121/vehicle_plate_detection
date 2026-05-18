import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .database import Base, engine, get_db
from .inference_client import InferenceService
from .models import Detection, DetectionStatus, TruckRecord, TruckRun
from .schemas import (
    DetectResponse,
    DetectionListResponse,
    DetectionRead,
    HealthResponse,
    RejectRequest,
    TruckRecordListResponse,
    TruckRecordRead,
    TruckRunImportRequest,
    TruckRunImportResponse,
    TruckRunListResponse,
    TruckRunRead,
    VideoAnalyzeFinalizeResponse,
    VideoAnalyzeStartResponse,
    VideoAnalyzeStatusResponse,
    VerifyRequest,
)
from .video_client import ColabVideoClient

app = FastAPI(title=settings.app_name, version="0.1.0")
inference_service = InferenceService()
video_client = ColabVideoClient()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


def _loads_json_or_none(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _truck_record_to_read(record: TruckRecord) -> TruckRecordRead:
    return TruckRecordRead(
        id=record.id,
        run_id=record.run_id,
        track_id=record.track_id,
        truck_type=record.truck_type,
        first_seen_frame=record.first_seen_frame,
        last_seen_frame=record.last_seen_frame,
        first_seen_time_sec=record.first_seen_time_sec,
        last_seen_time_sec=record.last_seen_time_sec,
        duration_frames=record.duration_frames,
        duration_sec=record.duration_sec,
        confidence_avg=record.confidence_avg,
        last_bbox=_loads_json_or_none(record.last_bbox_json),
        associated_info=_loads_json_or_none(record.associated_info_json),
        created_at=record.created_at,
    )


def _store_truck_run(payload: TruckRunImportRequest, db: Session) -> TruckRunImportResponse:
    run_id = uuid4().hex
    run_row = TruckRun(
        run_id=run_id,
        video_path=payload.session.video_path,
        total_frames=payload.session.total_frames,
        video_fps=payload.session.video_fps,
        resolution=payload.session.resolution,
        started_at=payload.session.started_at,
        finished_at=payload.session.finished_at,
        device=payload.session.device,
        model=payload.session.model,
        frames_processed=payload.session.frames_processed,
        total_trucks_tracked=payload.summary.total_trucks_tracked,
        trucks_with_container=payload.summary.trucks_with_container,
        trucks_without_container=payload.summary.trucks_without_container,
    )
    db.add(run_row)

    stored_count = 0
    for key, truck in payload.trucks.items():
        try:
            fallback_track_id = int(key)
        except ValueError:
            fallback_track_id = -1
        track_id = truck.track_id if truck.track_id is not None else fallback_track_id
        row = TruckRecord(
            run_id=run_id,
            track_id=track_id,
            truck_type=truck.type,
            first_seen_frame=truck.first_seen_frame,
            last_seen_frame=truck.last_seen_frame,
            first_seen_time_sec=truck.first_seen_time_sec,
            last_seen_time_sec=truck.last_seen_time_sec,
            duration_frames=truck.duration_frames,
            duration_sec=truck.duration_sec,
            confidence_avg=truck.confidence_avg,
            last_bbox_json=json.dumps(truck.last_bbox) if truck.last_bbox is not None else None,
            associated_info_json=(
                json.dumps(truck.associated_info) if truck.associated_info is not None else None
            ),
        )
        db.add(row)
        stored_count += 1

    db.commit()
    return TruckRunImportResponse(run_id=run_id, stored_trucks=stored_count)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        colab_inference_configured=bool(settings.colab_infer_url),
    )


@app.post("/api/detect", response_model=DetectResponse)
async def run_detection(
    file: UploadFile = File(...),
    source_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> DetectResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    safe_source_name = source_name or file.filename or f"upload-{uuid4().hex[:8]}.jpg"
    detections, model_payload, used_mock = inference_service.infer(image_bytes, safe_source_name)

    if not detections:
        raise HTTPException(status_code=422, detail="No detections returned by inference service.")

    run_id = uuid4().hex
    created_items: list[Detection] = []

    for det in detections:
        row = Detection(
            source_name=safe_source_name,
            plate_text_predicted=det.plate_text,
            confidence=det.confidence,
            bbox_json=json.dumps(det.bbox) if det.bbox is not None else None,
            status=DetectionStatus.PENDING.value,
            model_response_json=json.dumps(model_payload),
        )
        db.add(row)
        created_items.append(row)

    db.commit()
    for row in created_items:
        db.refresh(row)

    return DetectResponse(
        run_id=run_id,
        stored_count=len(created_items),
        used_mock_inference=used_mock,
        items=[DetectionRead.model_validate(item) for item in created_items],
    )


@app.get("/api/detections", response_model=DetectionListResponse)
def list_detections(
    status: str | None = Query(default=None, pattern="^(pending|verified|rejected)$"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> DetectionListResponse:
    query = select(Detection)
    count_query = select(func.count(Detection.id))

    if status:
        query = query.where(Detection.status == status)
        count_query = count_query.where(Detection.status == status)

    query = query.order_by(Detection.created_at.desc()).offset(offset).limit(limit)
    items = list(db.scalars(query).all())
    total = db.scalar(count_query) or 0
    return DetectionListResponse(
        items=[DetectionRead.model_validate(item) for item in items],
        total=total,
    )


@app.patch("/api/detections/{detection_id}/verify", response_model=DetectionRead)
def verify_detection(
    detection_id: int,
    payload: VerifyRequest,
    db: Session = Depends(get_db),
) -> DetectionRead:
    record = db.get(Detection, detection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Detection not found.")

    record.plate_text_verified = payload.plate_text_verified.strip().upper()
    record.verified_by = payload.verified_by
    record.status = DetectionStatus.VERIFIED.value
    record.rejection_reason = None
    record.verified_at = datetime.now(UTC)
    db.commit()
    db.refresh(record)
    return DetectionRead.model_validate(record)


@app.patch("/api/detections/{detection_id}/reject", response_model=DetectionRead)
def reject_detection(
    detection_id: int,
    payload: RejectRequest,
    db: Session = Depends(get_db),
) -> DetectionRead:
    record = db.get(Detection, detection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Detection not found.")

    record.status = DetectionStatus.REJECTED.value
    record.verified_by = payload.verified_by
    record.rejection_reason = payload.reason.strip() if payload.reason else "Rejected by operator"
    record.verified_at = datetime.now(UTC)
    db.commit()
    db.refresh(record)
    return DetectionRead.model_validate(record)


@app.post("/api/truck-runs/import", response_model=TruckRunImportResponse)
def import_truck_run(payload: TruckRunImportRequest, db: Session = Depends(get_db)) -> TruckRunImportResponse:
    return _store_truck_run(payload, db)


@app.post("/api/truck-runs/upload-video", response_model=TruckRunImportResponse)
async def upload_video_and_import(
    video: UploadFile = File(...),
    analysis_json: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
) -> TruckRunImportResponse:
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    payload_dict: dict[str, Any]

    if analysis_json is not None:
        raw_json = await analysis_json.read()
        try:
            payload_dict = json.loads(raw_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid analysis JSON file: {exc}") from exc
    elif video_client.configured():
        try:
            payload_dict = video_client.analyze_video_sync(video_bytes, video.filename or "upload.mp4")
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"Video analysis service failed: {exc}") from exc
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "No analysis JSON provided and COLAB_INFER_URL is not configured for "
                "video analysis endpoint /analyze-video."
            ),
        )

    payload_dict.setdefault("session", {})
    payload_dict["session"].setdefault("video_path", video.filename)

    try:
        payload = TruckRunImportRequest.model_validate(payload_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid truck run payload from analysis: {exc}") from exc

    return _store_truck_run(payload, db)


@app.post("/api/truck-runs/video/start", response_model=VideoAnalyzeStartResponse)
async def start_video_job(
    video: UploadFile = File(...),
    analysis_json: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
) -> VideoAnalyzeStartResponse:
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    if analysis_json is not None:
        raw_json = await analysis_json.read()
        try:
            payload_dict = json.loads(raw_json.decode("utf-8"))
            payload = TruckRunImportRequest.model_validate(payload_dict)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid analysis JSON: {exc}") from exc
        saved = _store_truck_run(payload, db)
        return VideoAnalyzeStartResponse(
            mode="direct_json",
            state="completed",
            message="Imported directly from uploaded analysis JSON.",
            run_id=saved.run_id,
            stored_trucks=saved.stored_trucks,
        )

    if not video_client.configured():
        raise HTTPException(
            status_code=400,
            detail="COLAB_INFER_URL is not configured. Upload `analysis_json` or configure Colab API URL.",
        )

    try:
        data = video_client.start_video_job(video_bytes, video.filename or "upload.mp4")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Failed to start Colab video job: {exc}") from exc

    job_id = data.get("job_id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Invalid Colab response (missing job_id): {data}")

    return VideoAnalyzeStartResponse(
        mode="colab_async",
        job_id=job_id,
        state=str(data.get("state", "queued")),
        message=str(data.get("message", "Video job submitted.")),
    )


@app.get("/api/truck-runs/video/{job_id}/status", response_model=VideoAnalyzeStatusResponse)
def get_video_job_status(job_id: str) -> VideoAnalyzeStatusResponse:
    if not video_client.configured():
        raise HTTPException(status_code=400, detail="COLAB_INFER_URL is not configured.")
    try:
        data = video_client.get_video_job_status(job_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Failed to fetch job status: {exc}") from exc

    return VideoAnalyzeStatusResponse(
        job_id=job_id,
        state=str(data.get("state", "unknown")),
        progress=float(data["progress"]) if data.get("progress") is not None else None,
        frame_id=int(data["frame_id"]) if data.get("frame_id") is not None else None,
        total_frames=int(data["total_frames"]) if data.get("total_frames") is not None else None,
        fps=float(data["fps"]) if data.get("fps") is not None else None,
        message=data.get("message"),
    )


@app.post("/api/truck-runs/video/{job_id}/finalize", response_model=VideoAnalyzeFinalizeResponse)
def finalize_video_job(job_id: str, db: Session = Depends(get_db)) -> VideoAnalyzeFinalizeResponse:
    if not video_client.configured():
        raise HTTPException(status_code=400, detail="COLAB_INFER_URL is not configured.")

    try:
        data = video_client.get_video_job_result(job_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Failed to fetch job result: {exc}") from exc

    payload_dict = data.get("result", data)
    try:
        payload = TruckRunImportRequest.model_validate(payload_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid Colab job result payload: {exc}") from exc

    saved = _store_truck_run(payload, db)
    state = str(data.get("state", "completed"))
    return VideoAnalyzeFinalizeResponse(
        job_id=job_id,
        state=state,
        run_id=saved.run_id,
        stored_trucks=saved.stored_trucks,
    )


@app.get("/api/truck-runs/video/{job_id}/frame")
def get_video_job_frame(job_id: str) -> Response:
    if not video_client.configured():
        raise HTTPException(status_code=400, detail="COLAB_INFER_URL is not configured.")
    try:
        frame_bytes, content_type = video_client.get_video_job_frame(job_id)
    except requests.HTTPError as exc:  # pragma: no cover
        code = exc.response.status_code if exc.response is not None else None
        # Colab may return 404/409 while frame is not ready yet; don't surface as hard error.
        if code in (404, 409):
            return Response(status_code=204)
        raise HTTPException(status_code=502, detail=f"Failed to fetch job frame: upstream HTTP {code}") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Failed to fetch job frame: {exc}") from exc
    return Response(content=frame_bytes, media_type=content_type)


@app.get("/api/truck-runs", response_model=TruckRunListResponse)
def list_truck_runs(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> TruckRunListResponse:
    query = select(TruckRun).order_by(TruckRun.created_at.desc()).offset(offset).limit(limit)
    count_query = select(func.count(TruckRun.id))
    items = list(db.scalars(query).all())
    total = db.scalar(count_query) or 0
    return TruckRunListResponse(
        items=[TruckRunRead.model_validate(item) for item in items],
        total=total,
    )


@app.get("/api/trucks", response_model=TruckRecordListResponse)
def list_truck_records(
    run_id: str | None = Query(default=None),
    track_id: int | None = Query(default=None, ge=0),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> TruckRecordListResponse:
    query = select(TruckRecord)
    count_query = select(func.count(TruckRecord.id))

    if run_id:
        query = query.where(TruckRecord.run_id == run_id)
        count_query = count_query.where(TruckRecord.run_id == run_id)
    if track_id is not None:
        query = query.where(TruckRecord.track_id == track_id)
        count_query = count_query.where(TruckRecord.track_id == track_id)

    query = query.order_by(TruckRecord.track_id.asc()).offset(offset).limit(limit)
    items = list(db.scalars(query).all())
    total = db.scalar(count_query) or 0
    return TruckRecordListResponse(
        items=[_truck_record_to_read(item) for item in items],
        total=total,
    )
