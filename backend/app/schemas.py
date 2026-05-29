from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DetectionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_name: str
    plate_text_predicted: str
    plate_text_verified: str | None
    confidence: float | None
    bbox_json: str | None
    status: str
    rejection_reason: str | None
    verified_by: str | None
    verified_at: datetime | None
    created_at: datetime
    updated_at: datetime


class DetectionListResponse(BaseModel):
    items: list[DetectionRead]
    total: int


class DetectResponse(BaseModel):
    run_id: str
    stored_count: int
    used_mock_inference: bool
    items: list[DetectionRead]


class VerifyRequest(BaseModel):
    plate_text_verified: str = Field(min_length=1, max_length=64)
    verified_by: str = Field(default="operator-1", min_length=1, max_length=128)


class RejectRequest(BaseModel):
    verified_by: str = Field(default="operator-1", min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=255)


class HealthResponse(BaseModel):
    status: str
    environment: str
    colab_inference_configured: bool


class InferenceRecord(BaseModel):
    plate_text: str
    confidence: float | None = None
    bbox: dict[str, Any] | list[Any] | None = None


class SessionInfoIn(BaseModel):
    video_path: str | None = None
    total_frames: int | None = None
    video_fps: float | None = None
    resolution: str | None = None
    started_at: str | None = None
    device: str | None = None
    model: str | None = None
    finished_at: str | None = None
    frames_processed: int | None = None


class SummaryInfoIn(BaseModel):
    total_trucks_tracked: int | None = None
    trucks_with_container: int | None = None
    trucks_without_container: int | None = None


class TruckInfoIn(BaseModel):
    track_id: int | None = None
    type: str
    first_seen_frame: int | None = None
    last_seen_frame: int | None = None
    first_seen_time_sec: float | None = None
    last_seen_time_sec: float | None = None
    duration_frames: int | None = None
    duration_sec: float | None = None
    confidence_avg: float | None = None
    last_bbox: list[float] | None = None
    associated_info: dict[str, Any] | None = None


class TruckRunImportRequest(BaseModel):
    session: SessionInfoIn
    summary: SummaryInfoIn
    trucks: dict[str, TruckInfoIn]


class TruckRunImportResponse(BaseModel):
    run_id: str
    stored_trucks: int


class TruckRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    video_path: str | None
    total_frames: int | None
    video_fps: float | None
    resolution: str | None
    started_at: str | None
    finished_at: str | None
    device: str | None
    model: str | None
    frames_processed: int | None
    total_trucks_tracked: int | None
    trucks_with_container: int | None
    trucks_without_container: int | None
    created_at: datetime


class TruckRunListResponse(BaseModel):
    items: list[TruckRunRead]
    total: int


class TruckRecordRead(BaseModel):
    id: int
    run_id: str
    track_id: int
    truck_type: str
    first_seen_frame: int | None
    last_seen_frame: int | None
    first_seen_time_sec: float | None
    last_seen_time_sec: float | None
    duration_frames: int | None
    duration_sec: float | None
    confidence_avg: float | None
    last_bbox: list[float] | None
    associated_info: dict[str, Any] | None
    created_at: datetime


class TruckRecordListResponse(BaseModel):
    items: list[TruckRecordRead]
    total: int


class VideoAnalyzeStartResponse(BaseModel):
    mode: str
    job_id: str | None = None
    state: str
    message: str
    run_id: str | None = None
    stored_trucks: int | None = None


class VideoAnalyzeStatusResponse(BaseModel):
    job_id: str
    state: str
    progress: float | None = None
    frame_id: int | None = None
    total_frames: int | None = None
    fps: float | None = None
    message: str | None = None
    json_snapshot: str | None = None
    ocr_log: list[str] = Field(default_factory=list)


class VideoAnalyzeFinalizeResponse(BaseModel):
    job_id: str
    state: str
    run_id: str
    stored_trucks: int
