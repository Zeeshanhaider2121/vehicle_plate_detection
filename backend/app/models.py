from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class DetectionStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    plate_text_predicted: Mapped[str] = mapped_column(String(64), nullable=False)
    plate_text_verified: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=DetectionStatus.PENDING.value, index=True)
    model_response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    verified_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class TruckRun(Base):
    __tablename__ = "truck_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    frames_processed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_trucks_tracked: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trucks_with_container: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trucks_without_container: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class TruckRecord(Base):
    __tablename__ = "truck_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    truck_type: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_seen_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_seen_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_seen_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    associated_info_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
