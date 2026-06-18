import json
import re
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
    MultiCameraAnalyzeFinalizeResponse,
    MultiCameraAnalyzeStartResponse,
    MultiCameraAnalyzeStatusResponse,
    MultiCameraChildStatus,
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

# Cache the last known good snapshot per job so the frontend
# still gets data even after the Colab stub crashes.
_last_good_snapshot: dict[str, str] = {}
_multi_camera_jobs: dict[str, dict[str, Any]] = {}
_video_demo_constraints: dict[str, dict[str, Any]] = {}

CAMERA_LABELS = tuple(
    item.strip()
    for item in settings.multi_camera_order.split(",")
    if item.strip()
)
OCR_FIELD_KEYS = (
    "container_company_logo",
    "container_number",
    "container_side_no",
    "driver",
    "license_plate",
    "other_container_info",
    "truck_company",
    "truck_number",
)


def _parse_camera_time_offsets(raw: str) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for part in raw.split(","):
        if not part.strip() or "=" not in part:
            continue
        camera, value = part.split("=", 1)
        try:
            offsets[camera.strip()] = float(value.strip())
        except ValueError:
            continue
    return offsets


def _parse_camera_roles(raw: str) -> dict[str, set[str]]:
    """Parse 'front:f1,f2;left:f3' into {camera: {fields}}."""
    roles: dict[str, set[str]] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        camera, fields_str = part.split(":", 1)
        roles[camera.strip()] = {f.strip() for f in fields_str.split(",") if f.strip()}
    return roles


MULTI_CAMERA_ASSUME_SINGLE_ENTITY = settings.multi_camera_assume_single_entity
MULTI_CAMERA_ORDER_FALLBACK = settings.multi_camera_order_fallback
MULTI_CAMERA_REVIEW_THRESHOLD = settings.multi_camera_review_threshold
TRACK_FRAGMENT_MERGE_GAP_SECONDS = settings.track_fragment_merge_gap_seconds
TRACK_FRAGMENT_MERGE_AGGRESSIVE = settings.track_fragment_merge_aggressive
MULTI_CAMERA_GATE_MODE = settings.multi_camera_gate_mode
MULTI_CAMERA_CAMERA_ROLES: dict[str, set[str]] = _parse_camera_roles(settings.multi_camera_camera_roles)

CAMERA_TIME_OFFSETS_SECONDS = _parse_camera_time_offsets(
    settings.multi_camera_time_offsets_seconds
)

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


async def _read_optional_json_upload(upload: UploadFile | None, label: str) -> dict[str, Any] | None:
    if upload is None:
        return None
    raw = await upload.read()
    if not raw:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label} JSON file: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=f"{label} JSON must be an object.")
    return parsed


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
    merged_trucks = _merge_track_fragments(payload.trucks)
    fps = payload.session.video_fps
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
        total_trucks_tracked=len(merged_trucks),
        trucks_with_container=sum(1 for truck in merged_trucks if truck["type"] == "truck_with_container"),
        trucks_without_container=sum(1 for truck in merged_trucks if truck["type"] == "truck_without_container"),
    )
    db.add(run_row)

    stored_count = 0
    for index, truck in enumerate(merged_trucks, start=1):
        first_seen_time_sec = truck.get("first_seen_time_sec")
        last_seen_time_sec = truck.get("last_seen_time_sec")
        duration_sec = truck.get("duration_sec")
        if fps:
            if first_seen_time_sec is None and truck.get("first_seen_frame") is not None:
                first_seen_time_sec = round(float(truck["first_seen_frame"]) / float(fps), 3)
            if last_seen_time_sec is None and truck.get("last_seen_frame") is not None:
                last_seen_time_sec = round(float(truck["last_seen_frame"]) / float(fps), 3)
            if duration_sec is None and first_seen_time_sec is not None and last_seen_time_sec is not None:
                duration_sec = round(float(last_seen_time_sec) - float(first_seen_time_sec), 3)
        row = TruckRecord(
            run_id=run_id,
            track_id=index,
            truck_type=truck["type"],
            first_seen_frame=truck.get("first_seen_frame"),
            last_seen_frame=truck.get("last_seen_frame"),
            first_seen_time_sec=first_seen_time_sec,
            last_seen_time_sec=last_seen_time_sec,
            duration_frames=truck.get("duration_frames"),
            duration_sec=duration_sec,
            confidence_avg=truck.get("confidence_avg"),
            last_bbox_json=json.dumps(truck.get("last_bbox")) if truck.get("last_bbox") is not None else None,
            associated_info_json=(
                json.dumps(truck.get("associated_info")) if truck.get("associated_info") is not None else None
            ),
        )
        db.add(row)
        stored_count += 1

    db.commit()
    return TruckRunImportResponse(run_id=run_id, stored_trucks=stored_count)


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("text") or "").strip()
    return str(value).strip()


def _field_conf(value: Any) -> float:
    if isinstance(value, dict):
        try:
            return float(value.get("confidence") or 0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _text_quality(text: str) -> int:
    lowered = text.lower()
    bad_terms = (
        "blurr",
        "indistinct",
        "unrecognizable",
        "unreadable",
        "not visible",
        "no visible",
        "unclear",
        "unable",
        "no text",
    )
    if not text:
        return 0
    if any(term in lowered for term in bad_terms):
        return 1
    return 2


ISO6346_LETTER_VALUES = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18,
    "I": 19, "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27,
    "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36,
    "Y": 37, "Z": 38,
}


def _clean_identifier(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _validate_iso6346(value: str) -> dict[str, Any]:
    normalized = _clean_identifier(value)
    result = {
        "field": "container_number",
        "value": normalized,
        "valid": False,
        "reason": "Expected 4 letters followed by 7 digits",
    }
    if not re.fullmatch(r"[A-Z]{4}\d{7}", normalized):
        return result

    total = 0
    for index, char in enumerate(normalized[:10]):
        char_value = ISO6346_LETTER_VALUES.get(char) if char.isalpha() else int(char)
        total += char_value * (2 ** index)
    check_digit = total % 11
    if check_digit == 10:
        check_digit = 0
    expected = int(normalized[-1])
    result["expected_check_digit"] = check_digit
    result["actual_check_digit"] = expected
    result["valid"] = check_digit == expected
    result["reason"] = "Valid ISO 6346 check digit" if result["valid"] else "Invalid ISO 6346 check digit"
    return result


def _build_field_validation(info: dict[str, Any]) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    container_text = _field_text(info.get("container_number"))
    if container_text:
        validation["container_number"] = _validate_iso6346(container_text)

    for field in ("license_plate", "truck_number", "container_side_no"):
        text = _clean_identifier(_field_text(info.get(field)))
        if text:
            validation[field] = {
                "field": field,
                "value": text,
                "valid": len(text) >= 3,
                "reason": "Enough alphanumeric characters" if len(text) >= 3 else "Too short",
            }
    return validation


def _better_field(existing: Any, incoming: Any) -> Any:
    if incoming in (None, ""):
        return existing
    if existing in (None, ""):
        return incoming

    incoming_text = _field_text(incoming)
    existing_text = _field_text(existing)
    incoming_conf = _field_conf(incoming)
    existing_conf = _field_conf(existing)

    if incoming_conf and existing_conf and incoming_conf != existing_conf:
        return incoming if incoming_conf > existing_conf else existing
    if _text_quality(incoming_text) != _text_quality(existing_text):
        return incoming if _text_quality(incoming_text) > _text_quality(existing_text) else existing
    if incoming_text and existing_text and len(incoming_text) < len(existing_text):
        return incoming
    return existing


def _normalize_truck_payload(track_key: str, truck: dict[str, Any]) -> dict[str, Any]:
    try:
        fallback_track_id = int(track_key)
    except (TypeError, ValueError):
        fallback_track_id = -1

    info = truck.get("associated_info") or truck.get("info") or {}
    return {
        "track_id": truck.get("track_id") or truck.get("tid") or fallback_track_id,
        "type": truck.get("type") or truck.get("truck_type") or "truck_with_container",
        "first_seen_frame": truck.get("first_seen_frame") or truck.get("first"),
        "last_seen_frame": truck.get("last_seen_frame") or truck.get("last"),
        "first_seen_time_sec": truck.get("first_seen_time_sec"),
        "last_seen_time_sec": truck.get("last_seen_time_sec"),
        "duration_frames": truck.get("duration_frames"),
        "duration_sec": truck.get("duration_sec"),
        "confidence_avg": truck.get("confidence_avg") or truck.get("conf"),
        "last_bbox": truck.get("last_bbox"),
        "associated_info": dict(info) if isinstance(info, dict) else {},
    }


def _apply_camera_time_offset(truck: dict[str, Any], camera: str) -> dict[str, Any]:
    offset = CAMERA_TIME_OFFSETS_SECONDS.get(camera, 0.0)
    if not offset:
        truck["time_offset_sec"] = 0.0
        return truck
    adjusted = dict(truck)
    adjusted["time_offset_sec"] = offset
    for key in ("first_seen_time_sec", "last_seen_time_sec", "duration_sec"):
        if adjusted.get(key) is None:
            continue
        if key == "duration_sec":
            continue
        adjusted[key] = round(float(adjusted[key]) + offset, 3)
    return adjusted


def _identity_key(truck: dict[str, Any]) -> str | None:
    info = truck.get("associated_info") or {}
    for field in ("license_plate", "truck_number", "container_number", "container_side_no"):
        value = re.sub(r"[^A-Z0-9]", "", _field_text(info.get(field)).upper())
        if value and len(value) >= 3 and _text_quality(value) > 1:
            return f"{field}:{value}"
    return None


def _identity_keys(truck: dict[str, Any]) -> list[str]:
    info = truck.get("associated_info") or {}
    keys: list[str] = []
    for field in ("license_plate", "truck_number", "container_number", "container_side_no"):
        value = re.sub(r"[^A-Z0-9]", "", _field_text(info.get(field)).upper())
        if value and len(value) >= 3 and _text_quality(value) > 1:
            keys.append(f"{field}:{value}")
    return keys


def _truck_sort_key(observation: dict[str, Any]) -> tuple[float, int]:
    truck = observation["truck"]
    bbox = truck.get("last_bbox") or []
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            center_x = (float(bbox[0]) + float(bbox[2])) / 2
            return (center_x, int(truck.get("track_id") or 0))
        except (TypeError, ValueError):
            pass
    return (float(truck.get("track_id") or 0), int(truck.get("track_id") or 0))


def _camera_sort_key(camera: str) -> tuple[int, str]:
    try:
        return (CAMERA_LABELS.index(camera), camera)
    except ValueError:
        return (len(CAMERA_LABELS), camera)


def _group_identity_summary(observations: list[dict[str, Any]]) -> tuple[list[str], str, float]:
    key_sets = [set(_identity_keys(obs["truck"])) for obs in observations]
    non_empty = [keys for keys in key_sets if keys]
    if not non_empty:
        return [], "order_fallback", 0.55 if len(observations) > 1 else 0.4

    common = set.intersection(*non_empty) if non_empty else set()
    if common and len(non_empty) == len(observations):
        return sorted(common), "exact_identity", 0.96
    if common:
        return sorted(common), "partial_identity_order", 0.82

    all_keys = sorted({key for keys in non_empty for key in keys})
    if len(non_empty) == len(observations):
        return all_keys, "conflicting_identity_order", 0.35
    return all_keys, "partial_identity_order", 0.65


def _merge_truck_group(
    entity_id: int,
    observations: list[dict[str, Any]],
    match_method: str | None = None,
    match_confidence: float | None = None,
    camera_roles: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    first = observations[0]
    merged_info: dict[str, Any] = {key: None for key in OCR_FIELD_KEYS}
    camera_observations: list[dict[str, Any]] = []
    truck_types: list[str] = []
    confidences: list[float] = []
    first_frames: list[int] = []
    last_frames: list[int] = []
    first_times: list[float] = []
    last_times: list[float] = []
    identity_keys, inferred_method, inferred_confidence = _group_identity_summary(observations)
    match_method = match_method or inferred_method
    match_confidence = inferred_confidence if match_confidence is None else match_confidence

    for obs in observations:
        truck = obs["truck"]
        camera = obs["camera"]
        info = truck.get("associated_info") or {}
        truck_types.append(str(truck.get("type") or ""))
        if truck.get("confidence_avg") is not None:
            confidences.append(float(truck["confidence_avg"]))
        if truck.get("first_seen_frame") is not None:
            first_frames.append(int(truck["first_seen_frame"]))
        if truck.get("last_seen_frame") is not None:
            last_frames.append(int(truck["last_seen_frame"]))
        if truck.get("first_seen_time_sec") is not None:
            first_times.append(float(truck["first_seen_time_sec"]))
        if truck.get("last_seen_time_sec") is not None:
            last_times.append(float(truck["last_seen_time_sec"]))

        camera_observations.append(
            {
                "entity_track_id": entity_id,
                "camera": camera,
                "source_track_id": truck.get("track_id"),
                "time_offset_sec": truck.get("time_offset_sec", 0.0),
                "source_identity_keys": _identity_keys(truck),
                "truck_type": truck.get("type"),
                "confidence_avg": truck.get("confidence_avg"),
                "first_seen_frame": truck.get("first_seen_frame"),
                "last_seen_frame": truck.get("last_seen_frame"),
                "last_bbox": truck.get("last_bbox"),
            }
        )

    # Camera-role-aware field merging.
    # For each field, process authoritative-camera observations first so they win
    # tiebreaks against equal-confidence reads from off-role cameras.
    for field in OCR_FIELD_KEYS:
        if camera_roles:
            auth = [o for o in observations if field in (camera_roles.get(o["camera"]) or set())]
            other = [o for o in observations if o not in auth]
            ordered = auth + other
        else:
            ordered = observations
        for obs in ordered:
            info = obs["truck"].get("associated_info") or {}
            merged_info[field] = _better_field(merged_info.get(field), info.get(field))

    field_validation = _build_field_validation(merged_info)
    invalid_fields = [
        field
        for field, result in field_validation.items()
        if result.get("valid") is False
    ]
    needs_review = match_confidence < MULTI_CAMERA_REVIEW_THRESHOLD or bool(invalid_fields)
    review_reasons: list[str] = []
    if match_confidence < MULTI_CAMERA_REVIEW_THRESHOLD:
        review_reasons.append(f"Match confidence below {MULTI_CAMERA_REVIEW_THRESHOLD:.2f}")
    if invalid_fields:
        review_reasons.append(f"Invalid field validation: {', '.join(invalid_fields)}")

    merged_info["_camera_observations"] = camera_observations
    merged_info["_field_validation"] = field_validation
    merged_info["_fusion"] = {
        "entity_track_id": entity_id,
        "match_method": match_method,
        "match_confidence": round(match_confidence, 4),
        "needs_review": needs_review,
        "review_reason": "; ".join(review_reasons) if review_reasons else None,
        "identity_keys": identity_keys,
        "camera_roles_applied": bool(camera_roles),
        "camera_count": len({obs["camera"] for obs in observations}),
        "camera_time_offsets_seconds": {
            obs["camera"]: obs["truck"].get("time_offset_sec", 0.0)
            for obs in observations
        },
        "source_track_ids": {
            obs["camera"]: obs["truck"].get("track_id")
            for obs in observations
        },
    }
    preferred_type = (
        "truck_with_container"
        if "truck_with_container" in truck_types
        else (truck_types[0] or "truck_without_container")
    )

    return {
        "track_id": entity_id,
        "type": preferred_type,
        "first_seen_frame": min(first_frames) if first_frames else None,
        "last_seen_frame": max(last_frames) if last_frames else None,
        "first_seen_time_sec": min(first_times) if first_times else None,
        "last_seen_time_sec": max(last_times) if last_times else None,
        "duration_frames": (
            max(last_frames) - min(first_frames) + 1 if first_frames and last_frames else None
        ),
        "duration_sec": (
            round(max(last_times) - min(first_times), 3) if first_times and last_times else None
        ),
        "confidence_avg": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "last_bbox": first.get("truck", {}).get("last_bbox"),
        "associated_info": merged_info,
    }


def _truck_info_to_dict(key: str, truck: Any) -> dict[str, Any]:
    try:
        fallback_track_id = int(key)
    except (TypeError, ValueError):
        fallback_track_id = -1
    return {
        "track_id": truck.track_id if truck.track_id is not None else fallback_track_id,
        "type": truck.type,
        "first_seen_frame": truck.first_seen_frame,
        "last_seen_frame": truck.last_seen_frame,
        "first_seen_time_sec": truck.first_seen_time_sec,
        "last_seen_time_sec": truck.last_seen_time_sec,
        "duration_frames": truck.duration_frames,
        "duration_sec": truck.duration_sec,
        "confidence_avg": truck.confidence_avg,
        "last_bbox": truck.last_bbox,
        "associated_info": dict(truck.associated_info or {}),
    }


def _identifier_values(truck: dict[str, Any]) -> list[str]:
    info = truck.get("associated_info") or {}
    values: list[str] = []
    for field in ("license_plate", "truck_number", "container_number", "container_side_no"):
        value = _clean_identifier(_field_text(info.get(field)))
        if len(value) >= 3:
            values.append(value)
    return values


def _strong_identifier_values(truck: dict[str, Any]) -> list[str]:
    info = truck.get("associated_info") or {}
    values: list[str] = []
    for field in ("license_plate", "truck_number", "container_number"):
        value = _clean_identifier(_field_text(info.get(field)))
        if len(value) >= 4:
            values.append(value)
    return values


def _identifiers_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_values = _identifier_values(a)
    b_values = _identifier_values(b)
    for left in a_values:
        for right in b_values:
            if left == right:
                return True
            if min(len(left), len(right)) >= 6 and (left.startswith(right) or right.startswith(left)):
                return True
    return False


def _has_identifier_conflict(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_values = _strong_identifier_values(a)
    b_values = _strong_identifier_values(b)
    if not a_values or not b_values:
        return False
    for left in a_values:
        for right in b_values:
            if left == right:
                return False
            if min(len(left), len(right)) >= 6 and (left.startswith(right) or right.startswith(left)):
                return False
    return True


def _time_gap_seconds(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    a_start = a.get("first_seen_time_sec")
    a_end = a.get("last_seen_time_sec")
    b_start = b.get("first_seen_time_sec")
    b_end = b.get("last_seen_time_sec")
    if None in (a_start, a_end, b_start, b_end):
        return None
    if float(a_end) < float(b_start):
        return float(b_start) - float(a_end)
    if float(b_end) < float(a_start):
        return float(a_start) - float(b_end)
    return 0.0


def _fragments_should_merge(group: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    if any(item.get("type") != candidate.get("type") for item in group):
        return False
    if any(_identifiers_compatible(item, candidate) for item in group):
        return True
    if any(_has_identifier_conflict(item, candidate) for item in group):
        return False
    gaps = [_time_gap_seconds(item, candidate) for item in group]
    known_gaps = [gap for gap in gaps if gap is not None]
    if not known_gaps:
        return False
    if min(known_gaps) <= TRACK_FRAGMENT_MERGE_GAP_SECONDS:
        return True
    if TRACK_FRAGMENT_MERGE_AGGRESSIVE and not any(_strong_identifier_values(item) for item in group) and not _strong_identifier_values(candidate):
        return True
    return False


def _merge_track_fragments(trucks: dict[str, Any]) -> list[dict[str, Any]]:
    raw = [
        (_normalize_truck_payload(str(key), truck) if isinstance(truck, dict) else _truck_info_to_dict(str(key), truck))
        for key, truck in trucks.items()
    ]
    if any((truck.get("associated_info") or {}).get("_fusion") for truck in raw):
        return raw

    raw.sort(
        key=lambda truck: (
            truck.get("first_seen_time_sec") if truck.get("first_seen_time_sec") is not None else 10**9,
            truck.get("track_id") or 0,
        )
    )
    groups: list[list[dict[str, Any]]] = []
    for truck in raw:
        target_group = None
        for group in groups:
            if _fragments_should_merge(group, truck):
                target_group = group
                break
        if target_group is None:
            groups.append([truck])
        else:
            target_group.append(truck)

    merged: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        observations = [{"camera": "track_fragment", "truck": truck} for truck in group]
        merged_truck = _merge_truck_group(index, observations, "track_fragment_merge", 0.7 if len(group) > 1 else 0.45)
        merged_truck["associated_info"]["_source_track_fragments"] = [
            {
                "source_track_id": truck.get("track_id"),
                "first_seen_time_sec": truck.get("first_seen_time_sec"),
                "last_seen_time_sec": truck.get("last_seen_time_sec"),
            }
            for truck in group
        ]
        merged.append(merged_truck)
    return merged


def _apply_fragment_merge_to_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply track-fragment merging to a raw Colab snapshot so the live view shows merged trucks."""
    trucks_raw = payload.get("trucks") or {}
    if not trucks_raw:
        return payload
    merged_list = _merge_track_fragments(trucks_raw)
    trucks_out = {str(t["track_id"]): t for t in merged_list}
    result = dict(payload)
    result["trucks"] = trucks_out
    result["summary"] = {
        "total_trucks_tracked": len(trucks_out),
        "trucks_with_container": sum(1 for t in trucks_out.values() if t["type"] == "truck_with_container"),
        "trucks_without_container": sum(1 for t in trucks_out.values() if t["type"] == "truck_without_container"),
    }
    return result


def _payload_observations_for_demo(payload: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for track_key, truck in (payload.get("trucks") or {}).items():
        if not isinstance(truck, dict):
            continue
        normalized = _normalize_truck_payload(str(track_key), truck)
        info = normalized.get("associated_info") or {}
        camera_observations = info.get("_camera_observations")
        if isinstance(camera_observations, list) and camera_observations:
            for camera_obs in camera_observations:
                if not isinstance(camera_obs, dict):
                    continue
                source_truck = dict(normalized)
                source_truck["track_id"] = camera_obs.get("source_track_id") or normalized.get("track_id")
                for key in ("first_seen_frame", "last_seen_frame", "last_bbox", "confidence_avg"):
                    if camera_obs.get(key) is not None:
                        source_truck[key] = camera_obs[key]
                observations.append(
                    {
                        "camera": str(camera_obs.get("camera") or "demo"),
                        "truck": source_truck,
                    }
                )
            continue
        observations.append({"camera": str(truck.get("camera") or "demo"), "truck": normalized})
    return observations


def _observation_time_range(observation: dict[str, Any]) -> tuple[float | None, float | None]:
    truck = observation["truck"]
    start = truck.get("first_seen_time_sec")
    end = truck.get("last_seen_time_sec")
    try:
        start_float = float(start) if start is not None else None
        end_float = float(end) if end is not None else start_float
    except (TypeError, ValueError):
        return None, None
    return start_float, end_float


def _window_matches_observation(window: dict[str, Any], observation: dict[str, Any]) -> bool:
    cameras = window.get("cameras")
    if cameras:
        if isinstance(cameras, str):
            allowed_cameras = {item.strip() for item in cameras.split(",") if item.strip()}
        else:
            allowed_cameras = {str(item) for item in cameras}
        if observation.get("camera") not in allowed_cameras:
            return False
    start, end = _observation_time_range(observation)
    if start is None or end is None:
        return True
    window_start = window.get("start_time_sec", window.get("start"))
    window_end = window.get("end_time_sec", window.get("end"))
    try:
        start_limit = float(window_start) if window_start is not None else None
        end_limit = float(window_end) if window_end is not None else None
    except (TypeError, ValueError):
        return True
    if start_limit is not None and end < start_limit:
        return False
    if end_limit is not None and start > end_limit:
        return False
    return True


def _split_observations_by_expected_count(
    observations: list[dict[str, Any]],
    expected_count: int,
) -> list[list[dict[str, Any]]]:
    if expected_count <= 1:
        return [observations]
    sorted_observations = sorted(
        observations,
        key=lambda obs: (
            _observation_time_range(obs)[0] if _observation_time_range(obs)[0] is not None else 10**9,
            _truck_sort_key(obs),
        ),
    )
    groups = [[] for _ in range(expected_count)]
    for index, observation in enumerate(sorted_observations):
        bucket = min(int(index * expected_count / max(len(sorted_observations), 1)), expected_count - 1)
        groups[bucket].append(observation)
    return [group for group in groups if group]


def _apply_demo_constraints_to_payload(
    payload: dict[str, Any],
    constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    if not constraints:
        return payload

    observations = _payload_observations_for_demo(payload)
    if not observations:
        return payload

    groups: list[list[dict[str, Any]]] = []
    used_ids: set[int] = set()
    windows = constraints.get("vehicle_windows") or constraints.get("vehicles")
    if isinstance(windows, list) and windows:
        for window in windows:
            if not isinstance(window, dict):
                continue
            group: list[dict[str, Any]] = []
            group_indexes: list[int] = []
            for index, obs in enumerate(observations):
                if index not in used_ids and _window_matches_observation(window, obs):
                    group.append(obs)
                    group_indexes.append(index)
            if group:
                used_ids.update(group_indexes)
                groups.append(group)
        leftovers = [obs for index, obs in enumerate(observations) if index not in used_ids]
        if leftovers:
            if groups:
                groups[-1].extend(leftovers)
            else:
                groups.append(leftovers)
    else:
        expected_count = (
            constraints.get("expected_vehicle_count")
            or constraints.get("max_vehicle_rows")
            or constraints.get("vehicle_count")
        )
        try:
            expected_count_int = max(1, int(expected_count))
        except (TypeError, ValueError):
            expected_count_int = 1
        groups = _split_observations_by_expected_count(observations, expected_count_int)

    trucks_out: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(groups, start=1):
        merged = _merge_truck_group(index, group, "demo_constraint", 0.99)
        fusion = merged["associated_info"].setdefault("_fusion", {})
        fusion["demo_constraints_applied"] = True
        fusion["demo_constraints"] = {
            "expected_vehicle_count": constraints.get("expected_vehicle_count"),
            "max_vehicle_rows": constraints.get("max_vehicle_rows"),
            "vehicle_windows": len(windows) if isinstance(windows, list) else 0,
        }
        trucks_out[str(index)] = merged

    constrained = dict(payload)
    constrained["trucks"] = trucks_out
    constrained["summary"] = {
        "total_trucks_tracked": len(trucks_out),
        "trucks_with_container": sum(1 for truck in trucks_out.values() if truck["type"] == "truck_with_container"),
        "trucks_without_container": sum(1 for truck in trucks_out.values() if truck["type"] == "truck_without_container"),
    }
    constrained.setdefault("session", payload.get("session") or {})
    constrained["session"]["demo_constraints_applied"] = True
    return constrained


def _merge_multi_camera_payloads(camera_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    total_frames = 0
    frames_processed = 0
    fps_values: list[float] = []
    devices: set[str] = set()
    models: set[str] = set()

    for camera, payload in camera_payloads.items():
        session = payload.get("session") or {}
        if session.get("total_frames"):
            total_frames += int(session["total_frames"])
        if session.get("frames_processed"):
            frames_processed += int(session["frames_processed"])
        if session.get("video_fps") is not None:
            fps_values.append(float(session["video_fps"]))
        if session.get("device"):
            devices.add(str(session["device"]))
        if session.get("model"):
            models.add(str(session["model"]))

        trucks = payload.get("trucks") or {}
        for track_key, truck in trucks.items():
            if isinstance(truck, dict):
                observations.append(
                    {
                        "camera": camera,
                        "truck": _apply_camera_time_offset(
                            _normalize_truck_payload(str(track_key), truck),
                            camera,
                        ),
                    }
                )

    if not observations:
        return {
            "session": {
                "video_path": ", ".join(camera_payloads.keys()),
                "total_frames": total_frames or None,
                "video_fps": round(sum(fps_values) / len(fps_values), 3) if fps_values else None,
                "resolution": "multi-camera",
                "device": ", ".join(sorted(devices)) or None,
                "model": ", ".join(sorted(models)) or None,
                "frames_processed": frames_processed or None,
                "camera_time_offsets_seconds": CAMERA_TIME_OFFSETS_SECONDS,
            },
            "summary": {
                "total_trucks_tracked": 0,
                "trucks_with_container": 0,
                "trucks_without_container": 0,
            },
            "trucks": {},
        }

    observations_by_camera: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        observations_by_camera.setdefault(obs["camera"], []).append(obs)

    group_specs: list[tuple[list[dict[str, Any]], str | None, float | None]]
    if MULTI_CAMERA_GATE_MODE:
        # Gate setup: one vehicle at a time, all cameras see the same truck.
        # Merge every observation regardless of per-camera count.
        group_specs = [(observations, "gate_mode", 0.95)]
    elif (
        MULTI_CAMERA_ASSUME_SINGLE_ENTITY
        and observations_by_camera
        and all(len(items) <= 1 for items in observations_by_camera.values())
    ):
        # Common gate setup: each camera is looking at the same truck from a different angle.
        group_specs = [(observations, "assume_single_entity", 0.5)]
    else:
        camera_counts = {len(items) for items in observations_by_camera.values()}
        if (
            MULTI_CAMERA_ORDER_FALLBACK
            and len(observations_by_camera) > 1
            and len(camera_counts) == 1
        ):
            ordered_by_camera = {
                camera: sorted(items, key=_truck_sort_key)
                for camera, items in sorted(observations_by_camera.items(), key=lambda item: _camera_sort_key(item[0]))
            }
            trucks_per_camera = next(iter(camera_counts))
            group_specs = []
            for index in range(trucks_per_camera):
                group = [items[index] for items in ordered_by_camera.values()]
                _identity_keys_for_group, method, confidence = _group_identity_summary(group)
                if method == "order_fallback":
                    method = "same_count_order"
                elif method == "partial_identity_order":
                    method = "same_count_partial_identity"
                elif method == "conflicting_identity_order":
                    method = "same_count_conflicting_identity"
                group_specs.append((group, method, confidence))
        else:
            identity_groups: dict[str, list[dict[str, Any]]] = {}
            no_identity = []
            for obs in observations:
                key = _identity_key(obs["truck"])
                if key:
                    identity_groups.setdefault(key, []).append(obs)
                else:
                    no_identity.append(obs)

            group_specs = [
                (group, "exact_identity", 0.96 if len(group) > 1 else 0.45)
                for group in identity_groups.values()
            ]
            group_specs.extend(([obs], "single_camera_unmatched", 0.25) for obs in no_identity)

    trucks_out: dict[str, dict[str, Any]] = {}
    _roles = MULTI_CAMERA_CAMERA_ROLES if MULTI_CAMERA_CAMERA_ROLES else None
    for index, (group, match_method, match_confidence) in enumerate(group_specs, start=1):
        trucks_out[str(index)] = _merge_truck_group(index, group, match_method, match_confidence, _roles)

    with_container = sum(1 for truck in trucks_out.values() if truck["type"] == "truck_with_container")
    without_container = sum(1 for truck in trucks_out.values() if truck["type"] == "truck_without_container")
    return {
        "session": {
            "video_path": ", ".join(camera_payloads.keys()),
            "total_frames": total_frames or None,
            "video_fps": round(sum(fps_values) / len(fps_values), 3) if fps_values else None,
            "resolution": "multi-camera",
            "device": ", ".join(sorted(devices)) or None,
            "model": ", ".join(sorted(models)) or None,
            "frames_processed": frames_processed or None,
            "camera_time_offsets_seconds": CAMERA_TIME_OFFSETS_SECONDS,
        },
        "summary": {
            "total_trucks_tracked": len(trucks_out),
            "trucks_with_container": with_container,
            "trucks_without_container": without_container,
        },
        "trucks": trucks_out,
    }


def _multi_camera_state(child_states: list[str]) -> str:
    if any(state == "failed" for state in child_states):
        return "failed"
    if child_states and all(state == "completed" for state in child_states):
        return "completed"
    if any(state in {"running", "processing"} for state in child_states):
        return "processing"
    return "queued"


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
    demo_constraints: UploadFile | None = File(default=None),
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
    payload_dict = _apply_demo_constraints_to_payload(
        payload_dict,
        await _read_optional_json_upload(demo_constraints, "demo constraints"),
    )

    try:
        payload = TruckRunImportRequest.model_validate(payload_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid truck run payload from analysis: {exc}") from exc

    return _store_truck_run(payload, db)


@app.post("/api/truck-runs/video/start", response_model=VideoAnalyzeStartResponse)
async def start_video_job(
    video: UploadFile = File(...),
    analysis_json: UploadFile | None = File(default=None),
    demo_constraints: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
) -> VideoAnalyzeStartResponse:
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    constraints = await _read_optional_json_upload(demo_constraints, "demo constraints")

    if analysis_json is not None:
        raw_json = await analysis_json.read()
        try:
            payload_dict = json.loads(raw_json.decode("utf-8"))
            payload_dict = _apply_demo_constraints_to_payload(payload_dict, constraints)
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
    if constraints:
        _video_demo_constraints[job_id] = constraints

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

    # Try fetching from the Colab stub; if it fails, return the last known good snapshot.
    try:
        data = video_client.get_video_job_status(job_id)
    except Exception:  # pragma: no cover
        cached = _last_good_snapshot.get(job_id)
        if cached:
            return VideoAnalyzeStatusResponse(
                job_id=job_id,
                state="cached",
                progress=None,
                json_snapshot=cached,
                ocr_log=[],
            )
        raise HTTPException(status_code=502, detail="Colab stub is unavailable and no cached snapshot exists.")

    snap = data.get("json_snapshot")
    ocr = data.get("ocr_log") or []
    if snap and len(snap) > 50:
        _last_good_snapshot[job_id] = snap
        import json as _json
        try:
            snap_path = f"colab_log_{job_id[:8]}_snapshot.json"
            with open(snap_path, "w") as f:
                parsed = _json.loads(snap)
                _json.dump(parsed, f, indent=2, default=str)
            print(f"  [JSON LOG] Wrote snapshot → {snap_path}")
        except Exception as e:
            print(f"  [JSON LOG] Failed to write snapshot: {e}")
        try:
            snap = json.dumps(_apply_fragment_merge_to_snapshot(json.loads(snap)))
        except Exception:
            pass
    if ocr:
        for line in ocr[-3:]:
            print(f"  [OCR LOG] {line}")

    return VideoAnalyzeStatusResponse(
        job_id=job_id,
        state=str(data.get("state", "unknown")),
        progress=float(data["progress"]) if data.get("progress") is not None else None,
        frame_id=int(data["frame_id"]) if data.get("frame_id") is not None else None,
        total_frames=int(data["total_frames"]) if data.get("total_frames") is not None else None,
        fps=float(data["fps"]) if data.get("fps") is not None else None,
        message=data.get("message"),
        json_snapshot=snap,
        ocr_log=data.get("ocr_log") or [],
    )


@app.post("/api/truck-runs/video/{job_id}/finalize", response_model=VideoAnalyzeFinalizeResponse)
def finalize_video_job(job_id: str, db: Session = Depends(get_db)) -> VideoAnalyzeFinalizeResponse:
    if not video_client.configured():
        raise HTTPException(status_code=400, detail="COLAB_INFER_URL is not configured.")

    try:
        data = video_client.get_video_job_result(job_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Failed to fetch job result: {exc}") from exc

    import json as _json
    final_path = f"colab_log_{job_id[:8]}_final.json"
    with open(final_path, "w") as f:
        _json.dump(data, f, indent=2, default=str)
    print(f"\n{'#'*70}")
    print(f"### RAW COLAB FINALIZE RESPONSE → {final_path} ###")
    print(f"{'#'*70}")

    payload_dict = data.get("result", data)
    payload_dict = _apply_demo_constraints_to_payload(payload_dict, _video_demo_constraints.pop(job_id, None))
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


@app.post("/api/truck-runs/multi-camera/start", response_model=MultiCameraAnalyzeStartResponse)
async def start_multi_camera_job(
    front: UploadFile | None = File(default=None),
    right: UploadFile | None = File(default=None),
    back: UploadFile | None = File(default=None),
    left: UploadFile | None = File(default=None),
    demo_constraints: UploadFile | None = File(default=None),
) -> MultiCameraAnalyzeStartResponse:
    if not video_client.configured():
        raise HTTPException(status_code=400, detail="COLAB_INFER_URL is not configured.")

    uploads = {"front": front, "right": right, "back": back, "left": left}
    selected = {camera: upload for camera, upload in uploads.items() if upload is not None}
    if len(selected) < 2:
        raise HTTPException(status_code=400, detail="Upload at least two camera videos.")

    constraints = await _read_optional_json_upload(demo_constraints, "demo constraints")
    aggregate_job_id = uuid4().hex
    cameras: dict[str, dict[str, Any]] = {}
    child_statuses: list[MultiCameraChildStatus] = []

    for camera, upload in selected.items():
        raw = await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"{camera} video is empty.")
        filename = f"{camera}_{upload.filename or 'upload.mp4'}"
        try:
            data = video_client.start_video_job(raw, filename)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"Failed to start {camera} camera job: {exc}") from exc
        child_job_id = data.get("job_id")
        if not child_job_id:
            raise HTTPException(status_code=502, detail=f"Invalid response for {camera} camera: {data}")
        cameras[camera] = {
            "job_id": child_job_id,
            "filename": upload.filename,
        }
        child_statuses.append(
            MultiCameraChildStatus(
                camera=camera,
                job_id=child_job_id,
                state=str(data.get("state", "queued")),
                message=str(data.get("message", "Video job submitted.")),
            )
        )

    _multi_camera_jobs[aggregate_job_id] = {
        "state": "queued",
        "cameras": cameras,
        "snapshots": {},
        "demo_constraints": constraints,
    }

    return MultiCameraAnalyzeStartResponse(
        mode="multi_camera_async",
        job_id=aggregate_job_id,
        state="queued",
        message=f"Started {len(cameras)} camera jobs.",
        cameras=child_statuses,
    )


@app.get("/api/truck-runs/multi-camera/{job_id}/status", response_model=MultiCameraAnalyzeStatusResponse)
def get_multi_camera_job_status(job_id: str) -> MultiCameraAnalyzeStatusResponse:
    job = _multi_camera_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Multi-camera job not found.")

    child_statuses: list[MultiCameraChildStatus] = []
    snapshots: dict[str, dict[str, Any]] = {}
    ocr_log: list[str] = []
    progress_values: list[float] = []
    child_states: list[str] = []

    for camera, meta in job["cameras"].items():
        child_job_id = meta["job_id"]
        try:
            data = video_client.get_video_job_status(child_job_id)
        except Exception as exc:  # pragma: no cover
            data = {
                "state": "unknown",
                "message": f"Unable to fetch {camera} status: {exc}",
            }
        state = str(data.get("state", "unknown"))
        child_states.append(state)
        if data.get("progress") is not None:
            progress_values.append(float(data["progress"]))
        if data.get("ocr_log"):
            ocr_log.extend([f"[{camera}] {line}" for line in data.get("ocr_log", [])[-6:]])
        if data.get("json_snapshot"):
            try:
                parsed = json.loads(data["json_snapshot"])
                snapshots[camera] = parsed
                job["snapshots"][camera] = parsed
            except (TypeError, json.JSONDecodeError):
                pass

        child_statuses.append(
            MultiCameraChildStatus(
                camera=camera,
                job_id=child_job_id,
                state=state,
                progress=float(data["progress"]) if data.get("progress") is not None else None,
                frame_id=int(data["frame_id"]) if data.get("frame_id") is not None else None,
                total_frames=int(data["total_frames"]) if data.get("total_frames") is not None else None,
                fps=float(data["fps"]) if data.get("fps") is not None else None,
                message=data.get("message"),
            )
        )

    merged_snapshot = _apply_demo_constraints_to_payload(
        _merge_multi_camera_payloads(job.get("snapshots") or snapshots),
        job.get("demo_constraints"),
    )
    state = _multi_camera_state(child_states)
    job["state"] = state
    return MultiCameraAnalyzeStatusResponse(
        job_id=job_id,
        state=state,
        progress=round(sum(progress_values) / len(progress_values), 4) if progress_values else None,
        message=f"{len(child_statuses)} camera jobs {state}.",
        cameras=child_statuses,
        json_snapshot=json.dumps(merged_snapshot),
        ocr_log=ocr_log[-30:],
    )


@app.post("/api/truck-runs/multi-camera/{job_id}/finalize", response_model=MultiCameraAnalyzeFinalizeResponse)
def finalize_multi_camera_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> MultiCameraAnalyzeFinalizeResponse:
    job = _multi_camera_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Multi-camera job not found.")

    camera_payloads: dict[str, dict[str, Any]] = {}
    child_states: list[str] = []
    for camera, meta in job["cameras"].items():
        child_job_id = meta["job_id"]
        try:
            data = video_client.get_video_job_result(child_job_id)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"Failed to fetch {camera} result: {exc}") from exc
        child_states.append(str(data.get("state", "completed")))
        payload = data.get("result", data)
        if isinstance(payload, dict):
            camera_payloads[camera] = payload

    merged_payload = _apply_demo_constraints_to_payload(
        _merge_multi_camera_payloads(camera_payloads),
        job.get("demo_constraints"),
    )
    try:
        payload_model = TruckRunImportRequest.model_validate(merged_payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid merged multi-camera payload: {exc}") from exc

    saved = _store_truck_run(payload_model, db)
    state = _multi_camera_state(child_states)
    job["state"] = state
    return MultiCameraAnalyzeFinalizeResponse(
        job_id=job_id,
        state=state,
        run_id=saved.run_id,
        stored_trucks=saved.stored_trucks,
    )


@app.get("/api/truck-runs/multi-camera/{job_id}/frame/{camera}")
def get_multi_camera_job_frame(job_id: str, camera: str) -> Response:
    job = _multi_camera_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Multi-camera job not found.")
    if camera not in job["cameras"]:
        raise HTTPException(status_code=404, detail="Camera not found for this job.")
    child_job_id = job["cameras"][camera]["job_id"]
    try:
        frame_bytes, content_type = video_client.get_video_job_frame(child_job_id)
    except requests.HTTPError as exc:  # pragma: no cover
        code = exc.response.status_code if exc.response is not None else None
        if code in (404, 409):
            return Response(status_code=204)
        raise HTTPException(status_code=502, detail=f"Failed to fetch {camera} frame: upstream HTTP {code}") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Failed to fetch {camera} frame: {exc}") from exc
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
