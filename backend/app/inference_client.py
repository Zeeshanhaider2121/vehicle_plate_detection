from __future__ import annotations

import hashlib
from typing import Any

import requests

from .config import settings
from .schemas import InferenceRecord


def _pick_first(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _parse_plate_text(raw: dict[str, Any]) -> str:
    text = _pick_first(
        raw,
        ["plate_text", "plate", "license_plate", "licensePlate", "text", "prediction"],
    )
    if text is None:
        return "UNKNOWN"
    return str(text).strip().upper() or "UNKNOWN"


def _parse_confidence(raw: dict[str, Any]) -> float | None:
    confidence = _pick_first(raw, ["confidence", "score", "probability"])
    if confidence is None:
        return None

    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return None

    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return round(value, 4)


def _parse_bbox(raw: dict[str, Any]) -> dict[str, Any] | list[Any] | None:
    return _pick_first(raw, ["bbox", "box", "xyxy", "rect"])


def _extract_detections(payload: dict[str, Any]) -> list[InferenceRecord]:
    raw_detections: Any = _pick_first(payload, ["detections", "predictions", "results"])

    if raw_detections is None and "plate_text" in payload:
        raw_detections = [payload]

    if not isinstance(raw_detections, list):
        return []

    output: list[InferenceRecord] = []
    for item in raw_detections:
        if not isinstance(item, dict):
            continue

        output.append(
            InferenceRecord(
                plate_text=_parse_plate_text(item),
                confidence=_parse_confidence(item),
                bbox=_parse_bbox(item),
            )
        )
    return output


def _mock_detections(image_bytes: bytes, source_name: str | None) -> list[InferenceRecord]:
    digest = hashlib.sha256(image_bytes).hexdigest()
    seed = int(digest[:8], 16)
    suffix = seed % 9000 + 1000
    confidence = 0.7 + ((seed % 30) / 100)
    detection = InferenceRecord(
        plate_text=f"TEST{suffix}",
        confidence=min(round(confidence, 3), 0.99),
        bbox={"x1": 120, "y1": 220, "x2": 320, "y2": 300},
    )

    extra: list[InferenceRecord] = []
    if source_name and seed % 2 == 0:
        extra.append(
            InferenceRecord(
                plate_text=f"ALT{(suffix + 7) % 9999:04d}",
                confidence=min(round(confidence - 0.12, 3), 0.95),
                bbox={"x1": 340, "y1": 230, "x2": 520, "y2": 305},
            )
        )

    return [detection, *extra]


class InferenceService:
    def infer(self, image_bytes: bytes, source_name: str | None) -> tuple[list[InferenceRecord], dict[str, Any], bool]:
        if settings.colab_infer_url:
            try:
                infer_url = f"{settings.colab_infer_url.rstrip('/')}{settings.colab_infer_path}"
                response = requests.post(
                    infer_url,
                    files={"file": (source_name or "upload.jpg", image_bytes, "application/octet-stream")},
                    timeout=settings.inference_timeout_seconds,
                )
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                detections = _extract_detections(payload)
                if detections:
                    return detections, payload, False
                if not settings.mock_inference_if_unavailable:
                    raise ValueError("No detections in response payload")
            except Exception as exc:  # pragma: no cover
                if not settings.mock_inference_if_unavailable:
                    raise RuntimeError(f"Inference call failed: {exc}") from exc
                payload = {"mock_fallback_reason": str(exc)}
                return _mock_detections(image_bytes, source_name), payload, True

        payload = {"mock_fallback_reason": "COLAB_INFER_URL not configured"}
        return _mock_detections(image_bytes, source_name), payload, True
