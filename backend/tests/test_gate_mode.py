import pytest
from app.main import (
    _parse_camera_roles,
    _merge_truck_group,
    _merge_multi_camera_payloads,
)
import app.main as main_module


# ── helpers ───────────────────────────────────────────────────────────────────

def _obs(camera: str, track_id: int, truck_type: str, fields: dict) -> dict:
    """Build a minimal camera observation dict."""
    return {
        "camera": camera,
        "truck": {
            "track_id": track_id,
            "type": truck_type,
            "associated_info": {k: {"text": v, "confidence": 0.7} for k, v in fields.items()},
            "confidence_avg": 0.9,
            "first_seen_frame": 0,
            "last_seen_frame": 30,
            "first_seen_time_sec": 0.0,
            "last_seen_time_sec": 1.0,
            "time_offset_sec": 0.0,
        },
    }


def _camera_payload(track_id: int, truck_type: str, fields: dict) -> dict:
    """Build a minimal per-camera payload dict (as returned by Colab)."""
    info = {k: {"text": v, "confidence": 0.7} for k, v in fields.items()}
    return {
        "session": {"video_fps": 30, "total_frames": 300, "frames_processed": 300,
                    "device": "cuda", "model": "yolo", "video_path": "test.mp4"},
        "trucks": {
            str(track_id): {
                "track_id": track_id,
                "type": truck_type,
                "first_seen_frame": 0,
                "last_seen_frame": 30,
                "first_seen_time_sec": 0.0,
                "last_seen_time_sec": 1.0,
                "duration_frames": 31,
                "duration_sec": 1.0,
                "confidence_avg": 0.9,
                "last_bbox": [10, 10, 100, 200],
                "associated_info": info,
            }
        },
    }


# ── _parse_camera_roles ───────────────────────────────────────────────────────

def test_parse_camera_roles_empty():
    assert _parse_camera_roles("") == {}


def test_parse_camera_roles_single():
    result = _parse_camera_roles("front:license_plate,truck_number")
    assert result == {"front": {"license_plate", "truck_number"}}


def test_parse_camera_roles_multiple():
    raw = "front:license_plate,truck_number;left:container_number,container_side_no"
    result = _parse_camera_roles(raw)
    assert result["front"] == {"license_plate", "truck_number"}
    assert result["left"] == {"container_number", "container_side_no"}


def test_parse_camera_roles_ignores_empty_parts():
    result = _parse_camera_roles(";front:license_plate;;")
    assert result == {"front": {"license_plate"}}


def test_parse_camera_roles_strips_whitespace():
    result = _parse_camera_roles(" front : license_plate , truck_number ")
    assert result == {"front": {"license_plate", "truck_number"}}


# ── _merge_truck_group with camera_roles ─────────────────────────────────────

def test_camera_roles_auth_wins_tiebreak():
    """Authoritative camera value wins when confidence is equal."""
    observations = [
        _obs("front", 1, "truck_without_container", {"license_plate": "AUTH_PLATE"}),
        _obs("right", 2, "truck_without_container", {"license_plate": "OFF_ROLE_PLATE"}),
    ]
    roles = {"front": {"license_plate", "truck_number"}}
    result = _merge_truck_group(1, observations, "gate_mode", 0.95, roles)
    assert result["associated_info"]["license_plate"]["text"] == "AUTH_PLATE"


def test_camera_roles_higher_confidence_still_wins():
    """Off-role camera wins if its confidence is strictly higher."""
    observations = [
        _obs("front", 1, "truck_without_container", {"license_plate": "LOW_CONF"}),
        _obs("right", 2, "truck_without_container", {"license_plate": "HIGH_CONF"}),
    ]
    # Manually set confidences
    observations[0]["truck"]["associated_info"]["license_plate"]["confidence"] = 0.5
    observations[1]["truck"]["associated_info"]["license_plate"]["confidence"] = 0.95
    roles = {"front": {"license_plate"}}
    result = _merge_truck_group(1, observations, "gate_mode", 0.95, roles)
    assert result["associated_info"]["license_plate"]["text"] == "HIGH_CONF"


def test_camera_roles_no_roles_unchanged():
    """Without camera_roles, existing _better_field logic applies (no regression)."""
    observations = [
        _obs("front", 1, "truck_without_container", {"license_plate": "PLATE_A"}),
        _obs("back", 2, "truck_without_container", {"license_plate": "PLATE_B"}),
    ]
    result_no_roles = _merge_truck_group(1, observations, "gate_mode", 0.95, None)
    result_roles = _merge_truck_group(1, observations, "gate_mode", 0.95, {"front": {"license_plate"}})
    # With roles: front wins. Without roles: whichever _better_field picks (same conf → first seen).
    assert result_roles["associated_info"]["license_plate"]["text"] == "PLATE_A"
    # No roles: first observation (front) still wins because same confidence and same text quality.
    assert result_no_roles["associated_info"]["license_plate"]["text"] == "PLATE_A"


def test_camera_roles_written_to_fusion_metadata():
    """_fusion.camera_roles_applied is True when roles are provided."""
    observations = [_obs("front", 1, "truck_without_container", {})]
    result = _merge_truck_group(1, observations, "gate_mode", 0.95, {"front": {"license_plate"}})
    assert result["associated_info"]["_fusion"]["camera_roles_applied"] is True


def test_no_camera_roles_fusion_metadata_false():
    observations = [_obs("front", 1, "truck_without_container", {})]
    result = _merge_truck_group(1, observations, "gate_mode", 0.95, None)
    assert result["associated_info"]["_fusion"]["camera_roles_applied"] is False


# ── _merge_multi_camera_payloads gate_mode ────────────────────────────────────

def test_gate_mode_merges_all_cameras_into_one(monkeypatch):
    """With MULTI_CAMERA_GATE_MODE=True all observations collapse into 1 truck."""
    monkeypatch.setattr(main_module, "MULTI_CAMERA_GATE_MODE", True)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_CAMERA_ROLES", {})

    payloads = {
        "front": _camera_payload(1, "truck_without_container", {"license_plate": "ABC123"}),
        "right": _camera_payload(2, "truck_with_container", {"container_number": "MSCU1234560"}),
        "back":  _camera_payload(3, "truck_without_container", {"license_plate": "ABC123"}),
        "left":  _camera_payload(4, "truck_with_container", {"container_number": "MSCU1234560"}),
    }
    result = _merge_multi_camera_payloads(payloads)
    assert result["summary"]["total_trucks_tracked"] == 1


def test_gate_mode_confidence_is_095(monkeypatch):
    """Gate mode sets match_confidence to 0.95."""
    monkeypatch.setattr(main_module, "MULTI_CAMERA_GATE_MODE", True)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_CAMERA_ROLES", {})

    payloads = {
        "front": _camera_payload(1, "truck_without_container", {"license_plate": "XYZ"}),
        "left":  _camera_payload(2, "truck_with_container", {"container_number": "ABCD1234560"}),
    }
    result = _merge_multi_camera_payloads(payloads)
    fusion = result["trucks"]["1"]["associated_info"]["_fusion"]
    assert fusion["match_method"] == "gate_mode"
    assert fusion["match_confidence"] == 0.95


def test_gate_mode_false_does_not_force_merge(monkeypatch):
    """Without gate_mode, cameras with no shared identity stay separate."""
    monkeypatch.setattr(main_module, "MULTI_CAMERA_GATE_MODE", False)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_ASSUME_SINGLE_ENTITY", False)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_ORDER_FALLBACK", False)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_CAMERA_ROLES", {})

    payloads = {
        "front": _camera_payload(1, "truck_without_container", {"license_plate": "PLATE1"}),
        "left":  _camera_payload(2, "truck_with_container", {"container_number": "ABCD1234560"}),
    }
    result = _merge_multi_camera_payloads(payloads)
    # No shared identity → separate entries
    assert result["summary"]["total_trucks_tracked"] == 2


def test_gate_mode_with_camera_roles_fields_routed_correctly(monkeypatch):
    """Gate mode + camera roles: license_plate comes from front, container_number from left.

    Values are chosen so that _better_field cannot decide by length alone — the
    authoritative-camera ordering introduced by camera_roles is the deciding factor.
    Both cameras report a value for both fields; the authoritative camera's value wins
    for its own field because it is processed first (tiebreak: first-seen wins when
    confidence and text quality are equal).
    """
    roles = {
        "front": {"license_plate", "truck_number"},
        "left": {"container_number", "container_side_no"},
    }
    monkeypatch.setattr(main_module, "MULTI_CAMERA_GATE_MODE", True)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_CAMERA_ROLES", roles)

    # Use same-length strings so _better_field cannot prefer one by length; the
    # authoritative-camera-first ordering then determines the winner.
    payloads = {
        "front": _camera_payload(1, "truck_with_container", {
            "license_plate": "PLATE01",   # front is auth for license_plate → wins
            "container_number": "WRONG01",  # front is NOT auth for container_number
        }),
        "left": _camera_payload(2, "truck_with_container", {
            "license_plate": "WRONG01",    # left is NOT auth for license_plate
            "container_number": "CONTE01",  # left is auth for container_number → wins
        }),
    }
    result = _merge_multi_camera_payloads(payloads)
    info = result["trucks"]["1"]["associated_info"]
    assert info["license_plate"]["text"] == "PLATE01"
    assert info["container_number"]["text"] == "CONTE01"
