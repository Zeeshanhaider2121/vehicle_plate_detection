"""
Tests for the time-synchronized cross-camera gate sweep in
``app.main._merge_multi_camera_payloads`` (consolidation: one record per physical
truck).

Required cases (per spec):
  (a) ``right`` splits one truck into two ids while front/left see one continuous
      track  -> ONE consolidated truck, both right ids as members.
  (b) two trucks tailgating, all three front cams see both entries -> TWO trucks.
  (c) a back track that appears while the truck is still active up front is NOT
      attached (until after front deactivation).

Plus two controls: back attaches once the truck deactivates, and plate provenance
follows the configured camera order.

These call the live consolidation entry point with crafted per-camera payloads
(shaped exactly like a single-camera engine job result), so the whole sweep —
entry detection, absorption, back-attach, field merge — is exercised end to end.
"""

import pytest

import app.main as main_module
from app.main import _merge_multi_camera_payloads, _field_text


# ── builders ───────────────────────────────────────────────────────────────────

def _truck(track_id, type_, t0, t1, *, fps=30.0, **fields):
    """One per-camera truck record. `fields` are field_name=(text, confidence)."""
    info = {name: {"text": v, "confidence": c, "camera": ""} for name, (v, c) in fields.items()}
    return {
        "track_id": track_id,
        "type": type_,
        "first_seen_frame": int(t0 * fps),
        "last_seen_frame": int(t1 * fps),
        "first_seen_time_sec": float(t0),
        "last_seen_time_sec": float(t1),
        "duration_frames": int((t1 - t0) * fps) + 1,
        "duration_sec": round(t1 - t0, 3),
        "confidence_avg": 0.9,
        "last_bbox": [0, 0, 10, 10],
        "associated_info": info,
    }


def _payload(*tracks, fps=30.0):
    """One per-camera job payload: {session, trucks}."""
    return {
        "session": {
            "video_fps": fps,
            "total_frames": int(fps * 300),
            "frames_processed": int(fps * 300),
            "device": "cpu",
            "model": "test-model",
            "video_path": "clip.mp4",  # no NVR timestamp -> zero clock offset
        },
        "trucks": {str(t["track_id"]): t for t in tracks},
    }


def _trucks_out(result):
    return list(result["trucks"].values())


def _cam_observations(entity):
    return entity["associated_info"].get("_camera_observations") or []


# ── deterministic sweep config (independent of env / .env) ──────────────────────

@pytest.fixture(autouse=True)
def _sweep_config(monkeypatch):
    monkeypatch.setattr(main_module, "MULTI_CAMERA_GATE_SWEEP", True)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_GATE_MODE", False)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_ASSUME_SINGLE_ENTITY", False)
    monkeypatch.setattr(main_module, "TRUCK_TIME_BOUNDARIES", [])
    monkeypatch.setattr(main_module, "CAMERA_TIME_OFFSETS_SECONDS", {})
    monkeypatch.setattr(main_module, "FRONT_FACING_CAMERAS", {"front", "right", "left"})
    monkeypatch.setattr(main_module, "BACK_CAMERAS", {"back"})
    monkeypatch.setattr(main_module, "MIN_START_SUPPORT", 2)
    monkeypatch.setattr(main_module, "CORROBORATION_WINDOW_S", 1.5)
    monkeypatch.setattr(main_module, "NEW_TRUCK_GAP_S", 6.0)
    monkeypatch.setattr(main_module, "BACK_ATTACH_LEAD_S", 1.0)
    monkeypatch.setattr(main_module, "BACK_ATTACH_WINDOW_S", 13.0)
    monkeypatch.setattr(
        main_module, "MULTI_CAMERA_CAMERA_ROLES",
        {"back": {"truck_number", "truck_company", "driver"}},
    )
    monkeypatch.setattr(main_module, "PLATE_PROVENANCE_ORDER", ["front", "right", "left", "back"])


# ── (a) right splits one truck — must stay ONE truck ────────────────────────────

def test_right_split_is_one_truck():
    """right loses the truck mid-pass and re-acquires it with a new id while
    front/left stay continuous -> one consolidated truck owning BOTH right ids."""
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0,
                                 license_plate=("MU-A-1", 0.90))),
        "left":  _payload(_truck(1, "truck_with_container", 0.0, 119.0)),
        "right": _payload(
            _truck(1, "truck_with_container", 0.0, 50.0),    # first right id
            _truck(2, "truck_with_container", 80.0, 120.0),  # re-acquired right id
        ),
    }
    result = _merge_multi_camera_payloads(payloads)

    assert result["summary"]["total_trucks_tracked"] == 1
    entity = _trucks_out(result)[0]

    right_obs = [o for o in _cam_observations(entity) if o["camera"] == "right"]
    assert len(right_obs) == 2, "both right track ids must be members of the one truck"
    assert {o["source_track_id"] for o in right_obs} == {1, 2}


# ── (b) two tailgating trucks — must stay TWO trucks ────────────────────────────

def test_tailgating_two_trucks_stay_separate():
    """Two trucks enter ~6 s apart; all three front cams see BOTH entries.
    The second entry is corroborated -> a distinct truck (no merge)."""
    payloads = {
        "front": _payload(
            _truck(1, "truck_with_container", 0.0, 120.0, license_plate=("MU-A-1", 0.92)),
            _truck(2, "truck_without_container", 126.0, 240.0, license_plate=("MU-B-2", 0.92)),
        ),
        "right": _payload(
            _truck(1, "truck_with_container", 0.2, 118.0, license_plate=("MU-A-1", 0.80)),
            _truck(2, "truck_without_container", 126.3, 238.0, license_plate=("MU-B-2", 0.80)),
        ),
        "left": _payload(
            _truck(1, "truck_with_container", 0.4, 119.0),
            _truck(2, "truck_without_container", 126.6, 239.0),
        ),
    }
    result = _merge_multi_camera_payloads(payloads)

    assert result["summary"]["total_trucks_tracked"] == 2
    plates = sorted(_field_text(e["associated_info"].get("license_plate")) for e in _trucks_out(result))
    assert plates == ["MU-A-1", "MU-B-2"], "each truck keeps its own plate; no merge/bleed"


# ── (c) back must not attach while the truck is still active up front ───────────

def test_back_not_attached_while_truck_active():
    """A back track that fires at t=60 s, while the truck is still active up front
    (exit=120 s), is NOT attached and does NOT spawn a phantom truck."""
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0,
                                 license_plate=("MU-A-1", 0.90))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
        "back":  _payload(_truck(1, "truck_with_container", 60.0, 70.0,
                                 truck_number=("60752", 0.95))),
    }
    result = _merge_multi_camera_payloads(payloads)

    assert result["summary"]["total_trucks_tracked"] == 1, "back must not spawn a truck"
    entity = _trucks_out(result)[0]
    assert "back" not in [o["camera"] for o in _cam_observations(entity)]
    assert not _field_text(entity["associated_info"].get("truck_number")), (
        "back's truck_number must not attach while the truck is still active up front"
    )


def test_back_attaches_after_deactivation():
    """Same truck; the back track now fires at t=122 s, just after the front exit
    (120 s) and inside the attach window -> it attaches and contributes truck_number."""
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0,
                                 license_plate=("MU-A-1", 0.90))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
        "back":  _payload(_truck(1, "truck_with_container", 122.0, 130.0,
                                 truck_number=("60752", 0.95),
                                 truck_company=("Seapun", 0.93))),
    }
    result = _merge_multi_camera_payloads(payloads)

    assert result["summary"]["total_trucks_tracked"] == 1
    entity = _trucks_out(result)[0]
    assert "back" in [o["camera"] for o in _cam_observations(entity)]
    assert _field_text(entity["associated_info"].get("truck_number")) == "60752"
    assert _field_text(entity["associated_info"].get("truck_company")) == "Seapun"


# ── plate provenance follows the configured camera order ────────────────────────

def test_plate_provenance_prefers_configured_camera(monkeypatch):
    """front and right both read a plate. With order right-first, right's read wins
    even though front also has one — provenance is by camera order, not first-seen."""
    monkeypatch.setattr(main_module, "PLATE_PROVENANCE_ORDER", ["right", "front", "left", "back"])
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0,
                                 license_plate=("FRONT-PLATE", 0.99))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0,
                                 license_plate=("RIGHT-PLATE", 0.70))),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
    }
    result = _merge_multi_camera_payloads(payloads)
    entity = _trucks_out(result)[0]
    assert _field_text(entity["associated_info"]["license_plate"]) == "RIGHT-PLATE"
