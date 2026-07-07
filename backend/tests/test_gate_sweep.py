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

def _truck(track_id, type_, t0, t1, *, fps=30.0, bbox=None, union_bbox=None, ocr_history=None, **fields):
    """One per-camera truck record. `fields` are field_name=(text, confidence).

    `bbox`/`union_bbox` set the spatial position (for side-by-side split tests);
    `ocr_history` injects an associated_info["_ocr_history"] block (for the
    second-container-number rule)."""
    info = {name: {"text": v, "confidence": c, "camera": ""} for name, (v, c) in fields.items()}
    if ocr_history is not None:
        info["_ocr_history"] = ocr_history
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
        "last_bbox": bbox if bbox is not None else [0, 0, 10, 10],
        "union_bbox": union_bbox,
        "associated_info": info,
    }


def _payload(*tracks, fps=30.0, resolution="1280x720"):
    """One per-camera job payload: {session, trucks}."""
    return {
        "session": {
            "video_fps": fps,
            "total_frames": int(fps * 300),
            "frames_processed": int(fps * 300),
            "device": "cpu",
            "model": "test-model",
            "resolution": resolution,
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
    monkeypatch.setattr(main_module, "BACK_ATTACH_BY_TIME", True)
    monkeypatch.setattr(
        main_module, "MULTI_CAMERA_CAMERA_ROLES",
        {"back": {"truck_number", "truck_company", "driver"}},
    )
    monkeypatch.setattr(main_module, "PLATE_PROVENANCE_ORDER", ["front", "right", "left", "back"])
    # New knobs — pinned so existing tests stay deterministic and .env-independent.
    monkeypatch.setattr(main_module, "BACK_EXTRA_OFFSET_S", 0.0)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_SHARED_FIELDS", {"truck_number", "truck_company"})
    monkeypatch.setattr(main_module, "SHARED_FIELD_CONF_TIE_EPS", 0.02)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_SIDE_BY_SIDE_SPLIT", True)
    monkeypatch.setattr(main_module, "SIDE_BY_SIDE_MIN_OVERLAP_RATIO", 0.5)
    monkeypatch.setattr(main_module, "SIDE_BY_SIDE_MIN_CENTERX_GAP", 0.15)
    monkeypatch.setattr(main_module, "CAMERA_ORIENTATION_FLIP", set())
    monkeypatch.setattr(main_module, "DEFAULT_FRAME_WIDTH", 1280)


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


# ── side-by-side trucks (same time, different lanes) split by horizontal position ─

_LANE_A = [220, 300, 420, 600]    # centre x = 320 -> normalized 0.25 at 1280 px
_LANE_B = [860, 300, 1060, 600]   # centre x = 960 -> normalized 0.75 at 1280 px


def test_side_by_side_two_lanes_split():
    """Two trucks pass at the SAME time in different lanes; right & left each see
    BOTH. They must become TWO dashboard entries, split by horizontal position,
    each keeping its own plate (no cross-lane bleed)."""
    payloads = {
        "right": _payload(
            _truck(1, "truck_with_container", 0.0, 120.0, bbox=_LANE_A, union_bbox=_LANE_A,
                   license_plate=("LANE-A", 0.90)),
            _truck(2, "truck_with_container", 0.1, 118.0, bbox=_LANE_B, union_bbox=_LANE_B,
                   license_plate=("LANE-B", 0.90)),
        ),
        "left": _payload(
            _truck(1, "truck_with_container", 0.0, 119.0, bbox=_LANE_A, union_bbox=_LANE_A,
                   license_plate=("LANE-A", 0.85)),
            _truck(2, "truck_with_container", 0.2, 117.0, bbox=_LANE_B, union_bbox=_LANE_B,
                   license_plate=("LANE-B", 0.85)),
        ),
    }
    result = _merge_multi_camera_payloads(payloads)

    assert result["summary"]["total_trucks_tracked"] == 2
    plates = sorted(_field_text(e["associated_info"].get("license_plate")) for e in _trucks_out(result))
    assert plates == ["LANE-A", "LANE-B"], "each lane keeps its own plate; no merge/bleed"


def test_side_by_side_no_split_without_overlap():
    """Same two horizontal positions but the second truck arrives AFTER the first
    leaves (no time overlap) -> a re-acquisition is absorbed, not a side-by-side
    split. Guards against splitting sequential trucks in one lane."""
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0, bbox=_LANE_A, union_bbox=_LANE_A,
                                 license_plate=("ONLY-ONE", 0.90))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0, bbox=_LANE_A, union_bbox=_LANE_A)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0, bbox=_LANE_A, union_bbox=_LANE_A)),
    }
    result = _merge_multi_camera_payloads(payloads)
    assert result["summary"]["total_trucks_tracked"] == 1


def test_side_by_side_flip_orientation(monkeypatch):
    """Side cameras face the gate from opposite angles, so the SAME truck appears on
    opposite screen sides. With `left` flagged 'flip', the normalized positions
    re-align and the lanes are paired correctly (no cross-lane bleed)."""
    monkeypatch.setattr(main_module, "CAMERA_ORIENTATION_FLIP", {"left"})
    payloads = {
        "right": _payload(
            _truck(1, "truck_with_container", 0.0, 120.0, bbox=_LANE_A, union_bbox=_LANE_A,
                   license_plate=("WORLD-LEFT", 0.90)),
            _truck(2, "truck_with_container", 0.1, 118.0, bbox=_LANE_B, union_bbox=_LANE_B,
                   license_plate=("WORLD-RIGHT", 0.90)),
        ),
        # mirrored on the left camera's screen; the flip undoes it
        "left": _payload(
            _truck(1, "truck_with_container", 0.0, 119.0, bbox=_LANE_B, union_bbox=_LANE_B,
                   license_plate=("WORLD-LEFT", 0.80)),
            _truck(2, "truck_with_container", 0.2, 117.0, bbox=_LANE_A, union_bbox=_LANE_A,
                   license_plate=("WORLD-RIGHT", 0.80)),
        ),
    }
    result = _merge_multi_camera_payloads(payloads)
    assert result["summary"]["total_trucks_tracked"] == 2
    plates = sorted(_field_text(e["associated_info"].get("license_plate")) for e in _trucks_out(result))
    assert plates == ["WORLD-LEFT", "WORLD-RIGHT"]


# ── truck_number / truck_company taken from ALL cameras (back tiebreaker) ────────

def test_shared_truck_number_best_confidence():
    """truck_number is read by both front and back; the higher-confidence FRONT read
    wins even though back is the role owner."""
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0, truck_number=("11111", 0.95))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
        "back":  _payload(_truck(1, "truck_with_container", 122.0, 130.0, truck_number=("22222", 0.80))),
    }
    result = _merge_multi_camera_payloads(payloads)
    entity = _trucks_out(result)[0]
    assert _field_text(entity["associated_info"]["truck_number"]) == "11111"


def test_shared_truck_number_tie_prefers_back():
    """Equal confidence -> the role owner (back) wins the tie."""
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0, truck_number=("11111", 0.90))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
        "back":  _payload(_truck(1, "truck_with_container", 122.0, 130.0, truck_number=("22222", 0.90))),
    }
    result = _merge_multi_camera_payloads(payloads)
    entity = _trucks_out(result)[0]
    assert _field_text(entity["associated_info"]["truck_number"]) == "22222"


# ── two container numbers -> second becomes the side number ──────────────────────

_CONTAINER_HIST = {
    "container_number": {
        "MSCU1234567": {"text": "MSCU1234567", "confidence": 0.95, "count": 12, "camera": "front"},
        "TGHU7654321": {"text": "TGHU7654321", "confidence": 0.88, "count": 7, "camera": "right"},
    }
}


def test_two_containers_second_becomes_side():
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0,
                                 container_number=("MSCU1234567", 0.95),
                                 ocr_history=_CONTAINER_HIST)),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
    }
    info = _trucks_out(_merge_multi_camera_payloads(payloads))[0]["associated_info"]
    assert _field_text(info["container_number"]) == "MSCU1234567"
    assert _field_text(info["container_side_no"]) == "TGHU7654321"


def test_two_containers_respects_existing_side_no():
    """An existing (real) side-number read must not be overwritten by the 2nd
    container number."""
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0,
                                 container_number=("MSCU1234567", 0.95),
                                 container_side_no=("SIDE-999", 0.90),
                                 ocr_history=_CONTAINER_HIST)),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
    }
    info = _trucks_out(_merge_multi_camera_payloads(payloads))[0]["associated_info"]
    assert _field_text(info["container_side_no"]) == "SIDE-999"


# ── back camera +10 s processing lag ─────────────────────────────────────────────

def test_back_plus_10s_lag(monkeypatch):
    """The back clip's local time shows the truck at 112 s; the +10 s lag shifts it
    to 122 s on the shared axis, just after the front exit (120 s) and inside the
    (widened) attach window -> it attaches and contributes truck_number."""
    monkeypatch.setattr(main_module, "BACK_EXTRA_OFFSET_S", 10.0)
    monkeypatch.setattr(main_module, "BACK_ATTACH_WINDOW_S", 23.0)
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0, license_plate=("MU-A-1", 0.90))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
        "back":  _payload(_truck(1, "truck_with_container", 112.0, 120.0, truck_number=("60752", 0.95))),
    }
    result = _merge_multi_camera_payloads(payloads)
    assert result["summary"]["total_trucks_tracked"] == 1
    entity = _trucks_out(result)[0]
    assert "back" in [o["camera"] for o in _cam_observations(entity)]
    assert _field_text(entity["associated_info"]["truck_number"]) == "60752"


# ── back reads attributed BY TIME to the right truck (multi-truck runs) ───────────

def _back_hist(**field_variants):
    """Build an _ocr_history block: field=[(text, conf, count, t_first, t_last), ...]."""
    hist: dict = {}
    for field, variants in field_variants.items():
        hist[field] = {
            text: {
                "text": text, "confidence": conf, "count": count, "camera": "back",
                "time_first_sec": tf, "time_last_sec": tl,
            }
            for (text, conf, count, tf, tl) in variants
        }
    return hist


def test_back_reads_split_by_time_across_two_trucks():
    """The back camera runs ONE continuous track whose OCR history spans BOTH trucks.
    Each timestamped read must attach to the truck whose front exit it is nearest, so
    truck 1 gets its own number/company and truck 2 gets its own — no cross-bleed."""
    back_hist = _back_hist(
        truck_number=[("111", 0.95, 8, 121.0, 124.0), ("222", 0.95, 8, 241.0, 244.0)],
        truck_company=[("Alpha", 0.90, 6, 121.0, 124.0), ("Beta", 0.90, 6, 241.0, 244.0)],
    )
    payloads = {
        "front": _payload(
            _truck(1, "truck_with_container", 0.0, 120.0, license_plate=("MU-A-1", 0.92)),
            _truck(2, "truck_without_container", 126.0, 240.0, license_plate=("MU-B-2", 0.92)),
        ),
        "right": _payload(
            _truck(1, "truck_with_container", 0.2, 118.0),
            _truck(2, "truck_without_container", 126.3, 238.0),
        ),
        "left": _payload(
            _truck(1, "truck_with_container", 0.4, 119.0),
            _truck(2, "truck_without_container", 126.6, 239.0),
        ),
        # ONE back track spanning both trucks, reads timed near each front exit.
        "back": _payload(_truck(1, "truck_with_container", 118.0, 246.0, ocr_history=back_hist)),
    }
    result = _merge_multi_camera_payloads(payloads)

    assert result["summary"]["total_trucks_tracked"] == 2, "back must not spawn extra trucks"
    by_plate = {
        _field_text(e["associated_info"].get("license_plate")): e["associated_info"]
        for e in _trucks_out(result)
    }
    assert _field_text(by_plate["MU-A-1"].get("truck_number")) == "111"
    assert _field_text(by_plate["MU-A-1"].get("truck_company")) == "Alpha"
    assert _field_text(by_plate["MU-B-2"].get("truck_number")) == "222"
    assert _field_text(by_plate["MU-B-2"].get("truck_company")) == "Beta"


def test_back_read_outside_all_windows_is_dropped():
    """A back read captured while the truck is still active up front (before ANY exit)
    matches no attach window and is dropped — it must not leak onto the truck."""
    back_hist = _back_hist(
        truck_number=[
            ("EARLY", 0.95, 5, 60.0, 63.0),    # mid 61.5 — no exit in [38.5, 62.5]
            ("60752", 0.95, 8, 121.0, 124.0),  # mid 122.5 — matches the 120 s exit
        ],
    )
    payloads = {
        "front": _payload(_truck(1, "truck_with_container", 0.0, 120.0, license_plate=("MU-A-1", 0.90))),
        "right": _payload(_truck(1, "truck_with_container", 0.2, 118.0)),
        "left":  _payload(_truck(1, "truck_with_container", 0.4, 119.0)),
        "back":  _payload(_truck(1, "truck_with_container", 55.0, 126.0, ocr_history=back_hist)),
    }
    result = _merge_multi_camera_payloads(payloads)
    assert result["summary"]["total_trucks_tracked"] == 1
    entity = _trucks_out(result)[0]
    assert _field_text(entity["associated_info"].get("truck_number")) == "60752", (
        "only the in-window read attaches; the early read is dropped"
    )
