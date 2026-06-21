"""
Cross-camera aggregator tests.

Core scenario (the bug the aggregator exists to kill): two trucks sit in two lanes
at the SAME time. RIGHT and LEFT each see BOTH trucks at overlapping timestamps.
Grouping on time alone would merge truck A's plate with truck B's. The aggregator
must keep each lane's fields separate.

Asserts:
  1. No cross-lane field bleed — lane 1 keeps its plate, lane 2 keeps its plate.
  2. BACK attaches to the correct lane only (and never spawns a phantom truck).
  3. main_class (`type`) is never sourced from BACK, even when BACK emits one.
"""

from app.aggregator import AggregatorConfig, aggregate


def _field(text, conf, camera):
    """One OCR field reading in this engine's real shape ({text, confidence, camera})."""
    return {"text": text, "confidence": conf, "camera": camera}


def _truck(lane_id, track_id, first, last, camera, *, type_=None, type_conf=0.9, **fields):
    """Build one per-camera engine record. `fields` are field_name=(text, conf)."""
    rec = {
        "lane_id": lane_id,
        "track_id": track_id,
        "first_seen_time_sec": first,
        "last_seen_time_sec": last,
        "associated_info": {
            name: _field(val, conf, camera) for name, (val, conf) in fields.items()
        },
    }
    if type_ is not None:
        rec["type"] = type_
        rec["confidence_avg"] = type_conf
    return rec


def _build_results():
    """
    Two trucks, two lanes, overlapping time windows.

      Lane 1: Truck A, plate MU-A-1, container CONTA-1   (leaves at t=120)
      Lane 2: Truck B, plate MU-B-2, container CONTB-2   (leaves at t=125)

    FRONT sees lane 1 only. RIGHT and LEFT each see BOTH lanes at the same time.
    BACK fires at t=132 — within 10s of lane 2's exit (gap 7) but NOT lane 1's
    (gap 12) — so it must attach to lane 2 only. BACK also (wrongly) emits a plate
    and a `type`, which must both be ignored.
    """
    front = [
        _truck("1", 11, 0.0, 120.0, "front", type_="truck_with_container",
               license_plate=("MU-A-1", 0.91), container_number=("CONTA-1", 0.90)),
    ]
    right = [
        _truck("1", 21, 0.5, 119.0, "right", type_="truck_with_container",
               license_plate=("MU-A-1", 0.88), container_number=("CONTA-1", 0.87)),
        _truck("2", 22, 0.4, 125.0, "right", type_="truck_without_container",
               license_plate=("MU-B-2", 0.86), container_number=("CONTB-2", 0.85)),
    ]
    left = [
        _truck("1", 31, 1.0, 118.0, "left", type_="truck_with_container",
               license_plate=("MU-A-1", 0.84)),
        _truck("2", 32, 0.8, 124.0, "left", type_="truck_without_container",
               license_plate=("MU-B-2", 0.83)),
    ]
    # BACK: interior cam. Carries truck_number/truck_company (its real job) plus a
    # bogus license_plate and a bogus `type` that the aggregator must ignore.
    back = [
        _truck("2", 99, 132.0, 138.0, "back", type_="truck_with_container", type_conf=0.99,
               truck_number=("60752", 0.95), truck_company=("Seapun", 0.94),
               license_plate=("XX-BACK-XX", 0.99)),
    ]
    return {"front": front, "right": right, "left": left, "back": back}


def test_no_cross_lane_field_bleed():
    out = aggregate(_build_results(), AggregatorConfig())
    rows = {r["lane_id"]: r for r in out["trucks"]}

    assert out["truck_count"] == 2, "expected exactly two consolidated trucks"
    assert set(rows) == {"1", "2"}

    # Lane 1 keeps its own plate/container; lane 2 keeps its own. No bleed.
    assert rows["1"]["fields"]["license_plate"]["value"] == "MU-A-1"
    assert rows["2"]["fields"]["license_plate"]["value"] == "MU-B-2"
    assert rows["1"]["fields"]["container_number"]["value"] == "CONTA-1"
    assert rows["2"]["fields"]["container_number"]["value"] == "CONTB-2"

    # Gate fields must come from gate cameras only.
    assert rows["1"]["fields"]["license_plate"]["source"] in {"FRONT", "RIGHT", "LEFT"}
    assert rows["2"]["fields"]["license_plate"]["source"] in {"FRONT", "RIGHT", "LEFT"}


def test_back_attaches_to_correct_lane_only():
    out = aggregate(_build_results(), AggregatorConfig())
    rows = {r["lane_id"]: r for r in out["trucks"]}

    # BACK's interior fields land on lane 2 (the only event inside the window)...
    assert rows["2"]["fields"]["truck_number"]["value"] == "60752"
    assert rows["2"]["fields"]["truck_company"]["value"] == "Seapun"
    assert rows["2"]["fields"]["truck_number"]["source"] == "BACK"
    assert "BACK" in rows["2"]["camera_source"]

    # ...and never bleed onto lane 1.
    assert "truck_number" not in rows["1"]["fields"]
    assert "truck_company" not in rows["1"]["fields"]
    assert "BACK" not in rows["1"]["camera_source"]

    # No phantom truck was created for BACK, and no orphan here.
    assert out["truck_count"] == 2
    assert out["validation"]["orphan_back_records"] == []


def test_main_class_never_from_back():
    out = aggregate(_build_results(), AggregatorConfig())
    rows = {r["lane_id"]: r for r in out["trucks"]}

    # BACK emitted type=truck_with_container @0.99 for the lane-2 truck, but lane 2 is
    # truck_without_container per the gate cams. main_class must come from gate only.
    lane2_type = rows["2"]["fields"]["type"]
    assert lane2_type["value"] == "truck_without_container"
    assert lane2_type["source"] != "BACK"
    assert lane2_type["source"] in {"FRONT", "RIGHT", "LEFT"}

    # Lane 1's class comes from gate too.
    assert rows["1"]["fields"]["type"]["value"] == "truck_with_container"
    assert rows["1"]["fields"]["type"]["source"] != "BACK"


def test_back_attach_anchor_uses_gate_crossing():
    """
    The truck crosses the gate line at t=100 but lingers in gate view until t=120.
    Back appears at t=108 — within 10s of the physical CROSSING, but *before* last_seen.

    Default anchor (last_seen) -> back can't attach (108 < 120) -> orphan.
    gate_cross anchor          -> back attaches (108 is 8s after the 100s crossing).
    """
    def gate_rec(cam, tid, first, last, gct, **fields):
        r = _truck("1", tid, first, last, cam, type_="truck_with_container", **fields)
        r["gate_cross_time_sec"] = gct
        return r

    results = {
        "front": [gate_rec("front", 1, 0.0, 120.0, 100.0, license_plate=("MU-1", 0.9))],
        "right": [gate_rec("right", 2, 0.0, 119.0, 100.0)],
        "left": [gate_rec("left", 3, 0.0, 118.0, 100.0)],
        "back": [_truck("1", 9, 108.0, 113.0, "back", truck_number=("777", 0.9))],
    }

    out_default = aggregate(results, AggregatorConfig())
    assert "truck_number" not in out_default["trucks"][0]["fields"]
    assert len(out_default["validation"]["orphan_back_records"]) == 1

    cfg = AggregatorConfig()
    cfg.back_attach_anchor = "gate_cross"
    out_gc = aggregate(results, cfg)
    assert out_gc["trucks"][0]["fields"]["truck_number"]["value"] == "777"
    assert out_gc["validation"]["orphan_back_records"] == []


def test_back_orphan_when_outside_window():
    """BACK far outside the window attaches to nothing and never makes a phantom."""
    results = _build_results()
    results["back"][0]["first_seen_time_sec"] = 500.0
    results["back"][0]["last_seen_time_sec"] = 506.0

    out = aggregate(results, AggregatorConfig())
    assert out["truck_count"] == 2  # still just the two gate trucks
    assert len(out["validation"]["orphan_back_records"]) == 1
