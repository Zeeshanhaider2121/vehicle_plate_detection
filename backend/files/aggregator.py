"""
PlateFlow cross-camera aggregator.

Takes the four per-camera job results (front / right / left / back), each produced
independently by the single-camera engine in main.py, and collapses them into ONE
consolidated row per physical truck.

Key ideas
---------
1. GATE GROUPING is lane-aware, not time-only.
   FRONT/RIGHT/LEFT fire near-simultaneously, but RIGHT and LEFT each see BOTH lanes,
   so two different trucks can produce overlapping timestamps. Grouping purely on time
   would cross-contaminate their OCR fields. Therefore the grouping key is
   (lane_id + concurrent time window). Records in DIFFERENT lanes are NEVER merged,
   even when their timestamps overlap.

2. BACK ATTACHMENT is deferred and time-based.
   BACK sits just inside the gate door and sees the same truck a few seconds after it
   clears the gate cameras. A BACK record attaches to a gate event when its
   first_seen falls in (gate.last_seen , gate.last_seen + BACK_ATTACH_WINDOW_SEC].
   If several gate events qualify, the MOST RECENT one wins.

3. FIELD PROVENANCE is role-enforced.
   - Gate role (FRONT/RIGHT/LEFT): container_no, company_logo, side_no, license_plate,
     and the main class (truck_with_container / truck_without_container).
   - Interior role (BACK): truck_no, truck_company ONLY. BACK never contributes a main
     class. Out-of-role fields from any camera are ignored.

4. NO HARDCODING.
   Window length, sync tolerance, per-camera time offsets, the camera->role map, and
   the field->role map all live in AggregatorConfig. Another gate works by config only.

This module does NOT import or modify main.py.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Configuration  (everything tunable lives here — no magic numbers in logic)  #
# --------------------------------------------------------------------------- #

GATE_ROLE = "gate"
INTERIOR_ROLE = "interior"


@dataclass
class AggregatorConfig:
    # camera name -> role
    camera_roles: Dict[str, str] = field(
        default_factory=lambda: {
            "front": GATE_ROLE,
            "right": GATE_ROLE,
            "left": GATE_ROLE,
            "back": INTERIOR_ROLE,
        }
    )

    # which OCR fields each role is allowed to fill
    gate_fields: List[str] = field(
        default_factory=lambda: [
            "container_no",
            "company_logo",
            "side_no",
            "license_plate",
        ]
    )
    interior_fields: List[str] = field(
        default_factory=lambda: ["truck_no", "truck_company"]
    )

    # main class (truck_with_container / truck_without_container) comes from gate only
    main_class_field: str = "main_class"
    main_class_role: str = GATE_ROLE

    # timing
    gate_sync_tolerance_sec: float = 3.0      # how far apart gate cams may fire
    back_attach_window_sec: float = 10.0      # back must appear within this after gate leaves
    camera_time_offset_sec: Dict[str, float] = field(default_factory=dict)  # align video clocks

    # back attachment refinement
    match_back_by_lane: bool = False          # set True if BACK emits a trustworthy lane_id

    # validation: fields that identify a physical truck
    identity_fields: List[str] = field(
        default_factory=lambda: ["container_no", "license_plate", "truck_no"]
    )

    def role_of(self, camera: str) -> Optional[str]:
        return self.camera_roles.get(camera.lower())

    def offset_of(self, camera: str) -> float:
        return self.camera_time_offset_sec.get(camera.lower(), 0.0)


# --------------------------------------------------------------------------- #
# Normalised record model                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class FieldReading:
    value: Any
    confidence: float
    camera: str


@dataclass
class TruckRecord:
    camera: str
    lane_id: Optional[str]
    track_id: Optional[str]           # ByteTrack random id — used only to dedupe one camera
    first_seen: float                 # offset-adjusted
    last_seen: float                  # offset-adjusted
    fields: Dict[str, FieldReading]   # field_name -> reading
    main_class: Optional[FieldReading] = None
    raw: Optional[dict] = None


def _coerce_reading(camera: str, value: Any, confidence: Any) -> Optional[FieldReading]:
    if value is None or value == "":
        return None
    try:
        conf = float(confidence) if confidence is not None else 0.0
    except (TypeError, ValueError):
        conf = 0.0
    return FieldReading(value=value, confidence=conf, camera=camera.upper())


def normalize_record(raw: dict, camera: str, cfg: AggregatorConfig) -> TruckRecord:
    """
    Turn one raw per-camera JSON record into a TruckRecord.

    Expected (flexible) shape per record:
        {
          "lane_id": "1",
          "track_id": 42,
          "first_seen_time_sec": 0.0,
          "last_seen_time_sec": 120.0,
          "associated_info": {
              "container_no":   {"value": "ABCD1234567", "confidence": 0.93},
              "license_plate":  {"value": "MU-AB-1234",  "confidence": 0.88},
              ...
          },
          "main_class": {"value": "truck_with_container", "confidence": 0.97}
        }
    A field may also be given flat as {"container_no": "ABCD...", "container_no_conf": 0.9}.
    """
    offset = cfg.offset_of(camera)
    first = float(raw.get("first_seen_time_sec", raw.get("first_seen", 0.0))) + offset
    last = float(raw.get("last_seen_time_sec", raw.get("last_seen", first))) + offset

    info = raw.get("associated_info", raw)
    fields: Dict[str, FieldReading] = {}
    for fname in cfg.gate_fields + cfg.interior_fields:
        if fname in info and isinstance(info[fname], dict):
            r = _coerce_reading(camera, info[fname].get("value"),
                                 info[fname].get("confidence"))
        else:
            r = _coerce_reading(camera, info.get(fname),
                                 info.get(f"{fname}_conf", info.get(f"{fname}_confidence")))
        if r is not None:
            fields[fname] = r

    mc_raw = raw.get(cfg.main_class_field)
    main_class = None
    if isinstance(mc_raw, dict):
        main_class = _coerce_reading(camera, mc_raw.get("value"), mc_raw.get("confidence"))
    elif mc_raw is not None:
        main_class = _coerce_reading(camera, mc_raw,
                                     raw.get(f"{cfg.main_class_field}_conf"))

    return TruckRecord(
        camera=camera.lower(),
        lane_id=(str(raw["lane_id"]) if raw.get("lane_id") is not None else None),
        track_id=(str(raw["track_id"]) if raw.get("track_id") is not None else None),
        first_seen=first,
        last_seen=last,
        fields=fields,
        main_class=main_class,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Gate event + consolidated truck                                             #
# --------------------------------------------------------------------------- #

@dataclass
class GateEvent:
    lane_id: Optional[str]
    gate_records: List[TruckRecord] = field(default_factory=list)
    back_records: List[TruckRecord] = field(default_factory=list)

    @property
    def first_seen(self) -> float:
        return min(r.first_seen for r in self.gate_records)

    @property
    def last_seen(self) -> float:
        return max(r.last_seen for r in self.gate_records)


# --------------------------------------------------------------------------- #
# Step 1 — gate grouping (lane-aware)                                         #
# --------------------------------------------------------------------------- #

def group_gate_events(gate_records: List[TruckRecord],
                      cfg: AggregatorConfig) -> List[GateEvent]:
    """
    Cluster FRONT/RIGHT/LEFT records into gate events.

    Rule: same lane_id AND concurrent/overlapping time window -> one event.
    Different lane_id -> never merged. This is the cross-lane fix.
    A time gap larger than the sync tolerance starts a new event (= next truck).
    """
    by_lane: Dict[Optional[str], List[TruckRecord]] = defaultdict(list)
    for r in gate_records:
        by_lane[r.lane_id].append(r)

    events: List[GateEvent] = []
    tol = cfg.gate_sync_tolerance_sec

    for lane, recs in by_lane.items():
        recs.sort(key=lambda r: r.first_seen)
        cluster: List[TruckRecord] = []
        cluster_end = None
        for r in recs:
            if cluster and r.first_seen <= cluster_end + tol:
                cluster.append(r)
                cluster_end = max(cluster_end, r.last_seen)
            else:
                if cluster:
                    events.append(GateEvent(lane_id=lane, gate_records=cluster))
                cluster = [r]
                cluster_end = r.last_seen
        if cluster:
            events.append(GateEvent(lane_id=lane, gate_records=cluster))

    events.sort(key=lambda e: e.first_seen)
    return events


# --------------------------------------------------------------------------- #
# Step 2 — back attachment (deferred window, most-recent wins)                #
# --------------------------------------------------------------------------- #

def attach_back_records(events: List[GateEvent],
                        back_records: List[TruckRecord],
                        cfg: AggregatorConfig) -> List[TruckRecord]:
    """
    Attach each BACK record to the most recent gate event whose last_seen is within
    the back-attach window before the back record's first_seen.

    Returns the list of orphan back records that matched no gate event.
    """
    orphans: List[TruckRecord] = []
    window = cfg.back_attach_window_sec

    for back in sorted(back_records, key=lambda r: r.first_seen):
        candidates = [
            e for e in events
            if 0.0 <= (back.first_seen - e.last_seen) <= window
        ]
        if cfg.match_back_by_lane and back.lane_id is not None:
            lane_matched = [e for e in candidates if e.lane_id == back.lane_id]
            if lane_matched:
                candidates = lane_matched
        if candidates:
            chosen = max(candidates, key=lambda e: e.last_seen)  # most recent
            chosen.back_records.append(back)
        else:
            orphans.append(back)

    return orphans


# --------------------------------------------------------------------------- #
# Step 3 — field merge (role-enforced, confidence + frequency)                #
# --------------------------------------------------------------------------- #

def _pick_best(records: List[TruckRecord], fname: str) -> Optional[FieldReading]:
    """Highest confidence wins; ties broken by most-frequent value, then confidence."""
    readings = [r.fields[fname] for r in records if fname in r.fields]
    if not readings:
        return None
    freq = Counter(rd.value for rd in readings)
    # sort: highest confidence first, then most frequent value, then value for stability
    readings.sort(key=lambda rd: (rd.confidence, freq[rd.value]), reverse=True)
    return readings[0]


def _pick_main_class(records: List[TruckRecord]) -> Optional[FieldReading]:
    cands = [r.main_class for r in records if r.main_class is not None]
    if not cands:
        return None
    freq = Counter(c.value for c in cands)
    cands.sort(key=lambda c: (c.confidence, freq[c.value]), reverse=True)
    return cands[0]


def merge_event(event: GateEvent, cfg: AggregatorConfig) -> Dict[str, FieldReading]:
    merged: Dict[str, FieldReading] = {}

    gate_recs = [r for r in event.gate_records if cfg.role_of(r.camera) == GATE_ROLE]
    back_recs = [r for r in event.back_records if cfg.role_of(r.camera) == INTERIOR_ROLE]

    # gate-role fields only from gate cameras
    for fname in cfg.gate_fields:
        best = _pick_best(gate_recs, fname)
        if best is not None:
            merged[fname] = best

    # interior-role fields only from back camera
    for fname in cfg.interior_fields:
        best = _pick_best(back_recs, fname)
        if best is not None:
            merged[fname] = best

    # main class from gate cameras only — BACK's main_class is deliberately ignored
    if cfg.main_class_role == GATE_ROLE:
        mc = _pick_main_class(gate_recs)
    else:
        mc = _pick_main_class(back_recs)
    if mc is not None:
        merged[cfg.main_class_field] = mc

    return merged


# --------------------------------------------------------------------------- #
# Step 4 — build consolidated rows                                            #
# --------------------------------------------------------------------------- #

def _consolidated_id(event: GateEvent) -> str:
    lane = event.lane_id if event.lane_id is not None else "x"
    return f"truck-L{lane}-{int(round(event.first_seen))}"


def _camera_source(event: GateEvent) -> str:
    cams = {r.camera.upper() for r in event.gate_records}
    cams |= {r.camera.upper() for r in event.back_records}
    order = ["FRONT", "RIGHT", "LEFT", "BACK"]
    ordered = [c for c in order if c in cams] + sorted(cams - set(order))
    return "+".join(ordered)


def build_rows(events: List[GateEvent], cfg: AggregatorConfig) -> List[dict]:
    rows = []
    for ev in events:
        merged = merge_event(ev, cfg)
        fields_out = {
            fname: {
                "value": rd.value,
                "confidence": round(rd.confidence, 4),
                "source": rd.camera,
            }
            for fname, rd in merged.items()
        }
        rows.append({
            "consolidated_id": _consolidated_id(ev),
            "lane_id": ev.lane_id,
            "camera_source": _camera_source(ev),
            "fields": fields_out,
            "timing": {
                "gate_first_seen_sec": round(ev.first_seen, 3),
                "gate_last_seen_sec": round(ev.last_seen, 3),
                "back_first_seen_sec": (round(min(b.first_seen for b in ev.back_records), 3)
                                        if ev.back_records else None),
            },
        })
    return rows


# --------------------------------------------------------------------------- #
# Step 5 — validation report                                                  #
# --------------------------------------------------------------------------- #

def build_validation_report(events: List[GateEvent],
                            rows: List[dict],
                            orphan_back: List[TruckRecord],
                            cfg: AggregatorConfig) -> dict:
    report: Dict[str, list] = {
        "same_identity_under_two_ids": [],   # (a)
        "conflicting_identities_one_id": [],  # (b)
        "orphan_back_records": [],
    }

    # (a) one identity value mapped to multiple consolidated_ids
    for fname in cfg.identity_fields:
        value_to_ids: Dict[Any, set] = defaultdict(set)
        for row in rows:
            f = row["fields"].get(fname)
            if f and f["value"] is not None:
                value_to_ids[f["value"]].add(row["consolidated_id"])
        for value, ids in value_to_ids.items():
            if len(ids) > 1:
                report["same_identity_under_two_ids"].append({
                    "field": fname, "value": value, "consolidated_ids": sorted(ids),
                })

    # (b) within one event, source records disagree on an identity field
    for ev in events:
        all_recs = ev.gate_records + ev.back_records
        for fname in cfg.identity_fields:
            distinct = {r.fields[fname].value for r in all_recs if fname in r.fields}
            if len(distinct) > 1:
                report["conflicting_identities_one_id"].append({
                    "consolidated_id": _consolidated_id(ev),
                    "field": fname,
                    "values": sorted(map(str, distinct)),
                })

    for b in orphan_back:
        report["orphan_back_records"].append({
            "camera": b.camera, "lane_id": b.lane_id,
            "first_seen_sec": round(b.first_seen, 3),
            "note": "no gate event within back-attach window",
        })

    report["is_clean"] = not any(report[k] for k in
                                 ("same_identity_under_two_ids",
                                  "conflicting_identities_one_id"))
    return report


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def aggregate(per_camera_results: Dict[str, List[dict]],
              cfg: Optional[AggregatorConfig] = None) -> dict:
    """
    per_camera_results: {"front": [...], "right": [...], "left": [...], "back": [...]}
    Each list element is one raw per-truck record from the single-camera engine.
    """
    cfg = cfg or AggregatorConfig()

    gate_records: List[TruckRecord] = []
    back_records: List[TruckRecord] = []

    for camera, raw_list in per_camera_results.items():
        role = cfg.role_of(camera)
        if role is None:
            continue  # unknown camera -> ignore (config-driven)
        for raw in (raw_list or []):
            rec = normalize_record(raw, camera, cfg)
            if role == GATE_ROLE:
                gate_records.append(rec)
            elif role == INTERIOR_ROLE:
                back_records.append(rec)

    events = group_gate_events(gate_records, cfg)
    orphan_back = attach_back_records(events, back_records, cfg)
    rows = build_rows(events, cfg)
    report = build_validation_report(events, rows, orphan_back, cfg)

    return {"trucks": rows, "validation": report, "truck_count": len(rows)}


# --------------------------------------------------------------------------- #
# FastAPI endpoint                                                            #
# --------------------------------------------------------------------------- #

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    router = APIRouter(prefix="/aggregator", tags=["aggregator"])

    # Hook the host app wires to its own job store. Returns the raw record list.
    JobLoader = Callable[[str], List[dict]]
    _job_loader: Optional[JobLoader] = None

    def set_job_loader(loader: JobLoader) -> None:
        """Register how job_ids are resolved to per-camera record lists."""
        global _job_loader
        _job_loader = loader

    class ConfigOverride(BaseModel):
        gate_sync_tolerance_sec: Optional[float] = None
        back_attach_window_sec: Optional[float] = None
        match_back_by_lane: Optional[bool] = None
        camera_time_offset_sec: Optional[Dict[str, float]] = None

    class AggregateRequest(BaseModel):
        # provide EITHER inline results ...
        results: Optional[Dict[str, List[dict]]] = Field(
            default=None,
            description='e.g. {"front":[...], "right":[...], "left":[...], "back":[...]}',
        )
        # ... OR job ids to be resolved by the registered loader
        job_ids: Optional[Dict[str, str]] = Field(
            default=None,
            description='e.g. {"front":"job_1","right":"job_2","left":"job_3","back":"job_4"}',
        )
        config: Optional[ConfigOverride] = None

    def _apply_overrides(cfg: AggregatorConfig, ov: Optional[ConfigOverride]) -> AggregatorConfig:
        if not ov:
            return cfg
        for k, v in ov.model_dump(exclude_none=True).items():
            setattr(cfg, k, v)
        return cfg

    @router.post("/aggregate")
    def aggregate_endpoint(req: AggregateRequest) -> dict:
        if req.results:
            per_camera = req.results
        elif req.job_ids:
            if _job_loader is None:
                raise HTTPException(
                    status_code=400,
                    detail="job_ids supplied but no job loader registered "
                           "(call set_job_loader at startup).",
                )
            per_camera = {cam: _job_loader(jid) for cam, jid in req.job_ids.items()}
        else:
            raise HTTPException(status_code=400,
                                detail="Provide either 'results' or 'job_ids'.")

        cfg = _apply_overrides(AggregatorConfig(), req.config)
        return aggregate(per_camera, cfg)

except ImportError:  # FastAPI not installed in this environment — module still importable
    router = None
