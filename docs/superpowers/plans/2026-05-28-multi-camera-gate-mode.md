# Multi-Camera Gate Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gate-mode 4-camera fusion to `_merge_multi_camera_payloads` so all camera observations are always merged into one vehicle entity, with OCR fields preferring values from the camera angle that is configured as authoritative for each field.

**Architecture:** Two new env vars (`MULTI_CAMERA_GATE_MODE`, `MULTI_CAMERA_CAMERA_ROLES`) drive the change. Gate mode collapses all camera observations into one group unconditionally. Camera roles re-order observations before the existing `_better_field` tiebreaker so the authoritative camera wins on equal confidence. No new endpoints; no schema changes.

**Tech Stack:** Python 3.12, FastAPI, Pydantic Settings, pytest

---

## File Map

| File | Change |
|---|---|
| `backend/app/config.py` | Add 2 new `Settings` fields |
| `backend/.env.example` | Document the 2 new vars |
| `backend/app/main.py` | Add parser + 2 module constants; modify `_merge_multi_camera_payloads`; modify `_merge_truck_group` |
| `backend/tests/__init__.py` | Create (empty, marks package) |
| `backend/tests/conftest.py` | Set env vars before app import |
| `backend/tests/test_gate_mode.py` | Pytest tests for all new logic |

---

## Task 1: Config fields and env example

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/.env.example`

- [ ] **Step 1: Add two fields to `Settings`**

Open `backend/app/config.py`. After the existing `multi_camera_review_threshold` field (line 26), insert:

```python
    multi_camera_gate_mode: bool = False
    multi_camera_camera_roles: str = ""
```

Full updated `Settings` class body (lines 7–38):
```python
class Settings(BaseSettings):
    app_name: str = "PlateFlow API"
    app_env: str = "dev"
    database_url: str = "sqlite:///./plateflow.db"
    cors_origins: str = "http://localhost:5173"

    colab_infer_url: str | None = None
    colab_infer_path: str = "/infer"
    colab_analyze_video_path: str = "/analyze-video"
    colab_start_video_job_path: str = "/analyze-video/start"
    colab_video_job_status_path_template: str = "/analyze-video/jobs/{job_id}"
    colab_video_job_result_path_template: str = "/analyze-video/jobs/{job_id}/result"
    colab_video_job_frame_path_template: str = "/analyze-video/jobs/{job_id}/frame"
    inference_timeout_seconds: int = 45
    video_job_timeout_seconds: int = 600
    mock_inference_if_unavailable: bool = True
    multi_camera_order: str = "front,right,back,left"
    multi_camera_time_offsets_seconds: str = ""
    multi_camera_order_fallback: bool = True
    multi_camera_assume_single_entity: bool = False
    multi_camera_review_threshold: float = 0.75
    multi_camera_gate_mode: bool = False
    multi_camera_camera_roles: str = ""
    track_fragment_merge_gap_seconds: float = 20.0
    track_fragment_merge_aggressive: bool = True
```

- [ ] **Step 2: Document in `.env.example`**

Append to `backend/.env.example` after `MULTI_CAMERA_REVIEW_THRESHOLD=0.75`:

```
# Gate mode: assume exactly one vehicle passes the checkpoint at a time.
# All camera observations are merged into a single entity (confidence 0.95).
MULTI_CAMERA_GATE_MODE=false

# Camera role hints: authoritative fields per camera, used as tiebreaker when
# multiple cameras detect the same field at equal confidence.
# Format: camera:field1,field2;camera:field3
# Example for a 4-camera gate (front/back → plates, sides → container info):
# MULTI_CAMERA_CAMERA_ROLES=front:license_plate,truck_number,truck_company,driver;back:license_plate,truck_number;left:container_number,container_side_no,container_company_logo,other_container_info;right:container_number,container_side_no,container_company_logo,other_container_info
MULTI_CAMERA_CAMERA_ROLES=
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/config.py backend/.env.example
git commit -m "feat: add MULTI_CAMERA_GATE_MODE and MULTI_CAMERA_CAMERA_ROLES config fields"
```

---

## Task 2: Parser function and module-level constants in main.py

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: Add `_parse_camera_roles` function**

In `backend/app/main.py`, insert the following function directly after `_parse_camera_time_offsets` (currently at line 72). Place it at approximately line 84 (after the closing brace of `_parse_camera_time_offsets` and before `CAMERA_TIME_OFFSETS_SECONDS`):

```python
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
```

- [ ] **Step 2: Add the two module-level constants**

In the block of module-level constants (currently lines 65–69), add two lines after `TRACK_FRAGMENT_MERGE_AGGRESSIVE`:

```python
MULTI_CAMERA_GATE_MODE = settings.multi_camera_gate_mode
MULTI_CAMERA_CAMERA_ROLES: dict[str, set[str]] = _parse_camera_roles(settings.multi_camera_camera_roles)
```

The full block should now read:

```python
MULTI_CAMERA_ASSUME_SINGLE_ENTITY = settings.multi_camera_assume_single_entity
MULTI_CAMERA_ORDER_FALLBACK = settings.multi_camera_order_fallback
MULTI_CAMERA_REVIEW_THRESHOLD = settings.multi_camera_review_threshold
TRACK_FRAGMENT_MERGE_GAP_SECONDS = settings.track_fragment_merge_gap_seconds
TRACK_FRAGMENT_MERGE_AGGRESSIVE = settings.track_fragment_merge_aggressive
MULTI_CAMERA_GATE_MODE = settings.multi_camera_gate_mode
MULTI_CAMERA_CAMERA_ROLES: dict[str, set[str]] = _parse_camera_roles(settings.multi_camera_camera_roles)
```

Note: `_parse_camera_roles` must be defined **before** this block. Move the function insertion point to just before the existing `CAMERA_TIME_OFFSETS_SECONDS = _parse_camera_time_offsets(...)` call if needed, or place it immediately after `_parse_camera_time_offsets`.

- [ ] **Step 3: Verify the app still starts (smoke test)**

```bash
cd backend
.venv/bin/python -c "from app.main import MULTI_CAMERA_GATE_MODE, MULTI_CAMERA_CAMERA_ROLES; print(MULTI_CAMERA_GATE_MODE, MULTI_CAMERA_CAMERA_ROLES)"
```

Expected output:
```
False {}
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: parse MULTI_CAMERA_GATE_MODE and MULTI_CAMERA_CAMERA_ROLES at startup"
```

---

## Task 3: Gate-mode grouping in `_merge_multi_camera_payloads`

**Files:**
- Modify: `backend/app/main.py` — `_merge_multi_camera_payloads` (currently lines 831–959)

- [ ] **Step 1: Replace the grouping strategy block**

Locate the `group_specs` assignment block starting at line 889 (the `if MULTI_CAMERA_ASSUME_SINGLE_ENTITY` block). Replace the entire `if / else` grouping strategy section with the version below. The change adds a new top-level `if MULTI_CAMERA_GATE_MODE` branch **before** the existing logic:

```python
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
```

- [ ] **Step 2: Pass `MULTI_CAMERA_CAMERA_ROLES` to `_merge_truck_group`**

The line that calls `_merge_truck_group` (currently line 937–938):
```python
    trucks_out: dict[str, dict[str, Any]] = {}
    for index, (group, match_method, match_confidence) in enumerate(group_specs, start=1):
        trucks_out[str(index)] = _merge_truck_group(index, group, match_method, match_confidence)
```

Change to:
```python
    trucks_out: dict[str, dict[str, Any]] = {}
    _roles = MULTI_CAMERA_CAMERA_ROLES if MULTI_CAMERA_CAMERA_ROLES else None
    for index, (group, match_method, match_confidence) in enumerate(group_specs, start=1):
        trucks_out[str(index)] = _merge_truck_group(index, group, match_method, match_confidence, _roles)
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: gate_mode path in _merge_multi_camera_payloads — always merge into one entity"
```

---

## Task 4: Camera-role-aware field selection in `_merge_truck_group`

**Files:**
- Modify: `backend/app/main.py` — `_merge_truck_group` (currently lines 409–516)

- [ ] **Step 1: Add `camera_roles` parameter to the signature**

Change the function signature from:
```python
def _merge_truck_group(
    entity_id: int,
    observations: list[dict[str, Any]],
    match_method: str | None = None,
    match_confidence: float | None = None,
) -> dict[str, Any]:
```
to:
```python
def _merge_truck_group(
    entity_id: int,
    observations: list[dict[str, Any]],
    match_method: str | None = None,
    match_confidence: float | None = None,
    camera_roles: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
```

- [ ] **Step 2: Separate the observation loop from field merging**

Currently lines 428–460 contain a single `for obs in observations:` loop that collects stats AND merges fields. Split it into two passes.

Replace the loop:
```python
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

        for field in OCR_FIELD_KEYS:
            merged_info[field] = _better_field(merged_info.get(field), info.get(field))

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
```

with:
```python
    for obs in observations:
        truck = obs["truck"]
        camera = obs["camera"]
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
```

- [ ] **Step 3: Record `camera_roles_applied` in `_fusion` metadata**

In the `_fusion` dict (currently lines 477–493), add one key after `"identity_keys"`:

```python
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
```

- [ ] **Step 4: Smoke test**

```bash
cd backend
.venv/bin/python -c "
from app.main import _merge_truck_group
obs = [
    {'camera': 'front', 'truck': {'track_id': 1, 'type': 'truck_without_container', 'associated_info': {'license_plate': {'text': 'ABC123', 'confidence': 0.7}}, 'confidence_avg': 0.9, 'first_seen_frame': 0, 'last_seen_frame': 10, 'first_seen_time_sec': 0.0, 'last_seen_time_sec': 0.33, 'time_offset_sec': 0.0}},
    {'camera': 'right', 'truck': {'track_id': 2, 'type': 'truck_without_container', 'associated_info': {'license_plate': {'text': 'WRONGPLATE', 'confidence': 0.7}}, 'confidence_avg': 0.85, 'first_seen_frame': 0, 'last_seen_frame': 10, 'first_seen_time_sec': 0.0, 'last_seen_time_sec': 0.33, 'time_offset_sec': 0.0}},
]
roles = {'front': {'license_plate', 'truck_number'}, 'right': {'container_number', 'container_side_no'}}
result = _merge_truck_group(1, obs, 'gate_mode', 0.95, roles)
plate = result['associated_info']['license_plate']
print('license_plate:', plate)
assert plate['text'] == 'ABC123', f'Expected ABC123, got {plate[\"text\"]}'
print('PASS: front-camera license_plate wins tiebreak')
"
```

Expected output:
```
license_plate: {'text': 'ABC123', 'confidence': 0.7}
PASS: front-camera license_plate wins tiebreak
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: camera-role-aware field selection in _merge_truck_group"
```

---

## Task 5: Tests

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_gate_mode.py`

- [ ] **Step 1: Install pytest**

```bash
cd backend
.venv/bin/pip install pytest
```

Expected: `Successfully installed pytest-X.Y.Z`

- [ ] **Step 2: Create `tests/__init__.py`**

Create `backend/tests/__init__.py` as an empty file.

- [ ] **Step 3: Create `backend/tests/conftest.py`**

```python
import os

# Set env vars before the app module is imported so Settings reads them.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_plateflow.db")
os.environ.setdefault("MOCK_INFERENCE_IF_UNAVAILABLE", "true")
os.environ.setdefault("COLAB_INFER_URL", "")
os.environ.setdefault("MULTI_CAMERA_GATE_MODE", "false")
os.environ.setdefault("MULTI_CAMERA_CAMERA_ROLES", "")
```

- [ ] **Step 4: Write the failing tests**

Create `backend/tests/test_gate_mode.py`:

```python
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
    """Gate mode + camera roles: license_plate comes from front, container_number from left."""
    roles = {
        "front": {"license_plate", "truck_number"},
        "left": {"container_number", "container_side_no"},
    }
    monkeypatch.setattr(main_module, "MULTI_CAMERA_GATE_MODE", True)
    monkeypatch.setattr(main_module, "MULTI_CAMERA_CAMERA_ROLES", roles)

    payloads = {
        "front": _camera_payload(1, "truck_with_container", {
            "license_plate": "FRONT_PLATE",
            "container_number": "FRONT_WRONG",
        }),
        "left": _camera_payload(2, "truck_with_container", {
            "license_plate": "LEFT_WRONG",
            "container_number": "MSCU1234560",
        }),
    }
    result = _merge_multi_camera_payloads(payloads)
    info = result["trucks"]["1"]["associated_info"]
    assert info["license_plate"]["text"] == "FRONT_PLATE"
    assert info["container_number"]["text"] == "MSCU1234560"
```

- [ ] **Step 5: Run tests — verify they fail before implementation is complete**

```bash
cd backend
.venv/bin/python -m pytest tests/test_gate_mode.py -v 2>&1 | head -40
```

At this point (after Tasks 1–4 are done) all tests should PASS. If running before Tasks 3–4, expect failures on `_merge_truck_group` and `_merge_multi_camera_payloads` tests.

- [ ] **Step 6: Run full test suite**

```bash
cd backend
.venv/bin/python -m pytest tests/test_gate_mode.py -v
```

Expected output (all 14 tests):
```
tests/test_gate_mode.py::test_parse_camera_roles_empty PASSED
tests/test_gate_mode.py::test_parse_camera_roles_single PASSED
tests/test_gate_mode.py::test_parse_camera_roles_multiple PASSED
tests/test_gate_mode.py::test_parse_camera_roles_ignores_empty_parts PASSED
tests/test_gate_mode.py::test_parse_camera_roles_strips_whitespace PASSED
tests/test_gate_mode.py::test_camera_roles_auth_wins_tiebreak PASSED
tests/test_gate_mode.py::test_camera_roles_higher_confidence_still_wins PASSED
tests/test_gate_mode.py::test_camera_roles_no_roles_unchanged PASSED
tests/test_gate_mode.py::test_camera_roles_written_to_fusion_metadata PASSED
tests/test_gate_mode.py::test_no_camera_roles_fusion_metadata_false PASSED
tests/test_gate_mode.py::test_gate_mode_merges_all_cameras_into_one PASSED
tests/test_gate_mode.py::test_gate_mode_confidence_is_095 PASSED
tests/test_gate_mode.py::test_gate_mode_false_does_not_force_merge PASSED
tests/test_gate_mode.py::test_gate_mode_with_camera_roles_fields_routed_correctly PASSED

14 passed in X.XXs
```

- [ ] **Step 7: Commit**

```bash
git add backend/tests/ backend/pyproject.toml
git commit -m "test: add pytest suite for gate_mode and camera_roles merging"
```

---

## Activation

To enable gate mode for a 4-camera gate checkpoint, set these in `backend/.env`:

```
MULTI_CAMERA_GATE_MODE=true
MULTI_CAMERA_CAMERA_ROLES=front:license_plate,truck_number,truck_company,driver;back:license_plate,truck_number;left:container_number,container_side_no,container_company_logo,other_container_info;right:container_number,container_side_no,container_company_logo,other_container_info
```

No code restart is needed if the app reads `.env` at startup via `pydantic-settings`.
