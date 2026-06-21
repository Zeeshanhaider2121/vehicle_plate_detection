# Frontend Lane-ROI + Gate-Line Drawing — Design

Date: 2026-06-22
Status: in progress (foundation being built)

## Goal

Let the operator draw lane separation **and a physical gate line** directly in the
browser, on real frames from the front/left/right camera videos, test the result,
and — once happy — save it so every future job reuses it. This produces the
per-camera `lane_id` that the lane-aware merge / aggregator needs to stop cross-lane
field bleed.

## Confirmed decisions

- **Draw once, reuse.** One-time setup per camera position; redraw only if a camera
  physically moves. (Operator wants to test first, then save permanently.)
- **Frame source:** a representative still frame extracted from the uploaded camera
  videos (front/left/right). Same geometry as the processed video.
- **Test before save:** (c) quick overlay to iterate polygon shape, then one full run
  to confirm clean detections, then Save.
- **Lanes:** fixed at 2 (Lane 1 / Lane 2) per camera in the UI; back camera excluded.
  Lane count kept in config, not hardcoded, so another gate can differ.
- **Integration:** Approach 2 — inline in the multi-camera upload flow.
- **Gate line semantics (default, changeable):** (b) tripwire. A truck counts once its
  box crosses the line; the crossing moment locks its ID and is the canonical sync
  timestamp used to align the 3 gate cameras and to attach the back record.

## The work is THREE independent subsystems

Decomposed deliberately so each is testable and low-risk, and so engine-touching work
is isolated from UI work.

### Subsystem 1 — Drawing tool (UI + storage)  ← built first
- **Backend** `app/lane_setup.py` (separate router, engine + main.py merge untouched,
  wired with one `include_router` line like the aggregator):
  - `POST /api/lane-setup/extract-frame` — multipart video in → representative JPEG out.
  - `POST /api/lane-setup/save` — `{camera, image_width, image_height, lanes:{ "1":[[x,y]*4], "2":[...] }, gate_line:[[x,y],[x,y]] }`
    → writes `{camera}_lane_rois.json` in `backend/` (the dir the engine reads), in the
    engine's existing format plus a `gate_line` key (engine ignores unknown keys today).
  - `GET /api/lane-setup/{camera}` — return saved ROI for re-editing.
  - Coordinates are stored in **native image pixels**; the frontend maps display→native.
- **Frontend** inline SVG-canvas drawing step in `App.jsx`: per camera, draw 2 lane
  polygons + 1 gate line, with undo/reset, then Save.

### Subsystem 2 — Gate-line crossing logic (engine)  ← DONE 2026-06-22
- `local_inference.py`: `_load_gate_line(camera)` reads `gate_line` from the ROI file;
  `_gate_side(bbox, line)` gives which side the truck's bottom-centre is on; the
  `TruckRegistry` stamps `gate_crossed`/`gate_cross_frame`/`gate_cross_time_sec` the
  first time a track flips sides. Internal `_gate_prev_side` is stripped from output by
  the existing `_`-prefix cleaner. Additive + opt-in via presence of a gate line.
- NOT YET: dropping trucks that never cross (behaviour-changing; left as a follow-up
  flag), and feeding `gate_cross_time_sec` into back-attach as the canonical sync point
  (belongs to Subsystem 3).
- Verified: geometry + crossing unit-checked; downward-crossing truck stamped, a
  non-crosser stays False; aggregator tests still 4/4 green.

### Subsystem 3 — Synchronized parallel processing  ← ALREADY EXISTED + bridge added
- **Already implemented:** each camera child job runs in its own daemon thread
  (`video_client.py:41`), so front/right/left/back process concurrently. Time-zone sync
  exists in `_derive_camera_sessions` (main.py): front is the reference clock and
  `offsets[camera] = (start_dt - reference)` aligns all cameras (the `[CAMERA-SYNC]`
  log line). No rebuild needed.
- **Bridge added (Subsystem 2 -> 3):** the aggregator can anchor the deferred
  back-attach on the physical gate-crossing time instead of file clocks.
  `AggregatorConfig.back_attach_anchor` = `"last_seen"` (default, unchanged) or
  `"gate_cross"`; `TruckRecord.gate_cross_time` is read from `gate_cross_time_sec`.
  Verified by `test_back_attach_anchor_uses_gate_crossing` (5/5 aggregator tests green).

## Quick-test (overlay) note

The "quick overlay" test runs truck-box detection on a sampled frame and colours each
box by which lane polygon its bottom-centre falls in (reusing `_get_detection_lane`).
This reuses the YOLO model, so it lands with Subsystem 1's frontend step but is
verified against the real engine call — not guessed.

## Out of scope (YAGNI)

- Variable lane counts in the UI (config supports it; UI targets 2).
- Editing ROIs of past/finished jobs.
- Any change to the single-camera engine's detection logic in Subsystem 1.
