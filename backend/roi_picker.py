"""
ROI Lane Picker — click 4 corner points per lane to define polygon ROIs.

Output is saved as  {camera}_lane_rois.json  in the backend/ directory (next to
local_inference.py), which is the ONLY place the engine's _load_lane_rois() looks.
Run it once per camera that sees more than one lane (typically right and left).

Usage
-----
    python roi_picker.py path/to/right_cam.mp4 --camera right
    python roi_picker.py path/to/left_cam.mp4  --camera left

The --camera name MUST match the camera name the inference pipeline uses, or the
engine won't pair the ROI file with the job.

Controls
--------
    Left-click   add a point (polygon closes automatically at 4th point)
    Right-click  undo last point
    Space        advance 30 frames
    B            go back 30 frames
    R            reset current lane
    N            finish current lane, start next lane
    S            save all lanes and exit
    Q / Esc      quit without saving
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_LANES = 2
# BGR colours — one per lane
LANE_COLORS = [(0, 240, 80), (80, 100, 255)]
GATE_COLOR = (60, 200, 255)   # amber — the gate tripwire
ROI_COLOR = (221, 138, 55)    # blue — the single ROI rectangle

# Set True by --gate: pick a 2-point gate tripwire instead of lane polygons.
GATE_MODE = False
# Set True by --roi: pick a single 2-corner rectangular ROI (the new default flow).
ROI_MODE = False

# ── state shared with the mouse callback ──────────────────────────────────────
_state: dict = {
    "frame": None,
    "all_lanes": [],       # list[list[tuple[int,int]]]  – completed lanes
    "current_pts": [],     # list[tuple[int,int]]        – lane being defined
    "current_lane": 0,
    "gate_pts": [],        # list[tuple[int,int]]        – the 2 gate-line endpoints
    "roi_pts": [],         # list[tuple[int,int]]        – the 2 ROI-rectangle corners
    "mouse_pos": (0, 0),
}
WIN = "ROI Lane Picker"


# ── rendering ─────────────────────────────────────────────────────────────────

def _render_gate() -> np.ndarray:
    """Render the gate-line picker: 2 endpoints + the tripwire between them."""
    vis = _state["frame"].copy()
    pts = _state["gate_pts"]
    mx, my = _state["mouse_pos"]
    for i, p in enumerate(pts):
        cv2.circle(vis, p, 7, GATE_COLOR, -1, cv2.LINE_AA)
        cv2.putText(vis, str(i + 1), (p[0] + 9, p[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GATE_COLOR, 2, cv2.LINE_AA)
    if len(pts) == 2:
        cv2.line(vis, pts[0], pts[1], GATE_COLOR, 3, cv2.LINE_AA)
    elif len(pts) == 1:
        cv2.line(vis, pts[0], (mx, my), GATE_COLOR, 1, cv2.LINE_AA)
    lines = [
        f"Drawing GATE LINE  —  {len(pts)}/2 points",
        "LClick=add  RClick=undo  Space=+30fr  B=-30fr  R=reset  S=save  Q=quit",
    ]
    for i, txt in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _render_roi() -> np.ndarray:
    """Render the single-ROI picker: 2 opposite corners + the rectangle between them."""
    vis = _state["frame"].copy()
    pts = _state["roi_pts"]
    mx, my = _state["mouse_pos"]
    p2 = pts[1] if len(pts) == 2 else ((mx, my) if len(pts) == 1 else None)
    if len(pts) >= 1 and p2 is not None:
        x1, y1 = pts[0]
        x2, y2 = p2
        cv2.rectangle(vis, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)),
                      ROI_COLOR, 2, cv2.LINE_AA)
        cv2.putText(vis, "ROI", (min(x1, x2) + 6, min(y1, y2) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(vis, p, 7, ROI_COLOR, -1, cv2.LINE_AA)
        cv2.putText(vis, str(i + 1), (p[0] + 9, p[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 2, cv2.LINE_AA)
    lines = [
        f"Drawing ROI RECTANGLE  —  {len(pts)}/2 corners",
        "LClick=add corner  RClick=undo  Space=+30fr  B=-30fr  R=reset  S=save  Q=quit",
    ]
    for i, txt in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _render() -> np.ndarray:
    if ROI_MODE:
        return _render_roi()
    if GATE_MODE:
        return _render_gate()
    vis = _state["frame"].copy()
    all_lanes: list = _state["all_lanes"]
    current_pts: list = _state["current_pts"]
    current_lane: int = _state["current_lane"]
    mx, my = _state["mouse_pos"]

    # draw completed lanes
    for idx, pts in enumerate(all_lanes):
        col = LANE_COLORS[idx % len(LANE_COLORS)]
        arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [arr], True, col, 2, cv2.LINE_AA)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [arr], col)
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
        for p in pts:
            cv2.circle(vis, p, 6, col, -1, cv2.LINE_AA)
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        cv2.putText(vis, f"Lane {idx + 1}", (cx - 22, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, f"Lane {idx + 1}", (cx - 22, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)

    # draw in-progress points + lines
    col = LANE_COLORS[current_lane % len(LANE_COLORS)]
    for i, p in enumerate(current_pts):
        cv2.circle(vis, p, 6, col, -1, cv2.LINE_AA)
        if i > 0:
            cv2.line(vis, current_pts[i - 1], p, col, 2, cv2.LINE_AA)
        cv2.putText(vis, str(i + 1), (p[0] + 8, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    # preview line from last point to mouse cursor
    if current_pts and len(current_pts) < 4:
        cv2.line(vis, current_pts[-1], (mx, my), col, 1, cv2.LINE_AA)

    # HUD
    lines = [
        f"Defining Lane {current_lane + 1}  —  {len(current_pts)}/4 points",
        "LClick=add  RClick=undo  Space=+30fr  B=-30fr  R=reset  N=next lane  S=save  Q=quit",
    ]
    for i, txt in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ── mouse callback ─────────────────────────────────────────────────────────────

def _on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
    _state["mouse_pos"] = (x, y)

    if ROI_MODE:
        pts = _state["roi_pts"]
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) >= 2:
                pts.clear()  # third click starts a fresh rectangle
            pts.append((x, y))
            print(f"  [ROI] Corner {len(pts)}: ({x}, {y})")
            if len(pts) == 2:
                print("  ROI rectangle complete — press S to save, R to redo.")
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            print(f"  Removed ROI corner {pts.pop()}")
        cv2.imshow(WIN, _render())
        return

    if GATE_MODE:
        pts = _state["gate_pts"]
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            print(f"  [Gate] Point {len(pts)}: ({x}, {y})")
            if len(pts) == 2:
                print("  Gate line complete — press S to save, R to redo.")
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            print(f"  Removed gate point {pts.pop()}")
        cv2.imshow(WIN, _render())
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        pts = _state["current_pts"]
        if len(pts) < 4:
            pts.append((x, y))
            print(f"  [Lane {_state['current_lane'] + 1}] Point {len(pts)}: ({x}, {y})")
            if len(pts) == 4:
                print(f"  Lane {_state['current_lane'] + 1} polygon complete — press N for next lane or S to save.")
        cv2.imshow(WIN, _render())

    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = _state["current_pts"]
        if pts:
            removed = pts.pop()
            print(f"  Removed point {removed}")
        cv2.imshow(WIN, _render())

    elif event == cv2.EVENT_MOUSEMOVE:
        cv2.imshow(WIN, _render())


# ── save helper ───────────────────────────────────────────────────────────────

def _save_gate(video_path: str, frame_no: int, camera_name: str = "") -> None:
    pts = _state["gate_pts"]
    if len(pts) != 2:
        print(f"  Need 2 points for the gate line — have {len(pts)}.")
        return
    # Engine's _load_gate_line() reads {camera}_gate_line.json (then *_lane_rois.json)
    # from the directory local_inference.py lives in (backend/). roi_picker.py is in
    # that same dir, so write there.
    stem = f"{camera_name}_gate_line" if camera_name else "gate_line"
    out_path = Path(__file__).resolve().parent / f"{stem}.json"
    payload = {
        "video": str(video_path),
        "camera": camera_name,
        "frame": frame_no,
        "gate_line": [list(pts[0]), list(pts[1])],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[ROI Picker] Saved gate line → {out_path}")
    print(f"  gate_line = {payload['gate_line']}\n")


def _save_roi(video_path: str, frame_no: int, camera_name: str = "") -> None:
    pts = _state["roi_pts"]
    if len(pts) != 2:
        print(f"  Need 2 corners for the ROI rectangle — have {len(pts)}.")
        return
    # Engine's _load_roi() reads the `roi` key of {camera}_lane_rois.json from the
    # directory local_inference.py lives in (backend/); write there.
    stem = f"{camera_name}_lane_rois" if camera_name else "lane_rois"
    out_path = Path(__file__).resolve().parent / f"{stem}.json"
    # Preserve any existing lanes/gate_line already saved for this camera.
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
    h, w = _state["frame"].shape[:2]
    (x1, y1), (x2, y2) = pts
    payload = {
        **existing,
        "video": str(video_path),
        "camera": camera_name,
        "frame": frame_no,
        "image_width": int(w),
        "image_height": int(h),
        "roi": [[min(x1, x2), min(y1, y2)], [max(x1, x2), max(y1, y2)]],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[ROI Picker] Saved ROI → {out_path}")
    print(f"  roi = {payload['roi']}  ({w}x{h})\n")


def _save(video_path: str, frame_no: int, camera_name: str = "") -> None:
    if ROI_MODE:
        _save_roi(video_path, frame_no, camera_name)
        return
    if GATE_MODE:
        _save_gate(video_path, frame_no, camera_name)
        return
    lanes_to_save = list(_state["all_lanes"])
    if len(_state["current_pts"]) == 4:
        lanes_to_save.append(list(_state["current_pts"]))
    if not lanes_to_save:
        print("  Nothing to save — define at least one lane first.")
        return

    # Save as  {camera}_lane_rois.json  — local_inference.py looks for this first.
    # IMPORTANT: the engine's _load_lane_rois() only searches the directory that
    # local_inference.py lives in (backend/), NOT next to the video. roi_picker.py
    # sits in that same backend/ directory, so write there or the engine never finds it.
    stem = f"{camera_name}_lane_rois" if camera_name else "lane_rois"
    out_path = Path(__file__).resolve().parent / f"{stem}.json"
    payload = {
        "video": str(video_path),
        "camera": camera_name,
        "frame": frame_no,
        "lanes": {str(i + 1): pts for i, pts in enumerate(lanes_to_save)},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[ROI Picker] Saved → {out_path}\n")

    # Print ready-to-paste Python snippet
    print("# ── copy this into local_inference.py ──────────────────────────")
    print("LANE_ROIS = {")
    for i, pts in enumerate(lanes_to_save):
        print(f"    {i + 1}: np.array({pts}, np.int32),")
    print("}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main(video_path: str, camera_name: str = "") -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        sys.exit(1)

    _state["frame"] = frame
    frame_no = 0

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    cv2.resizeWindow(WIN, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(WIN, _on_mouse)
    cv2.imshow(WIN, _render())

    cam_label = f"  camera='{camera_name}'" if camera_name else ""
    print(f"\n[ROI Picker] Opened: {video_path}  ({w}x{h}, {total} frames){cam_label}")
    if ROI_MODE:
        print("ROI MODE: click 2 opposite corners to draw the rectangular ROI, then S to save.\n")
    elif GATE_MODE:
        print("GATE MODE: click 2 points across the lane to draw the tripwire, then S to save.\n")
    else:
        print("Click 4 corners to define Lane 1 ROI.\n")

    saved = False
    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):           # Q or Esc — quit
            print("[ROI Picker] Quit without saving.")
            break

        elif key == ord('r'):               # R — reset current lane / gate line / ROI
            if ROI_MODE:
                _state["roi_pts"].clear()
                print("  Reset ROI rectangle")
            elif GATE_MODE:
                _state["gate_pts"].clear()
                print("  Reset gate line")
            else:
                _state["current_pts"].clear()
                print(f"  Reset Lane {_state['current_lane'] + 1}")
            cv2.imshow(WIN, _render())

        elif key == ord('n') and not GATE_MODE and not ROI_MODE:   # N — finish lane, start next (lane mode only)
            pts = _state["current_pts"]
            if len(pts) == 4:
                _state["all_lanes"].append(list(pts))
                _state["current_pts"] = []
                _state["current_lane"] += 1
                print(f"  Lane {len(_state['all_lanes'])} saved. Now defining Lane {_state['current_lane'] + 1}.")
                cv2.imshow(WIN, _render())
            else:
                print(f"  Need 4 points — only have {len(pts)}.")

        elif key == ord('s'):               # S — save and exit
            _save(video_path, frame_no, camera_name)
            saved = True
            break

        elif key == ord(' '):               # Space — advance 30 frames
            advanced = 0
            for _ in range(30):
                ret, f = cap.read()
                if not ret:
                    break
                _state["frame"] = f
                frame_no += 1
                advanced += 1
            print(f"  Frame {frame_no}/{total}  (+{advanced})")
            cv2.imshow(WIN, _render())

        elif key == ord('b'):               # B — go back 30 frames
            target = max(0, frame_no - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, f = cap.read()
            if ret:
                _state["frame"] = f
                frame_no = target
            print(f"  Frame {frame_no}/{total}")
            cv2.imshow(WIN, _render())

    cap.release()
    cv2.destroyAllWindows()
    if not saved:
        print("[ROI Picker] Exited without saving.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Click lane ROI polygons on a video frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--camera", "-c", default="",
        help="Camera name (e.g. front, right, left). "
             "Saves as  {camera}_lane_rois.json  so local_inference.py loads it per camera.",
    )
    parser.add_argument(
        "--gate", action="store_true",
        help="Gate-line mode: click 2 points to draw the tripwire (saved as "
             "{camera}_gate_line.json) instead of 4-point lane polygons.",
    )
    parser.add_argument(
        "--roi", action="store_true",
        help="Single-ROI mode: click 2 opposite corners to draw one rectangular ROI "
             "(saved as the `roi` key of {camera}_lane_rois.json). This is the new, "
             "generic separation region the engine reads via _load_roi.",
    )
    args = parser.parse_args()
    GATE_MODE = args.gate
    ROI_MODE = args.roi
    main(args.video, camera_name=args.camera)
