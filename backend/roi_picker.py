"""
ROI Lane Picker — click 4 corner points per lane to define polygon ROIs.
Output is saved as  lane_rois.json  next to the video file.

Usage
-----
    python roi_picker.py path/to/video.mp4

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

# ── state shared with the mouse callback ──────────────────────────────────────
_state: dict = {
    "frame": None,
    "all_lanes": [],       # list[list[tuple[int,int]]]  – completed lanes
    "current_pts": [],     # list[tuple[int,int]]        – lane being defined
    "current_lane": 0,
    "mouse_pos": (0, 0),
}
WIN = "ROI Lane Picker"


# ── rendering ─────────────────────────────────────────────────────────────────

def _render() -> np.ndarray:
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

def _save(video_path: str, frame_no: int, camera_name: str = "") -> None:
    lanes_to_save = list(_state["all_lanes"])
    if len(_state["current_pts"]) == 4:
        lanes_to_save.append(list(_state["current_pts"]))
    if not lanes_to_save:
        print("  Nothing to save — define at least one lane first.")
        return

    # Save as  {camera}_lane_rois.json  — local_inference.py looks for this first
    stem = f"{camera_name}_lane_rois" if camera_name else "lane_rois"
    out_path = Path(video_path).parent / f"{stem}.json"
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
    print("Click 4 corners to define Lane 1 ROI.\n")

    saved = False
    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):           # Q or Esc — quit
            print("[ROI Picker] Quit without saving.")
            break

        elif key == ord('r'):               # R — reset current lane
            _state["current_pts"].clear()
            print(f"  Reset Lane {_state['current_lane'] + 1}")
            cv2.imshow(WIN, _render())

        elif key == ord('n'):               # N — finish lane, start next
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
    args = parser.parse_args()
    main(args.video, camera_name=args.camera)
