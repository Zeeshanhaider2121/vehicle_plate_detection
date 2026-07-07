"""
Supervisor camera — the controller over the front trio (front / left / right).

This is the single authority that owns *synchronized* processing of the three
front-facing cameras. It:

  * steps the three cameras in lockstep by frame / fps (with a per-camera clock
    offset for recordings that do not start at the same instant),
  * enforces the DISTINCT-COORDINATES rule at the per-frame level (two same-class
    detections in one camera at one frame that do not spatially overlap are
    different trucks and can never share a consolidated id — the "id-68" bug),
  * decides when a truck is CONFIRMED (front-trio agreement: the same class active
    in the same frame window across enough front cameras), and
  * emits the confirmed trucks to the consolidation layer (``app.main``), which
    attaches the back camera afterward. The back camera is deliberately NOT part
    of the supervisor.

It also owns a DEDICATED log file (a named ``"supervisor"`` logger with its own
FileHandler, separate from the app's main log), where it records CONFIRMED /
REJECTED / SPLIT decisions with frame ids and timestamps. The back-attach step in
``app.main`` writes its outcome to the same logger so the whole gate decision is in
one place.

Everything is config-driven (see ``SupervisorConfig`` / ``app.config``): no
hardcoded camera counts, ids, or frame numbers.

Per-frame input shape (one row per kept truck detection per frame), produced by the
single-camera engine (``local_inference.py`` -> job result ``"frame_tracks"``):

    {
      "frame_id": 123,
      "track_id": 68,          # RAW ByteTrack id (for spatial-jump re-splits)
      "perm_id": 2,            # engine's permanent id (== final record track_id)
      "class_name": "truck_with_container",
      "confidence": 0.97,
      "bbox": {"x1": .., "y1": .., "x2": .., "y2": ..},
      "crop_path": "…/f000123_i000_t68_conf0.970.jpg"  # or null
    }
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TRUCK_CLASSES = {"truck_with_container", "truck_without_container"}


@dataclass
class SupervisorConfig:
    # Front-facing cameras the supervisor owns, in preferred order.
    front_cameras: List[str] = field(default_factory=lambda: ["front", "left", "right"])
    # Frame tolerance for "active in the same frame window" (SYNC_WINDOW_FRAMES).
    sync_window_frames: int = 15
    # Per-camera clock offset in FRAMES; a camera's common-timeline frame =
    # local frame_id (normalised to the reference fps) + this offset.
    clock_offsets_frames: Dict[str, int] = field(default_factory=dict)
    # Two same-class boxes are DIFFERENT trucks when their IoU is below this AND their
    # centres are farther apart than center_dist_frac of the frame (either axis). Using
    # AND (not OR) protects the "never split one truck across ids" invariant: a single
    # moving truck keeps high frame-to-frame IoU, so it is never cut; genuinely
    # side-by-side trucks have both low IoU and distant centres, so they always split.
    iou_same_object: float = 0.3
    center_dist_frac: float = 0.15
    # A truck is CONFIRMED only with at least this many DISTINCT front cameras agreeing.
    # 0 -> require ALL present front cameras (all three, when front/left/right present).
    min_confirm_cameras: int = 0
    # Back camera read window after a truck's front-trio exit (seconds). Owned here so
    # the consolidation step and the supervisor log agree on the window.
    back_window_s: float = 10.0
    # Whether a corroborated burst that shows trucks side-by-side is split into one
    # confirmed truck per horizontal lane (the distinct-coordinates rule at the
    # cross-camera level). Spatial separation always overrides a timing merge.
    side_by_side_split: bool = True
    # Fallback frame size used to scale center_dist_frac when a camera's real frame size
    # cannot be inferred from its detection boxes.
    default_frame_width: int = 1280
    default_frame_height: int = 720
    # Logging
    log_path: str = "plateflow_outputs/logs/supervisor.log"
    log_level: str = "INFO"


def supervisor_config_from_settings(settings: Any) -> SupervisorConfig:
    """Build a SupervisorConfig from the app Settings object (app.config.settings).

    Falls back to sensible defaults for any missing attribute so the supervisor is
    usable even against a partial settings object (e.g. in tests)."""
    def _get(name: str, default: Any) -> Any:
        return getattr(settings, name, default)

    front_raw = str(_get("supervisor_front_cameras", "") or "").strip()
    if not front_raw:
        front_raw = str(_get("front_facing_cameras", "front,right,left") or "")
    front = [c.strip().lower() for c in front_raw.split(",") if c.strip()]

    offsets: Dict[str, int] = {}
    for part in str(_get("supervisor_clock_offsets_frames", "") or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        cam, val = part.split(":", 1)
        try:
            offsets[cam.strip().lower()] = int(float(val.strip()))
        except ValueError:
            continue

    return SupervisorConfig(
        front_cameras=front or ["front", "left", "right"],
        sync_window_frames=int(_get("supervisor_sync_window_frames", 15)),
        clock_offsets_frames=offsets,
        iou_same_object=float(_get("supervisor_iou_same_object", 0.3)),
        center_dist_frac=float(_get("supervisor_center_dist_frac", 0.15)),
        min_confirm_cameras=int(_get("supervisor_min_confirm_cameras", 0)),
        back_window_s=float(_get("supervisor_back_window_s", 10.0)),
        side_by_side_split=bool(_get("multi_camera_side_by_side_split", True)),
        default_frame_width=int(_get("default_frame_width", 1280)),
        log_path=str(_get("supervisor_log_path", "plateflow_outputs/logs/supervisor.log")),
        log_level=str(_get("supervisor_log_level", "INFO")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dedicated supervisor logger
# ─────────────────────────────────────────────────────────────────────────────

_LOGGER_NAME = "supervisor"
_HANDLER_MARKER = "_supervisor_file_handler"


def configure_supervisor_logger(log_path: str, level: str = "INFO") -> logging.Logger:
    """Return the dedicated ``"supervisor"`` logger, wiring a FileHandler to
    ``log_path`` (creating its directory) exactly once per distinct path.

    The logger does NOT propagate to the root logger, so its output stays separate
    from the app's main log. Calling this again with the same path is a no-op; with a
    new path it swaps the FileHandler (useful for tests that redirect the log)."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    abs_path = os.path.abspath(log_path)
    for h in list(logger.handlers):
        if getattr(h, _HANDLER_MARKER, False):
            if getattr(h, "_supervisor_path", None) == abs_path:
                return logger  # already wired to this path
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass

    log_dir = os.path.dirname(abs_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(abs_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    setattr(handler, _HANDLER_MARKER, True)
    setattr(handler, "_supervisor_path", abs_path)
    logger.addHandler(handler)
    return logger


def get_supervisor_logger() -> logging.Logger:
    """The dedicated supervisor logger (assumes configure_supervisor_logger ran)."""
    return logging.getLogger(_LOGGER_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (stdlib only — no numpy dependency)
# ─────────────────────────────────────────────────────────────────────────────

Bbox = Tuple[float, float, float, float]


def _as_bbox(raw: Any) -> Optional[Bbox]:
    if isinstance(raw, dict):
        try:
            return (float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"]))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except (TypeError, ValueError):
            return None
    return None


def _iou(a: Bbox, b: Bbox) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xb - xa), max(0.0, yb - ya)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b: Bbox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraTrack:
    """One spatially-coherent track within a single camera (after re-splitting on
    spatial jumps). All frames are in the common (reference-fps aligned) timeline."""
    camera: str
    raw_id: Optional[int]
    perm_ids: List[int]
    class_name: str
    f0: int
    f1: int
    t0: float
    t1: float
    cx: Optional[float]           # normalised center-x [0,1] (for lane ordering)
    confidence: float


@dataclass
class ConfirmedTruck:
    consolidated_id: str
    class_name: str
    cameras: List[str]
    # camera -> list of member dicts {perm_id, raw_id, f0, f1, t0, t1, cx}
    members: Dict[str, List[Dict[str, Any]]]
    entry_frame: int
    exit_frame: int
    entry_time: float
    exit_time: float
    lane_index: int = 0

    def perm_ids_for(self, camera: str) -> List[int]:
        return [m["perm_id"] for m in self.members.get(camera, []) if m.get("perm_id") is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor camera
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorCamera:
    def __init__(self, config: Optional[SupervisorConfig] = None, logger: Optional[logging.Logger] = None) -> None:
        self.config = config or SupervisorConfig()
        self.logger = logger or configure_supervisor_logger(self.config.log_path, self.config.log_level)

    # ---- public API -----------------------------------------------------------

    def confirm_front_trio(
        self,
        frame_tracks_by_cam: Dict[str, List[Dict[str, Any]]],
        fps_by_cam: Optional[Dict[str, float]] = None,
    ) -> List[ConfirmedTruck]:
        """Confirm physical trucks from the front trio's per-frame track logs.

        Returns the list of CONFIRMED trucks (ordered by entry frame, then lane).
        Writes CONFIRMED / REJECTED / SPLIT decisions to the supervisor log.
        """
        cfg = self.config
        fps_by_cam = {k.lower(): float(v) for k, v in (fps_by_cam or {}).items() if v}

        # Only front-facing cameras that actually carry per-frame data participate.
        present: Dict[str, List[Dict[str, Any]]] = {}
        for cam in cfg.front_cameras:
            rows = frame_tracks_by_cam.get(cam) or frame_tracks_by_cam.get(cam.lower())
            rows = [r for r in (rows or []) if (r.get("class_name") in TRUCK_CLASSES)]
            if rows:
                present[cam.lower()] = rows
        if not present:
            self.logger.info("[SUPERVISOR] no front-trio frame tracks — nothing to confirm")
            return []

        ref_fps = max((fps_by_cam.get(c, 30.0) for c in present), default=30.0) or 30.0
        min_confirm = cfg.min_confirm_cameras or len(present)

        self.logger.info(
            "[SUPERVISOR] front cameras=%s  ref_fps=%.3f  sync_window=%d frames  "
            "min_confirm=%d  iou_same_object=%.2f",
            sorted(present.keys()), ref_fps, cfg.sync_window_frames, min_confirm, cfg.iou_same_object,
        )

        # Step 2 — build spatially-coherent camera-tracks per camera (with re-splits).
        all_tracks: List[CameraTrack] = []
        for cam, rows in present.items():
            cam_fps = fps_by_cam.get(cam, ref_fps) or ref_fps
            offset = cfg.clock_offsets_frames.get(cam, 0)
            frame_w, frame_h = self._infer_frame_size(rows)
            self._log_same_frame_splits(cam, rows, frame_w, frame_h)
            all_tracks.extend(
                self._build_camera_tracks(cam, rows, cam_fps, ref_fps, offset, frame_w, frame_h)
            )

        # Step 1 — confirm across cameras (temporal bursts -> lanes -> agreement).
        return self._confirm(all_tracks, min_confirm, ref_fps)

    # ---- step 2: per-camera spatial tracks ------------------------------------

    def _spatially_distinct(self, a: Bbox, b: Bbox, frame_w: float, frame_h: float) -> Tuple[bool, float]:
        """(distinct?, iou). Distinct = low IoU AND far centres (see config comment)."""
        cfg = self.config
        iou = _iou(a, b)
        (ax, ay), (bx, by) = _center(a), _center(b)
        far = (abs(ax - bx) > cfg.center_dist_frac * frame_w) or (
            abs(ay - by) > cfg.center_dist_frac * frame_h
        )
        return (iou < cfg.iou_same_object and far), iou

    def _infer_frame_size(self, rows: List[Dict[str, Any]]) -> Tuple[float, float]:
        w = self.config.default_frame_width
        h = self.config.default_frame_height
        for r in rows:
            b = _as_bbox(r.get("bbox"))
            if b:
                w = max(w, b[2])
                h = max(h, b[3])
        return float(w), float(h)

    def _build_camera_tracks(
        self,
        camera: str,
        rows: List[Dict[str, Any]],
        cam_fps: float,
        ref_fps: float,
        offset: int,
        frame_w: float,
        frame_h: float,
    ) -> List[CameraTrack]:
        """Group a camera's rows into spatially-coherent tracks.

        A raw ByteTrack id is followed frame-to-frame; when its box jumps to a
        spatially disjoint region between adjacent frames, the id is re-split into a
        new track (we do NOT trust the raw id blindly). Distinct raw ids are, of
        course, distinct tracks already."""
        by_raw: Dict[Any, List[Dict[str, Any]]] = {}
        for r in rows:
            by_raw.setdefault(r.get("track_id"), []).append(r)

        segments: List[List[Dict[str, Any]]] = []
        for raw_id, rrows in by_raw.items():
            rrows.sort(key=lambda r: int(r.get("frame_id", 0)))
            seg = [rrows[0]]
            for prev, cur in zip(rrows, rrows[1:]):
                pb, cb = _as_bbox(prev.get("bbox")), _as_bbox(cur.get("bbox"))
                if pb and cb:
                    distinct, iou = self._spatially_distinct(pb, cb, frame_w, frame_h)
                    if distinct:
                        self.logger.info(
                            "[SPLIT] cam=%s raw#%s spatial jump between f%s->f%s: "
                            "bbox %s / %s IoU=%.3f -> re-split into a new id",
                            camera, raw_id, prev.get("frame_id"), cur.get("frame_id"),
                            _fmt_bbox(pb), _fmt_bbox(cb), iou,
                        )
                        segments.append(seg)
                        seg = [cur]
                        continue
                seg.append(cur)
            segments.append(seg)

        tracks: List[CameraTrack] = []
        for seg in segments:
            tracks.append(self._make_track(camera, seg, cam_fps, ref_fps, offset, frame_w))
        return tracks

    def _make_track(
        self,
        camera: str,
        seg: List[Dict[str, Any]],
        cam_fps: float,
        ref_fps: float,
        offset: int,
        frame_w: float,
    ) -> CameraTrack:
        def aligned(frame_id: int) -> int:
            return int(round(frame_id * (ref_fps / cam_fps))) + offset

        frames = [int(r.get("frame_id", 0)) for r in seg]
        af = [aligned(f) for f in frames]
        f0, f1 = min(af), max(af)
        # dominant class
        class_counts: Dict[str, int] = {}
        for r in seg:
            class_counts[r.get("class_name", "")] = class_counts.get(r.get("class_name", ""), 0) + 1
        class_name = max(class_counts, key=lambda k: class_counts[k])
        perm_ids = sorted({int(r["perm_id"]) for r in seg if r.get("perm_id") is not None})
        cxs = []
        for r in seg:
            b = _as_bbox(r.get("bbox"))
            if b and frame_w > 0:
                cxs.append(max(0.0, min(1.0, ((b[0] + b[2]) / 2.0) / frame_w)))
        cx = sum(cxs) / len(cxs) if cxs else None
        conf = max((float(r.get("confidence", 0.0)) for r in seg), default=0.0)
        raw_id = seg[0].get("track_id")
        return CameraTrack(
            camera=camera,
            raw_id=int(raw_id) if raw_id is not None else None,
            perm_ids=perm_ids,
            class_name=class_name,
            f0=f0,
            f1=f1,
            t0=f0 / ref_fps,
            t1=f1 / ref_fps,
            cx=cx,
            confidence=conf,
        )

    def _log_same_frame_splits(
        self, camera: str, rows: List[Dict[str, Any]], frame_w: float, frame_h: float
    ) -> None:
        """Log (once per raw-id pair) every same-class pair of detections that co-occur
        in one frame but are spatially distinct — the direct id-68 evidence."""
        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_frame.setdefault(int(r.get("frame_id", 0)), []).append(r)
        seen: set = set()
        for frame_id, dets in by_frame.items():
            for a, b in combinations(dets, 2):
                if a.get("class_name") != b.get("class_name"):
                    continue
                ba, bb = _as_bbox(a.get("bbox")), _as_bbox(b.get("bbox"))
                if not (ba and bb):
                    continue
                distinct, iou = self._spatially_distinct(ba, bb, frame_w, frame_h)
                if not distinct:
                    continue
                pair = tuple(sorted((str(a.get("track_id")), str(b.get("track_id")))))
                if pair in seen:
                    continue
                seen.add(pair)
                self.logger.info(
                    "[SPLIT] cam=%s f%s: two %s detections at distinct coordinates "
                    "(raw#%s bbox %s vs raw#%s bbox %s, IoU=%.3f) -> separate ids, never merged",
                    camera, frame_id, a.get("class_name"),
                    a.get("track_id"), _fmt_bbox(ba), b.get("track_id"), _fmt_bbox(bb), iou,
                )

    # ---- step 1: cross-camera confirmation ------------------------------------

    def _confirm(
        self, tracks: List[CameraTrack], min_confirm: int, ref_fps: float
    ) -> List[ConfirmedTruck]:
        cfg = self.config
        confirmed: List[ConfirmedTruck] = []
        if not tracks:
            return confirmed

        # Cluster all camera-tracks into temporal bursts (overlap within tolerance).
        tracks_sorted = sorted(tracks, key=lambda t: t.f0)
        bursts: List[List[CameraTrack]] = []
        cur: List[CameraTrack] = [tracks_sorted[0]]
        cur_end = tracks_sorted[0].f1
        for t in tracks_sorted[1:]:
            if t.f0 <= cur_end + cfg.sync_window_frames:
                cur.append(t)
                cur_end = max(cur_end, t.f1)
            else:
                bursts.append(cur)
                cur = [t]
                cur_end = t.f1
        bursts.append(cur)

        counter = 0
        for burst in bursts:
            by_cam: Dict[str, List[CameraTrack]] = {}
            for t in burst:
                by_cam.setdefault(t.camera, []).append(t)
            # Side-by-side is a distinct-COORDINATES signal, not merely "two tracks".
            # A lane split requires, within some camera, two tracks that are CONCURRENT
            # (overlap in time) AND at distinct horizontal positions. Two tracks that are
            # SEQUENTIAL (one raw id re-acquired as another at the same spot) are the SAME
            # lane and must stay ONE truck — never split one truck across ids.
            n_lanes = 1
            if cfg.side_by_side_split:
                for ts in by_cam.values():
                    if self._has_concurrent_distinct(ts):
                        n_lanes = 2
                        break

            if n_lanes <= 1:
                lanes = [burst]
            else:
                lanes = self._assign_lanes(burst, n_lanes)
                self.logger.info(
                    "[SPLIT] side-by-side burst f%d-f%d -> %d lanes by horizontal position "
                    "(distinct coordinates override any timing merge)",
                    min(t.f0 for t in burst), max(t.f1 for t in burst), n_lanes,
                )

            for lane in lanes:
                if not lane:
                    continue
                # Confirm on the class shared by the most DISTINCT cameras.
                cams_by_class: Dict[str, set] = {}
                for t in lane:
                    cams_by_class.setdefault(t.class_name, set()).add(t.camera)
                best_class = max(cams_by_class, key=lambda c: len(cams_by_class[c]))
                agree_cams = cams_by_class[best_class]
                f0 = min(t.f0 for t in lane)
                f1 = max(t.f1 for t in lane)
                if len(agree_cams) < min_confirm:
                    self.logger.info(
                        "[REJECTED] class=%s cameras=%s (%d < %d required) window f%d-f%d "
                        "(%.2fs-%.2fs) — seen by too few cameras, not a truck",
                        best_class, sorted(agree_cams), len(agree_cams), min_confirm,
                        f0, f1, f0 / ref_fps, f1 / ref_fps,
                    )
                    continue

                counter += 1
                lane_members = [t for t in lane if t.class_name == best_class]
                members: Dict[str, List[Dict[str, Any]]] = {}
                for t in lane_members:
                    members.setdefault(t.camera, []).append({
                        "perm_id": t.perm_ids[0] if t.perm_ids else None,
                        "perm_ids": t.perm_ids,
                        "raw_id": t.raw_id,
                        "f0": t.f0, "f1": t.f1,
                        "t0": t.t0, "t1": t.t1,
                        "cx": t.cx,
                    })
                exit_frame = max(t.f1 for t in lane_members)
                entry_frame = min(t.f0 for t in lane_members)
                ct = ConfirmedTruck(
                    consolidated_id=f"S{counter}",
                    class_name=best_class,
                    cameras=sorted(agree_cams),
                    members=members,
                    entry_frame=entry_frame,
                    exit_frame=exit_frame,
                    entry_time=entry_frame / ref_fps,
                    exit_time=exit_frame / ref_fps,
                    lane_index=0,
                )
                confirmed.append(ct)
                self.logger.info(
                    "[CONFIRMED] id=%s class=%s cameras=%s window f%d-f%d (%.2fs-%.2fs) "
                    "exit=%.2fs",
                    ct.consolidated_id, best_class, ct.cameras, entry_frame, exit_frame,
                    ct.entry_time, ct.exit_time, ct.exit_time,
                )

        # Order by entry time, then lane position; renumber ids for stable output.
        confirmed.sort(key=lambda c: (c.entry_frame, _first_cx(c)))
        for lane_idx, ct in enumerate(confirmed):
            ct.consolidated_id = f"S{lane_idx + 1}"
            ct.lane_index = lane_idx
        return confirmed

    def _has_concurrent_distinct(self, tracks: List[CameraTrack]) -> bool:
        """True if a camera has two tracks that overlap in time AND sit at distinct
        horizontal positions (the true side-by-side signal). Sequential same-lane
        re-acquisitions do NOT count."""
        gap = self.config.center_dist_frac
        for a, b in combinations(tracks, 2):
            if a.cx is None or b.cx is None:
                continue
            overlap = min(a.f1, b.f1) - max(a.f0, b.f0)  # >= 0 means the windows touch
            if overlap >= 0 and abs(a.cx - b.cx) > gap:
                return True
        return False

    def _lane_centers(self, cxs: List[float], n_lanes: int) -> List[float]:
        """Cluster normalized center-x values into ``n_lanes`` lane centres by splitting
        at the largest gap (robust for the domain's "at most two trucks at once")."""
        cxs = sorted(cxs)
        if not cxs:
            return [0.5] * n_lanes
        if n_lanes < 2 or len(cxs) < 2:
            mean = sum(cxs) / len(cxs)
            return [mean] * n_lanes
        gap_idx = max(range(len(cxs) - 1), key=lambda i: cxs[i + 1] - cxs[i])
        left, right = cxs[: gap_idx + 1], cxs[gap_idx + 1:]
        return [sum(left) / len(left), sum(right) / len(right)]

    def _assign_lanes(self, burst: List[CameraTrack], n_lanes: int) -> List[List[CameraTrack]]:
        """Split a side-by-side burst into ``n_lanes`` lanes purely by horizontal
        position: every track (from any camera) joins the lane whose centre is nearest.

        Assigning by nearest centre — rather than zipping each camera's tracks in order —
        keeps a re-acquired same-lane track in ITS lane instead of forcing it into a
        second lane, protecting the "never split one truck across ids" invariant."""
        centers = self._lane_centers([t.cx for t in burst if t.cx is not None], n_lanes)
        lanes: List[List[CameraTrack]] = [[] for _ in range(n_lanes)]
        for t in burst:
            cx = t.cx if t.cx is not None else 0.5
            i = min(range(n_lanes), key=lambda j: abs(cx - centers[j]))
            lanes[i].append(t)
        return lanes


def _fmt_bbox(b: Bbox) -> str:
    return f"({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])})"


def _first_cx(ct: ConfirmedTruck) -> float:
    for members in ct.members.values():
        for m in members:
            if m.get("cx") is not None:
                return float(m["cx"])
    return 0.5
