from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PlateFlow API"
    app_env: str = "dev"
    database_url: str = "sqlite:///./plateflow.db"
    cors_origins: str = "http://localhost:5173"

    inference_timeout_seconds: int = 45
    video_job_timeout_seconds: int = 600
    multi_camera_order: str = "front,right,back,left"
    multi_camera_time_offsets_seconds: str = ""
    multi_camera_order_fallback: bool = True
    multi_camera_assume_single_entity: bool = False
    multi_camera_review_threshold: float = 0.75
    multi_camera_gate_mode: bool = False
    # Per-camera field authority.  The rear camera runs best_V2.pt, whose only OCR
    # classes are the truck cab fields, so it is the authoritative source for them:
    # its reads win even against a higher-confidence (but wrong) read from a
    # front/left/right camera running best.pt.
    multi_camera_camera_roles: str = "back:truck_number,truck_company,driver"
    track_fragment_merge_gap_seconds: float = 20.0
    track_fragment_merge_aggressive: bool = True
    # Fixed ground-truth truck-changeover timestamps for the gate.  Any track that
    # spans one of these points is split there (one physical truck per segment), and
    # same-camera fragments are never merged across a boundary.
    truck_time_boundaries_seconds: str = ""

    # ── Time-synchronized cross-camera gate sweep ──────────────────────────────
    # Consolidate per-camera tracks into one record per physical truck. Front-facing
    # cameras share one timeline and corroborate truck ENTRIES; the interior (back)
    # camera attaches only AFTER a truck deactivates up front. All knobs are config —
    # re-targeting another gate (different camera names/counts/timing) is config-only.
    multi_camera_gate_sweep: bool = True
    # Camera roles for the sweep (comma lists). Anything not listed as back is treated
    # as front-facing so unknown cameras still corroborate entries.
    front_facing_cameras: str = "front,right,left"
    back_cameras: str = "back"
    # A start opens a NEW truck only when >= min_start_support DISTINCT front cameras
    # start within corroboration_window_s, OR it is > new_truck_gap_s after the open
    # truck's last activity. Otherwise it is a re-acquisition/split and is ABSORBED.
    min_start_support: int = 2
    corroboration_window_s: float = 1.5
    new_truck_gap_s: float = 6.0
    # A back track attaches to a truck only if it STARTS within
    # [exit - back_attach_lead_s, exit + back_attach_window_s] of that truck's
    # front-camera exit time (last front end). Never while still active up front.
    # NOTE: back_attach_window_s is widened to absorb back_extra_offset_s below
    # (physical tolerance ~13 s + the ~10 s clock lag we add to the back timeline).
    back_attach_lead_s: float = 1.0
    back_attach_window_s: float = 23.0
    # Attribute back-camera reads to the correct truck BY TIME: the back camera runs one
    # continuous (virtual) track whose OCR history spans every truck that dwelt in front
    # of it, so instead of gluing the whole track to one front exit, each timestamped
    # read is routed to the front truck whose exit it is nearest (within the attach
    # window). Fixes multi-truck runs where cab fields (truck_number/company) otherwise
    # all land on one truck. Falls back to whole-track attach when a back track carries
    # no timestamped OCR history.
    multi_camera_back_attach_by_time: bool = True
    # Camera preference order for the consolidated plate (highest-confidence read in
    # the first listed camera that has one). Empty -> reuse multi_camera_order.
    plate_provenance_order: str = ""

    # ── Back-camera processing lag ─────────────────────────────────────────────
    # Trucks reach the interior (back) camera ~N s after they leave the front
    # cameras. We add this fixed offset to the back camera's timeline so its tracks
    # line up with the front exit they belong to. Coupled with back_attach_window_s.
    back_extra_offset_s: float = 10.0

    # ── Fields aggregated across all cameras (not role-exclusive) ───────────────
    # Collected from EVERY camera; highest-confidence wins, with a role-owner (back)
    # winning ties. truck_number/truck_company are read by every model, so a strong
    # front read may win. driver stays back-exclusive (it is NOT listed here).
    multi_camera_shared_fields: str = "truck_number,truck_company"
    shared_field_conf_tie_eps: float = 0.02

    # ── Side-by-side split ──────────────────────────────────────────────────────
    # Two trucks physically side-by-side (lane 1 + lane 2) are both seen by the side
    # cameras at once; without this they collapse into one dashboard row. Split a
    # corroborated entry burst into N trucks by horizontal (bbox center-x) position.
    # Fails safe: when geometry is ambiguous it does not split.
    multi_camera_side_by_side_split: bool = True
    # Two same-camera tracks count as side-by-side only when they overlap in time by
    # >= this fraction of the shorter track AND their normalized center-x differ by
    # >= the gap below (guards detector jitter / double boxes).
    side_by_side_min_overlap_ratio: float = 0.5
    side_by_side_min_centerx_gap: float = 0.15
    # Side cameras face the gate from opposite angles, so screen-left != world-left.
    # Mirror a camera's normalized center-x before left->right ordering, e.g.
    # "right:flip,left:flip".
    side_by_side_camera_orientation: str = ""
    # Frame width used to normalize bbox center-x when a camera's resolution is
    # unknown.
    default_frame_width: int = 1280

    # ── Supervisor camera (front-trio synchronized confirmation) ────────────────
    # A dedicated "supervisor camera" owns synchronized processing of the front trio
    # (front, left, right). It steps the three cameras in lockstep by frame/fps,
    # confirms a truck only when the front cameras agree on the same class in the same
    # frame window, and enforces the distinct-coordinates rule (two spatially disjoint
    # boxes in one frame are always different trucks). Back is attached afterward, by
    # the consolidation step, NOT the supervisor. Everything here is config-driven — no
    # hardcoded camera counts, ids, or frame numbers.
    supervisor_enabled: bool = True
    # Front-facing cameras owned by the supervisor. Empty -> reuse front_facing_cameras.
    supervisor_front_cameras: str = ""
    # Frame tolerance for "active in the same frame window" when confirming across the
    # front trio (SYNC_WINDOW_FRAMES in the spec).
    supervisor_sync_window_frames: int = 15
    # Per-camera clock offset (in FRAMES) applied when recordings do not start at the
    # same instant, e.g. "front:0,left:-3,right:5". A camera's common-timeline frame =
    # local frame_id + offset. Cameras absent from the list default to 0.
    supervisor_clock_offsets_frames: str = ""
    # Two same-class detections in ONE camera at ONE frame whose bbox IoU is below this
    # (AND whose centres are farther apart than supervisor_center_dist_frac of the frame)
    # are DIFFERENT trucks and must never share a consolidated id (IOU_SAME_OBJECT).
    supervisor_iou_same_object: float = 0.3
    # Center-distance threshold (fraction of frame width/height) that, together with the
    # IoU test, marks two boxes as spatially distinct. Either axis exceeding it splits.
    supervisor_center_dist_frac: float = 0.15
    # A truck is CONFIRMED only when at least this many DISTINCT front cameras agree.
    # 0 (default) means "require all present front cameras" (all three: front/left/right).
    supervisor_min_confirm_cameras: int = 0
    # Back camera is read ONLY within [exit, exit + BACK_WINDOW_S] after a truck's
    # front-trio exit. Default 10 s per spec.
    supervisor_back_window_s: float = 10.0
    # Dedicated supervisor log file (its own named logger + FileHandler, separate from
    # the app's main log). The directory is created if missing.
    supervisor_log_path: str = "plateflow_outputs/logs/supervisor.log"
    supervisor_log_level: str = "INFO"
    # PRESENCE-GLUE model. When True, the supervisor's confirmed trucks define the truck
    # WINDOWS (one truck per continuous physical presence of the front trio; a new truck
    # only when the frame empties), and consolidation glues EVERY detection whose time
    # overlaps a window to that ONE truck — nothing is dropped for lack of per-track
    # cross-camera agreement. When False, the legacy per-perm-id mapping is used (an
    # observation is dropped unless its (camera, perm_id) belongs to a confirmed truck).
    supervisor_presence_glue: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
