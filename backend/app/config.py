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
