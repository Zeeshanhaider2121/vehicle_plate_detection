from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
