from __future__ import annotations

from typing import Any

import requests

from .config import settings


class ColabVideoClient:
    def __init__(self) -> None:
        self.base_url = (settings.colab_infer_url or "").rstrip("/")

    def configured(self) -> bool:
        return bool(self.base_url)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def analyze_video_sync(self, video_bytes: bytes, filename: str) -> dict[str, Any]:
        response = requests.post(
            self._url(settings.colab_analyze_video_path),
            files={"file": (filename, video_bytes, "video/mp4")},
            timeout=max(settings.video_job_timeout_seconds, settings.inference_timeout_seconds),
        )
        response.raise_for_status()
        return response.json()

    def start_video_job(self, video_bytes: bytes, filename: str) -> dict[str, Any]:
        response = requests.post(
            self._url(settings.colab_start_video_job_path),
            files={"file": (filename, video_bytes, "video/mp4")},
            timeout=max(settings.inference_timeout_seconds, 120),
        )
        response.raise_for_status()
        return response.json()

    def get_video_job_status(self, job_id: str) -> dict[str, Any]:
        path = settings.colab_video_job_status_path_template.format(job_id=job_id)
        response = requests.get(
            self._url(path),
            timeout=max(settings.inference_timeout_seconds, 60),
        )
        response.raise_for_status()
        return response.json()

    def get_video_job_result(self, job_id: str) -> dict[str, Any]:
        path = settings.colab_video_job_result_path_template.format(job_id=job_id)
        response = requests.get(
            self._url(path),
            timeout=max(settings.inference_timeout_seconds, 120),
        )
        response.raise_for_status()
        return response.json()

    def get_video_job_frame(self, job_id: str) -> tuple[bytes, str]:
        path = settings.colab_video_job_frame_path_template.format(job_id=job_id)
        response = requests.get(
            self._url(path),
            timeout=max(settings.inference_timeout_seconds, 60),
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg")
        return response.content, content_type
