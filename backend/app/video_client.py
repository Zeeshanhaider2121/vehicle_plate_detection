from __future__ import annotations

import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import local_inference


class LocalVideoServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LocalVideoService:
    """In-process video inference service used by the main FastAPI app."""

    def analyze_video_sync(self, video_bytes: bytes, filename: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="plateflow_sync_") as tmp_dir:
            video_path = Path(tmp_dir) / filename
            video_path.write_bytes(video_bytes)
            return local_inference._run_video_analysis(str(video_path), job=None)

    def start_video_job(self, video_bytes: bytes, filename: str) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        tmp_dir = tempfile.mkdtemp(prefix="plateflow_job_")
        video_path = Path(tmp_dir) / filename
        video_path.write_bytes(video_bytes)

        with local_inference._jobs_lock:
            local_inference.JOBS[job_id] = local_inference.VideoJob(
                job_id=job_id,
                state="queued",
                message="Job queued.",
                temp_video_path=str(video_path),
            )
        threading.Thread(target=local_inference._run_job, args=(job_id,), daemon=True).start()
        return {"job_id": job_id, "state": "queued", "message": "Video job created."}

    def get_video_job_status(self, job_id: str) -> dict[str, Any]:
        with local_inference._jobs_lock:
            job = local_inference.JOBS.get(job_id)
        if job is None:
            raise LocalVideoServiceError("Job not found.", status_code=404)
        return {
            "job_id": job.job_id,
            "state": job.state,
            "progress": job.progress,
            "frame_id": job.frame_id,
            "latest_frame_id": job.latest_frame_id,
            "total_frames": job.total_frames,
            "fps": job.fps,
            "message": job.message,
            "error": job.error,
            "ocr_log": job.ocr_log[-20:],
            "json_snapshot": job.json_snapshot,
        }

    def get_video_job_result(self, job_id: str) -> dict[str, Any]:
        with local_inference._jobs_lock:
            job = local_inference.JOBS.get(job_id)
        if job is None:
            raise LocalVideoServiceError("Job not found.", status_code=404)
        if job.state == "failed":
            raise LocalVideoServiceError(job.error or "Job failed.", status_code=500)
        if job.state != "completed" or job.result is None:
            raise LocalVideoServiceError("Job is not completed yet.", status_code=409)
        return {"job_id": job.job_id, "state": job.state, "result": job.result}

    def get_video_job_frame(self, job_id: str) -> tuple[bytes, str]:
        with local_inference._jobs_lock:
            job = local_inference.JOBS.get(job_id)
        if job is None:
            raise LocalVideoServiceError("Job not found.", status_code=404)
        if not job.latest_frame_jpeg:
            raise LocalVideoServiceError("No frame available yet.", status_code=404)
        return job.latest_frame_jpeg, "image/jpeg"
