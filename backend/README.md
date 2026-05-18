# PlateFlow Backend (FastAPI)

## 1) Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

## 2) Configure Colab inference (recommended)

Edit `.env`:

```env
COLAB_INFER_URL=https://your-ngrok-or-cloudflared-url
COLAB_INFER_PATH=/infer
COLAB_ANALYZE_VIDEO_PATH=/analyze-video
COLAB_START_VIDEO_JOB_PATH=/analyze-video/start
COLAB_VIDEO_JOB_STATUS_PATH_TEMPLATE=/analyze-video/jobs/{job_id}
COLAB_VIDEO_JOB_RESULT_PATH_TEMPLATE=/analyze-video/jobs/{job_id}/result
VIDEO_JOB_TIMEOUT_SECONDS=600
MOCK_INFERENCE_IF_UNAVAILABLE=true
```

If `COLAB_INFER_URL` is empty, backend uses deterministic mock detections so frontend can still be tested.

## 3) Run API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: `http://localhost:8000/docs`

## API endpoints

- `GET /health`
- `POST /api/detect` (multipart file + optional `source_name`)
- `GET /api/detections?status=pending|verified|rejected&limit=50&offset=0`
- `PATCH /api/detections/{id}/verify`
- `PATCH /api/detections/{id}/reject`
- `POST /api/truck-runs/import` (paste full JSON with `session`, `summary`, `trucks`)
- `POST /api/truck-runs/upload-video` (multipart: `video` + optional `analysis_json`)
- `POST /api/truck-runs/video/start` (multipart: `video` + optional `analysis_json`)
- `GET /api/truck-runs/video/{job_id}/status`
- `POST /api/truck-runs/video/{job_id}/finalize`
- `GET /api/truck-runs/video/{job_id}/frame` (latest live JPEG frame)
- `GET /api/truck-runs`
- `GET /api/trucks?run_id=<run_id>&track_id=<truck_id>`

`/api/truck-runs/upload-video` behavior:
- If `analysis_json` is provided, backend imports it directly.
- Else backend calls `COLAB_INFER_URL/analyze-video` with uploaded video.

Async video flow (live processing):
1. `POST /api/truck-runs/video/start` → get `job_id`
2. Poll `GET /api/truck-runs/video/{job_id}/status` for progress
3. When `state=completed`, call `POST /api/truck-runs/video/{job_id}/finalize`
4. Use returned `run_id` in `GET /api/trucks?run_id=...`

## Colab side

Use `colab_inference_stub.py` in Colab. It now supports:
- `/infer` (image)
- `/analyze-video` (sync)
- `/analyze-video/start` + `/analyze-video/jobs/{job_id}` + `/result` (async live)
