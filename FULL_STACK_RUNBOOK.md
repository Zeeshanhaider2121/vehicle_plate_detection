# Full Stack Runbook (FastAPI + React + Colab GPU)

## Architecture

1. React frontend uploads a frame/image to FastAPI.
2. FastAPI calls Colab inference endpoint (`COLAB_INFER_URL + /infer`).
3. FastAPI stores each detection in SQLite with status `pending`.
4. Frontend table lets operator verify/reject rows.
5. Verified rows become status `verified` and are ready to pass forward.

## Start backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Start frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`.

## Colab GPU integration

1. In Colab Runtime, install:
```python
!pip install fastapi uvicorn python-multipart ultralytics pyngrok
```
2. Run `backend/colab_inference_stub.py` on port `8001`.
3. Expose with ngrok/cloudflared and copy public URL.
4. Put URL in `backend/.env` as `COLAB_INFER_URL`.
5. Restart backend.

## API payload expected from Colab

```json
{
  "detections": [
    {
      "plate_text": "LEB1234",
      "confidence": 0.92,
      "bbox": {"x1": 10, "y1": 20, "x2": 120, "y2": 65}
    }
  ]
}
```

Backend also accepts keys like `predictions` or `results`.

## Import your truck summary JSON

Live video flow endpoints:

- `POST /api/truck-runs/video/start`
- `GET /api/truck-runs/video/{job_id}/status`
- `POST /api/truck-runs/video/{job_id}/finalize`

Frontend now uses this flow to show live processing progress before loading dashboard rows.
