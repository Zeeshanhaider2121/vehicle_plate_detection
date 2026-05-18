# PlateFlow Frontend (React + Vite)

## Setup

```powershell
cd frontend
npm install
Copy-Item .env.example .env
```

## Run

```powershell
npm run dev
```

Frontend URL: `http://localhost:5173`

## Environment

- `VITE_API_BASE_URL`: backend base URL (default `http://localhost:8000`)

## Features now available

- Detection verification queue (`pending/verified/rejected`)
- Truck run JSON import (paste full run JSON)
- Truck ID explorer by `run_id` + `track_id`
- Video upload + model-run review form + approve/reject + approved trucks dashboard table
- Live video processing flow: start job → poll status → finalize result
