










# syntax=docker/dockerfile:1.7
#
# PlateFlow — single-image deploy for a GPU EC2 box (DLAMI, driver 595 / CUDA 13.x).
# One process: `uvicorn app.main:app` loads YOLO on the GPU, runs the OCR workers,
# serves the REST API AND the built React frontend on :8000.
#
# Two Python venvs on purpose (mirrors the working Windows setup):
#   /opt/venv      main app  — torch (cu124) + YOLO + FastAPI
#   /opt/venv-ocr  OCR worker — paddlepaddle-gpu + paddleocr + CPU torch
# The OCR worker runs as a SEPARATE process out of /opt/venv-ocr so paddle's CUDA
# libraries never collide with the main venv's torch-CUDA.

########################  Stage 1 — build the React frontend  ##################
FROM node:20-bookworm-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# VITE_API_BASE_URL is intentionally unset -> the app calls the API at a relative
# path, so it works behind http://<host>:8000 with no rebuild.
RUN npm run build          # -> /fe/dist

########################  Stage 2 — CUDA runtime + app  ########################
# CUDA 12.6 + cuDNN base (paddle wheels target cu126). torch brings its own cu124
# libs in-wheel; the host's newer driver (595) runs both via forward-compat.
# If this exact tag ever fails to pull, bump the patch (e.g. 12.6.2 / 12.4.1).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend

# Python 3.11 (numpy 2.3 needs >=3.11; Ubuntu 22.04 ships 3.10) + OpenCV/ffmpeg
# runtime libs (cv2 needs libGL; video decode needs ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---- main venv: the app process (torch-cu124 + YOLO + FastAPI) ----
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
COPY backend/requirements.txt /tmp/requirements.txt
# torch/torchvision are the +cu124 wheels from the PyTorch index; the rest from PyPI.
RUN pip install -r /tmp/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124

# ---- ocr venv: the PaddleOCR worker subprocess (isolated CUDA libs) ----
# Swap `paddlepaddle-gpu` -> `paddlepaddle` (and drop the cn index + cudnn line)
# if the GPU wheel is troublesome; OCR then runs on CPU (still fine for plates).
RUN python3.11 -m venv /opt/venv-ocr && \
    /opt/venv-ocr/bin/pip install --upgrade pip && \
    /opt/venv-ocr/bin/pip install torch \
        --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv-ocr/bin/pip install paddlepaddle-gpu==3.2.0 \
        --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
        --extra-index-url https://pypi.org/simple/ && \
    /opt/venv-ocr/bin/pip install paddleocr==3.2.0 nvidia-cudnn-cu12==9.9.0.52

# ---- application code + built frontend ----
WORKDIR /app/backend
COPY backend/ /app/backend/
COPY --from=frontend /fe/dist /app/frontend/dist

# Runtime dirs (also mounted as volumes in compose so data survives redeploys).
RUN mkdir -p /app/backend/data /app/backend/plateflow_outputs /app/backend/model

EXPOSE 8000
# Single worker: the GPU model is loaded once per process and is not fork-safe.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
