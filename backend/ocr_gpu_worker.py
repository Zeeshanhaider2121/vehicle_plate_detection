"""Standalone GPU PaddleOCR worker.

Runs in its OWN virtualenv (backend/.venv_ocr) where torch is CPU-only, so
modelscope's forced `import torch` loads no CUDA DLLs and paddlepaddle-gpu has the
GPU (RTX 3060) to itself. The main backend (which loads torch-CUDA for YOLO and
therefore cannot import paddle-gpu in-process on Windows — WinError 127) talks to
this process over localhost HTTP.

Protocol (tiny, dependency-free — only the paddle stack + stdlib):
  GET  /health -> 200 {"ready": bool, "device": str, "provider": "PaddleOCR"}
  POST /ocr    -> body is raw image bytes (PNG/JPEG); returns
                  200 {"text": str, "status": "ok"|"empty"|"error"}

Run:  python ocr_gpu_worker.py --host 127.0.0.1 --port 8899 --device gpu --lang en
The parent normally spawns this automatically; it can also be run by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# ── Engine (built once, guarded — PaddleOCR inference is not thread-safe) ─────
_engine = None
_engine_lock = threading.Lock()
_ready = False
_MIN_SCORE = float(os.getenv("PADDLE_OCR_MIN_SCORE", "0.5"))
_DEVICE = "gpu"
_LANG = "en"


def _build_engine() -> None:
    global _engine, _ready
    from paddleocr import PaddleOCR  # pulls paddlex -> modelscope -> torch(cpu) -> paddle(gpu)

    print(f"[ocr-worker] building PaddleOCR (device={_DEVICE}, lang={_LANG})…", flush=True)
    _engine = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=_LANG,
        device=_DEVICE,
    )
    _ready = True
    print("[ocr-worker] READY", flush=True)


def _extract_text(result) -> str:
    if not result:
        return ""
    texts = []
    for res in result:
        rec_texts = res.get("rec_texts") if hasattr(res, "get") else None
        if rec_texts is not None:  # PaddleOCR 3.x OCRResult
            rec_scores = res.get("rec_scores") or []
            for i, txt in enumerate(rec_texts):
                score = rec_scores[i] if i < len(rec_scores) else 1.0
                if txt and score >= _MIN_SCORE:
                    texts.append(str(txt).strip())
            continue
        if isinstance(res, (list, tuple)):  # PaddleOCR 2.x line list
            for line in res:
                try:
                    txt, score = line[1][0], float(line[1][1])
                except (IndexError, TypeError, ValueError):
                    continue
                if txt and score >= _MIN_SCORE:
                    texts.append(str(txt).strip())
    return " ".join(t for t in texts if t).strip()


def _run_ocr(image_bytes: bytes) -> tuple[str, str]:
    if not image_bytes:
        return "", "empty"
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # decodes to BGR — what paddle wants
    if img_bgr is None or img_bgr.size == 0:
        return "", "empty"
    try:
        with _engine_lock:
            result = _engine.predict(img_bgr) if hasattr(_engine, "predict") else _engine.ocr(img_bgr)
        text = _extract_text(result)
        return (text, "ok") if text else ("", "empty")
    except Exception as exc:  # noqa: BLE001 — never crash the worker on one bad crop
        print(f"[ocr-worker] predict error: {exc}", flush=True)
        return "", "error"


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._send(200, {"ready": _ready, "device": _DEVICE, "provider": "PaddleOCR"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/ocr"):
            self._send(404, {"error": "not found"})
            return
        if not _ready:
            self._send(503, {"text": "", "status": "error", "detail": "engine not ready"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            image_bytes = self.rfile.read(length) if length else b""
        except (TypeError, ValueError):
            self._send(400, {"text": "", "status": "error", "detail": "bad length"})
            return
        text, status = _run_ocr(image_bytes)
        self._send(200, {"text": text, "status": status})

    def log_message(self, *args):  # silence per-request stderr logging
        return


def main() -> None:
    global _DEVICE, _LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("OCR_GPU_WORKER_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("OCR_GPU_WORKER_PORT", "8899")))
    ap.add_argument("--device", default=os.getenv("PADDLE_OCR_DEVICE", "gpu"))
    ap.add_argument("--lang", default=os.getenv("PADDLE_OCR_LANG", "en"))
    args = ap.parse_args()
    _DEVICE, _LANG = args.device, args.lang

    # Build the engine before serving so /health flips to ready only when usable.
    try:
        _build_engine()
    except Exception as exc:  # noqa: BLE001
        print(f"[ocr-worker] FATAL: could not build engine: {exc}", flush=True)
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[ocr-worker] serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
