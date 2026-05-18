# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE INFERENCE PIPELINE WITH MINERU OCR
# Install: !pip install -q ultralytics opencv-python-headless requests matplotlib
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import os
import io
import re
import uuid
import time
import random
import zipfile
import requests
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

# ========================= CONFIGURATION =====================================
BEST_PT    = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\models\best.pt"
OUTPUT_DIR = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\output"
CONF       = 0.23
IOU        = 0.50
MASK_ALPHA = 0.35

# ✅ Use the token that actually works in your test
MINERU_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI5NTYwMDA2MiIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3ODc1MTM1MywiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiODA1Yzc0MTYtZmM1Zi00MmQ0LTljZjctZDA5YzM1MjNhM2I5IiwiZW1haWwiOiIiLCJleHAiOjE3ODY1MjczNTN9.0Fwe0jwakra4gmjjxzyzIKjTAtFubWdhJwlrzU5rBINwM5aqmgPb508Vgv-Z0P9XZYAMP6t-SlK4dMt5_cytxQ"

OCR_CLASSES = {
    "container_number", "container_side_no", "license_plate",
    "truck_number", "container_company_logo", "truck_company"
}

CLASS_NAMES = [
    "container_company_logo", "container_number", "container_side_no",
    "driver", "license_plate", "other_container_info",
    "truck_company", "truck_number", "truck_with_container",
    "truck_without_container",
]

random.seed(42)
PALETTE = {i: tuple(random.randint(60, 230) for _ in range(3))
           for i in range(len(CLASS_NAMES))}

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(BEST_PT)
print("✅ Model loaded.")
# =============================================================================


# ========================= MINERU HELPERS ====================================

def _parse_mineru_markdown(md_text: str) -> str:
    """
    Extract clean OCR text from MinerU markdown output.

    MinerU wraps image-detected text like this:
        ![](images/abc.jpg)
        <details>
        <summary>text_image</summary>
        GCXU555443
        </details>

    This function pulls out all text inside <details> blocks first,
    then falls back to plain non-image, non-tag lines.
    """
    # ── Priority 1: text inside <details>…</details> blocks ──────────────────
    details_texts = re.findall(
        r'<details>.*?<summary>[^<]*</summary>\s*(.*?)\s*</details>',
        md_text, flags=re.DOTALL
    )
    if details_texts:
        combined = " | ".join(t.strip() for t in details_texts if t.strip())
        if combined:
            return combined

    # ── Priority 2: plain text lines (skip image tags, HTML tags, blanks) ────
    lines = []
    for line in md_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("!["):        # markdown image
            continue
        if re.match(r'^<[^>]+>$', line): # bare HTML tag
            continue
        lines.append(line)

    return " ".join(lines).strip()


def _extract_md_from_zip(zip_url: str) -> str:
    """Download MinerU result zip, read full.md, return parsed clean text."""
    try:
        r = requests.get(zip_url, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                if name.endswith("full.md"):
                    raw_md = z.read(name).decode("utf-8")
                    text   = _parse_mineru_markdown(raw_md)
                    print(f"     ✓ OCR result: '{text[:120]}'")
                    return text
        print("     ⚠️ full.md not found in zip")
    except Exception as e:
        print(f"     ⚠️ Zip download/read error: {e}")
    return ""


def call_mineru_ocr(crop_rgb: np.ndarray) -> str:
    """
    Send an image crop to MinerU Precision API using the batch file-upload flow.
    Steps: get pre-signed URL → PUT image → poll batch result → unzip → parse text.
    Returns clean OCR text string, or "" on any failure.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}"
    }

    # ── Encode crop as PNG bytes ──────────────────────────────────────────────
    _, encoded = cv2.imencode(".png", crop_rgb)
    img_bytes  = encoded.tobytes()
    filename   = f"crop_{uuid.uuid4().hex[:8]}.png"
    data_id    = uuid.uuid4().hex

    # ── Step 1: Request pre-signed upload URL ─────────────────────────────────
    try:
        resp = requests.post(
            "https://mineru.net/api/v4/file-urls/batch",
            json={
                "files": [{"name": filename, "data_id": data_id}],
                "model_version": "vlm"
            },
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") != 0:
            print(f"     ⚠️ MinerU error: {result.get('msg')}")
            return ""
        batch_id   = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]
        print(f"     ✓ Upload URL received (batch: {batch_id[:8]}…)")
    except Exception as e:
        print(f"     ⚠️ MinerU Step-1 error: {e}")
        return ""

    # ── Step 2: PUT image bytes to pre-signed OSS URL (no auth header) ────────
    try:
        put_resp = requests.put(upload_url, data=img_bytes, timeout=60)
        if put_resp.status_code not in (200, 201):
            print(f"     ⚠️ Upload failed: HTTP {put_resp.status_code}")
            return ""
        print(f"     ✓ Image uploaded successfully")
    except Exception as e:
        print(f"     ⚠️ MinerU Step-2 error: {e}")
        return ""

    # ── Step 3: Poll batch result ─────────────────────────────────────────────
    poll_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    for attempt in range(40):      # ~120 s max
        time.sleep(3)
        try:
            poll_resp = requests.get(poll_url, headers=headers, timeout=30)
            poll_resp.raise_for_status()
            file_list = poll_resp.json().get("data", {}).get("extract_result", [])

            if not file_list:
                print(f"     … [{attempt+1}/40] waiting…")
                continue

            state = file_list[0].get("state", "unknown")
            print(f"     … [{attempt+1}/40] state: {state}")

            if state == "done":
                zip_url = file_list[0].get("full_zip_url", "")
                if not zip_url:
                    print("     ⚠️ No zip URL in response")
                    return ""
                return _extract_md_from_zip(zip_url)

            elif state == "failed":
                print(f"     ⚠️ Extraction failed: {file_list[0].get('err_msg')}")
                return ""

        except Exception as e:
            print(f"     ⚠️ Poll error [{attempt+1}]: {e}")

    print("     ⚠️ MinerU timed out after 120 s")
    return ""

# =============================================================================


# ========================= IMAGE PROCESSING ==================================

def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Upscale small crops to at least 64 px height; convert BGR→RGB."""
    h, w = crop_bgr.shape[:2]
    if h < 64:
        scale = 64 / h
        new_w = max(1, int(w * scale))
        crop_bgr = cv2.resize(crop_bgr, (new_w, 64), interpolation=cv2.INTER_CUBIC)
        print(f"     [preprocess] upscaled → {new_w}×64")
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def annotate_frame(frame, results, track_ids=False, ocr_map=None):
    """Draw segmentation masks, bounding boxes, class labels, and OCR text."""
    r     = results[0]
    boxes = r.boxes
    masks = r.masks
    H, W  = frame.shape[:2]

    if boxes is None or len(boxes) == 0:
        return frame

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id   = int(box.cls[0])
        conf_val = float(box.conf[0])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        colour   = PALETTE[cls_id]
        tid      = f"#{int(box.id[0])} " if track_ids and box.id is not None else ""

        # Segmentation mask or plain rectangle
        if masks is not None and i < len(masks.data):
            mask = masks.data[i].cpu().numpy()
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            overlay = frame.copy()
            overlay[mask > 0.5] = colour
            cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)
            contours, _ = cv2.findContours(
                (mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, colour, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Class label
        label = f"{tid}{cls_name} {conf_val:.2f}"
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - lh - bl - 4)), (x1 + lw, y1), colour, -1)
        cv2.putText(frame, label, (x1, max(lh, y1 - bl - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # OCR text (with shadow for readability on any background)
        if ocr_map and i in ocr_map and ocr_map[i]:
            txt   = ocr_map[i][:60]
            txt_y = y2 + 22
            if txt_y + 20 > H:
                txt_y = max(20, y1 - 10)
            cv2.putText(frame, txt, (x1 + 1, txt_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (x1, txt_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    return frame

# =============================================================================


# ========================= PIPELINE FUNCTIONS ================================

def run_image_with_ocr(image_path: str):
    """Single-image pipeline: detect → crop → OCR → annotate → save & display."""
    img_path = Path(image_path)
    frame    = cv2.imread(str(img_path))
    if frame is None:
        print(f"❌ Cannot read image: {img_path}")
        return

    print(f"\n{'='*60}")
    print(f"Image : {img_path.name}  ({frame.shape[1]}×{frame.shape[0]})")
    print(f"{'='*60}")

    results = model.predict(frame, conf=CONF, iou=IOU, verbose=False)
    boxes   = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("⚠️  No detections.")
        return

    print(f"Detected {len(boxes)} object(s)\n")

    crop_dir = os.path.join(OUTPUT_DIR, "crops", img_path.stem)
    os.makedirs(crop_dir, exist_ok=True)
    ocr_map = {}

    for i, box in enumerate(boxes):
        cls_id   = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(frame.shape[1], x2);  y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        cv2.imwrite(os.path.join(crop_dir, f"det{i}_{cls_name}.jpg"), crop)

        if cls_name not in OCR_CLASSES:
            continue

        print(f"\n  [{i}] {cls_name}  (crop: {crop.shape[1]}×{crop.shape[0]})")
        crop_rgb = preprocess_crop(crop)

        plt.figure(figsize=(8, 3))
        plt.imshow(crop_rgb)
        plt.title(f"Crop [{i}] — {cls_name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        text = call_mineru_ocr(crop_rgb)
        ocr_map[i] = text

    # Annotate & save
    annotated = annotate_frame(frame.copy(), results, track_ids=False, ocr_map=ocr_map)
    out_path  = os.path.join(OUTPUT_DIR, f"ocr_{img_path.name}")
    cv2.imwrite(out_path, annotated)
    print(f"\n✅ Annotated image → {out_path}")
    display(IPImage(out_path, width=900))

    # Summary table
    print("\n── OCR Summary ──────────────────────────────────────────────")
    for i, txt in ocr_map.items():
        cls_name = CLASS_NAMES[int(boxes[i].cls[0])]
        print(f"  [{i}] {cls_name:30s} → {f'{txt!r}' if txt else '(empty)'}")
    print("─────────────────────────────────────────────────────────────\n")


def run_video_with_ocr(video_path: str, every_n_frames: int = 5):
    """Video pipeline: track → OCR every N frames with ID-based caching."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(OUTPUT_DIR, "output_video_ocr.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"\nVideo: {W}×{H} @ {fps:.1f} FPS | {total} frames | OCR every {every_n_frames} frames")
    print(f"Output: {out_path}\n")

    track_ocr_cache = {}
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, conf=CONF, iou=IOU,
            tracker="bytetrack.yaml", persist=True, verbose=False
        )
        boxes   = results[0].boxes
        ocr_map = {}

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                cls_id   = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                if cls_name not in OCR_CLASSES:
                    continue

                track_id = int(box.id[0]) if box.id is not None else None
                run_now  = (frame_id % every_n_frames == 0)

                if track_id is not None and not run_now and track_id in track_ocr_cache:
                    ocr_map[i] = track_ocr_cache[track_id]
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1);  y1 = max(0, y1)
                x2 = min(W, x2);  y2 = min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue

                text = call_mineru_ocr(preprocess_crop(crop))
                ocr_map[i] = text
                if track_id is not None:
                    track_ocr_cache[track_id] = text

        annotated = annotate_frame(frame.copy(), results, track_ids=True, ocr_map=ocr_map)
        cv2.putText(annotated, f"Frame {frame_id}/{total}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(annotated)

        if frame_id % 50 == 0:
            n = len(boxes) if boxes is not None else 0
            print(f"  Frame {frame_id:5d}/{total} | dets: {n} | cache: {len(track_ocr_cache)}")
        frame_id += 1

    cap.release()
    writer.release()
    print(f"\n✅ Done — {frame_id} frames → {out_path}")

# =============================================================================


# ========================= ENTRY POINT =======================================
if __name__ == "__main__":
    run_image_with_ocr(r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\frame_conversion\data\output_frames\IN_LANE_3_PTZ(2)_NVR_20260429094351_20260429114352_1872780\frame_7193_119m_52s.jpg")

    # For video:
    # run_video_with_ocr("/content/your_video.mp4", every_n_frames=5)