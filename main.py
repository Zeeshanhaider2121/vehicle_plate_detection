"""
Vehicle / Multi-Object Detector with Frame Saving
"""

import cv2
import base64
import json
import sys
import requests
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
VIDEO_PATH      = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\data\WhatsApp Video 2026-04-28 at 16.32.32.mp4"
MISTRAL_API_KEY = "XZrK8S4UUomwXBgewOckRihF0EOqx7vu"

MODEL = "pixtral-large-latest"        # Better for vision than mistral-large
NUM_FRAMES   = 15                     # Increased a bit
JPEG_QUALITY = 90
API_URL      = "https://api.mistral.ai/v1/chat/completions"

DETECTION_PROMPT = (
    "You are a strict logistics visual inspector.\n\n"
    "Look VERY carefully at the trailer/chassis behind the truck cab.\n"
    "A shipping container is a large rectangular metal box (usually 20ft or 40ft long, 8ft wide, 8.5ft high).\n\n"
    
    "Answer this question: "
    "Is there a full shipping container (ISO container) loaded on the trailer behind this truck?\n\n"
    
    "Return ONLY this exact JSON format:\n"
    "{\n"
    '  "truck_with_container": true/false,\n'
    '  "confidence": "high/medium/low",\n'
    '  "reasoning": "One clear sentence explaining what you see on the trailer",\n'
    '  "summary": "Short scene description",\n'
    '  "trailer_type": "empty chassis / flatbed / container / unknown",\n'
    '  "container_color": "none / blue / red / white / etc."\n'
    "}"
)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def frame_to_base64(frame) -> str:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to JPEG-encode frame")
    return base64.standard_b64encode(buffer).decode("utf-8")


def analyze_frame(b64_image: str, frame_idx: int) -> dict:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }
        ],
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Mistral API error {response.status_code}: {response.text[:300]}")

    raw_text = response.json()["choices"][0]["message"]["content"].strip()

    # Clean markdown if present
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").lstrip("json").strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        result = {"error": "Failed to parse JSON", "raw": raw_text[:500]}

    result["frame_index"] = frame_idx
    return result


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Error: Could not open the video file!")
        sys.exit(1)

    # Video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("=" * 70)
    print("  VIDEO INFORMATION")
    print("=" * 70)
    print(f"  File         : {Path(VIDEO_PATH).name}")
    print(f"  Total Frames : {total_frames:,}")
    print(f"  FPS          : {fps:.2f}")
    print("=" * 70)

    # Create output folder
    video_dir = Path(VIDEO_PATH).parent
    frames_dir = video_dir / "extracted_frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"\n📁 Frames will be saved to: {frames_dir}\n")

    # Extract frames
    actual_count = min(NUM_FRAMES, total_frames)
    frames = {}

    print(f"📸 Extracting {actual_count} frames...")
    for idx in range(actual_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = frame
            
            # Save frame to disk
            frame_path = frames_dir / f"frame_{idx:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            print(f"   Saved: frame_{idx:03d}.jpg")
        else:
            print(f"   ⚠️  Failed to read frame {idx}")

    cap.release()
    print(f"✅ Extracted and saved {len(frames)} frames.\n")

    # Analyze frames
    print("=" * 70)
    print("  🔍 ANALYZING FRAMES WITH MISTRAL")
    print("=" * 70)

    results = []
    for frame_idx, frame in frames.items():
        print(f"\n  Frame {frame_idx:>3} → Sending to Mistral...", end="", flush=True)
        
        b64 = frame_to_base64(frame)
        
        try:
            result = analyze_frame(b64, frame_idx)
            print(" Done")
        except Exception as e:
            print(" Failed")
            result = {"frame_index": frame_idx, "error": str(e)}

        results.append(result)

        # Print summary
        summary = result.get("summary", "No summary")
        print(f"     📝 {summary[:120]}{'...' if len(summary) > 120 else ''}")

    # Save full results
    results_path = video_dir / "detection_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  ✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"  Frames saved       : {frames_dir}")
    print(f"  Results saved      : {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()