import cv2
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient
import os

# ---------- 1. Connect to Roboflow ----------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="cWbK2rE1bbovisDecNio"   # consider using env var
)

# ---------- 2. Video paths (use raw string to avoid escaping issues) ----------
input_video_path = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\Dataset_preparation\in-lane-3-back-zoom-nvr-20260429104732-20260429114529-1999865_UnbMQ8Jc (1).mp4"
output_video_path = "output_annotated.mp4"

# ---------- 3. Open video ----------
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {input_video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# ---------- 4. Process frames ----------
frame_id = 0
print(f"Processing {total_frames} frames...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Run workflow on the current frame (numpy array)
        result = client.run_workflow(
            workspace_name="zeeshans-workspace-oosca",
            workflow_id="detect-count-and-visualize-8",
            images={"image": frame},   # directly pass the numpy array (BGR)
            use_cache=True
        )
        # Result is a list of dicts (one per input image)
        workflow_output = result[0]
        annotated_base64 = workflow_output["output_image"]
        # Decode to image
        img_data = base64.b64decode(annotated_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        out.write(annotated_frame)

    except Exception as e:
        print(f"Error on frame {frame_id}: {e}")
        # Write original frame on error
        out.write(frame)

    frame_id += 1
    if frame_id % 100 == 0:
        print(f"Processed {frame_id}/{total_frames} frames")

# ---------- 5. Cleanup ----------
cap.release()
out.release()
print(f"✅ Video saved to {output_video_path}")