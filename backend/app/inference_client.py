from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import local_inference

from .schemas import InferenceRecord


class InferenceService:
    def infer(self, image_bytes: bytes, source_name: str | None) -> tuple[list[InferenceRecord], dict[str, Any], bool]:
        with tempfile.TemporaryDirectory(prefix="plateflow_image_") as tmp_dir:
            image_path = Path(tmp_dir) / (source_name or "upload.jpg")
            image_path.write_bytes(image_bytes)
            results = local_inference.model.predict(
                str(image_path),
                verbose=False,
                conf=local_inference.CONF,
                iou=local_inference.IOU_THRESH,
                half=local_inference.USE_HALF,
                device=local_inference.DEVICE,
            )

        detections: list[InferenceRecord] = []
        if results:
            names = results[0].names or {}
            boxes = results[0].boxes
            if boxes is not None:
                for det_index in range(len(boxes)):
                    cls_id = int(boxes.cls[det_index].item())
                    conf_val = float(boxes.conf[det_index].item())
                    cls_name = str(names.get(cls_id, f"class_{cls_id}"))
                    if cls_name in local_inference.TRUCK_CLASSES and conf_val < local_inference.TRUCK_CONF_THRESH:
                        continue
                    xyxy = boxes.xyxy[det_index].tolist()
                    detections.append(
                        InferenceRecord(
                            plate_text=cls_name,
                            confidence=round(conf_val, 4),
                            bbox={
                                "x1": round(float(xyxy[0]), 2),
                                "y1": round(float(xyxy[1]), 2),
                                "x2": round(float(xyxy[2]), 2),
                                "y2": round(float(xyxy[3]), 2),
                            },
                        )
                    )

        payload: dict[str, Any] = {
            "detections": [item.model_dump() for item in detections],
            "filename": source_name,
            "device": local_inference.DEVICE,
            "model": local_inference.MODEL_PATH,
        }
        return detections, payload, False
