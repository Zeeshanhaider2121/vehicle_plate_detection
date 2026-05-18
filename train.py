"""
YOLOE-26x-seg Training — Best Practices
========================================
Dataset  : Re-export from Roboflow as "YOLOv8 Segmentation" format
Model    : yoloe-26x-seg.pt  (Instance Segmentation)
Install  : pip install ultralytics albumentations

Usage    : python train.py
"""

import os
import zipfile
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — only section you need to touch
# ══════════════════════════════════════════════════════════════════════════════
ZIP_PATH    = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\Dataset_preparation\My First Project.v5i.yolov8.zip"
DATASET_DIR = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\dataset"
OUTPUT_DIR  = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\output"

# ── Hardware
BATCH    = 8       # 16 if VRAM ≥ 12 GB | 8 if 8 GB | 4 if 6 GB
WORKERS  = 4       # number of CPU threads for data loading

# ── Training schedule
EPOCHS   = 150
IMG_SIZE = 640
PATIENCE = 30      # stop early if no improvement for this many epochs

# ══════════════════════════════════════════════════════════════════════════════


# ── Step 1: Extract & fix dataset paths ──────────────────────────────────────
def prepare_dataset() -> str:
    dst = Path(DATASET_DIR)
    if not dst.exists():
        print("Extracting dataset ...")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(dst)
        print("Extracted.")

    text = (dst / "data.yaml").read_text()
    for split in ("train", "valid", "test"):
        for pat in (f"../{split}/images", f"{split}/images"):
            text = text.replace(pat, str(dst / split / "images"))

    fixed = dst / "data_fixed.yaml"
    fixed.write_text(text)
    print(f"YAML ready → {fixed}")
    return str(fixed)


# ── Step 2: Train ─────────────────────────────────────────────────────────────
def train(yaml_path: str):
    model = YOLO("yoloe-26x-seg.pt")   # downloads automatically on first run

    model.train(
        # ── Data ──────────────────────────────────────────────────────────────
        data        = yaml_path,
        imgsz       = IMG_SIZE,
        batch       = BATCH,
        workers     = WORKERS,

        # ── Schedule ──────────────────────────────────────────────────────────
        epochs      = EPOCHS,
        patience    = PATIENCE,         # early stopping

        # ── Optimizer (MuSGD — YOLO26 recommended) ────────────────────────────
        optimizer   = "SGD",
        lr0         = 0.01,             # initial learning rate
        lrf         = 0.01,             # final lr = lr0 × lrf
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs    = 5,           # gentle start — good for small datasets
        warmup_momentum  = 0.8,
        warmup_bias_lr   = 0.1,

        # ── Loss weights (tuned for multi-class vehicle scenes) ───────────────
        box         = 7.5,              # bounding box regression loss
        cls         = 0.5,             # classification loss
        dfl         = 1.5,             # distribution focal loss (detection head)

        # ── Augmentation — strong because dataset is small (686 images) ───────
        hsv_h       = 0.015,           # hue shift
        hsv_s       = 0.7,             # saturation shift
        hsv_v       = 0.4,             # brightness shift
        degrees     = 5.0,             # random rotation ±5°
        translate   = 0.1,             # random translation 10%
        scale       = 0.5,             # random scale 50%
        shear       = 2.0,             # random shear ±2°
        perspective = 0.0001,          # slight perspective warp (helps CCTV angles)
        flipud      = 0.0,             # no vertical flip (trucks aren't upside-down)
        fliplr      = 0.5,             # horizontal flip 50%
        mosaic      = 1.0,             # mosaic augmentation (combines 4 images)
        mixup       = 0.1,             # mixup augmentation
        copy_paste  = 0.1,             # copy-paste (great for instance seg)
        auto_augment= "randaugment",   # additional RandAugment policy

        # ── Regularisation ────────────────────────────────────────────────────
        dropout     = 0.0,             # no dropout (handled by augmentation)
        label_smoothing = 0.0,

        # ── Inference settings during validation ──────────────────────────────
        conf        = 0.001,           # low threshold during val for full mAP curve
        iou         = 0.6,

        # ── Output ────────────────────────────────────────────────────────────
        project     = os.path.join(OUTPUT_DIR, "runs"),
        name        = "yoloe26x_vehicle",
        exist_ok    = True,
        save        = True,
        save_period = 10,              # checkpoint every 10 epochs
        plots       = True,            # loss curves, confusion matrix, PR curve
        val         = True,
        verbose     = True,
    )

    best = Path(OUTPUT_DIR) / "runs" / "yoloe26x_vehicle" / "weights" / "best.pt"
    print(f"\n{'─'*60}")
    print(f"  Training complete!")
    print(f"  Best weights → {best}")
    print(f"  Training plots → {OUTPUT_DIR}/runs/yoloe26x_vehicle/")
    print(f"{'─'*60}")
    return str(best)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    yaml_path = prepare_dataset()
    best_pt   = train(yaml_path)