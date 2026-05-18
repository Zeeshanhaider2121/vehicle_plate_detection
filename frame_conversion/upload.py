import os
from roboflow import Roboflow
from dotenv import load_dotenv

# ── Load .env file ─────────────────────────────────────────
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("❌ ROBOFLOW_API_KEY not found in .env file.")
    exit()

# ====================== SETTINGS ======================
workspace_id  = "zeeshans-workspace-oosca"
project_id    = "my-first-project-oowzs"
frames_folder = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\Dataset_preparation\data\output_frames\IN_LANE_3_PTZ(2)_NVR_20260429094351_20260429114352_1872780"
batch_size    = 100     # number of images per batch
split         = "train" # train / valid / test
# ======================================================

# Initialize
rf      = Roboflow(api_key=api_key)
project = rf.workspace(workspace_id).project(project_id)


def collect_images(frames_folder):
    """Collect all jpg images from all subfolders."""
    all_images = []
    for root, dirs, files in os.walk(frames_folder):
        for file in files:
            if file.lower().endswith(".jpg"):
                all_images.append(os.path.join(root, file))
    return all_images


def chunk_list(lst, chunk_size):
    """Split a list into chunks of chunk_size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def upload_batch(project, batch_images, batch_name, batch_num,
                 total_batches, split, total_images):
    """Upload a single batch of images."""
    print(f"\n{'=' * 60}")
    print(f"📦 Batch {batch_num}/{total_batches}  —  {batch_name}  ({len(batch_images)} images)")
    print(f"{'=' * 60}")

    success = 0
    failed  = 0

    for i, image_path in enumerate(batch_images, start=1):
        # Global sequence number across ALL batches
        global_index = (batch_num - 1) * len(batch_images) + i

        print(f"  ⬆️  [{i}/{len(batch_images)}] {os.path.basename(image_path)}")
        try:
            project.upload(
                image_path        = image_path,
                batch_name        = batch_name,
                split             = split,
                num_retry_uploads = 3,
                tag_names         = ["vehicle", "plate"],
                sequence_number   = global_index,
                sequence_size     = total_images
            )
            success += 1
        except Exception as e:
            print(f"  ❌ Failed: {os.path.basename(image_path)} — {e}")
            failed += 1

    print(f"\n  ✅ Batch {batch_num} done — {success} uploaded, {failed} failed")
    return success, failed


def upload_all_in_batches(frames_folder, project, batch_size, split):
    """Main upload pipeline: collect → split into batches → upload."""
    all_images = collect_images(frames_folder)

    if not all_images:
        print(f"❌ No images found in: {frames_folder}")
        exit()

    total_images  = len(all_images)
    batches       = list(chunk_list(all_images, batch_size))
    total_batches = len(batches)

    print(f"📂 Found {total_images} images")
    print(f"📦 Splitting into {total_batches} batches of {batch_size} each\n")

    total_success = 0
    total_failed  = 0

    for batch_num, batch_images in enumerate(batches, start=1):
        batch_name = f"vehicle_plates_batch_{batch_num:03d}"   # e.g. vehicle_plates_batch_001

        success, failed = upload_batch(
            project, batch_images,
            batch_name, batch_num, total_batches,
            split, total_images
        )
        total_success += success
        total_failed  += failed

    print(f"\n{'=' * 60}")
    print(f"🏁 All batches uploaded!")
    print(f"   Total images   : {total_images}")
    print(f"   Total batches  : {total_batches}")
    print(f"   ✅ Successful  : {total_success}")
    print(f"   ❌ Failed      : {total_failed}")
    print(f"{'=' * 60}")


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    upload_all_in_batches(frames_folder, project, batch_size, split)