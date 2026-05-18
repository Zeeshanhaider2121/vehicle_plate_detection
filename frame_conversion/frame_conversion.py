import cv2
import os
import sys

# ====================== SETTINGS ======================
videos_folder = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\Dataset_preparation\data\videos"
output_root   = r"C:\Users\Admin\Downloads\projects\vehicle_plate_detection\Dataset_preparation\data\output_frames"
skip_seconds  = 0.5
# ======================================================

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v")


def validate_folders(videos_folder, output_root):
    """Check input folder exists and create output folder."""
    if not os.path.exists(videos_folder):
        print(f"❌ Videos folder not found: {videos_folder}")
        sys.exit()
    os.makedirs(output_root, exist_ok=True)


def collect_videos(videos_folder):
    """Return list of video filenames found in the folder."""
    video_files = [
        f for f in os.listdir(videos_folder)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not video_files:
        print(f"❌ No video files found in: {videos_folder}")
        sys.exit()
    print(f"📂 Found {len(video_files)} video(s) to process\n")
    print("=" * 60)
    return video_files


def open_video(video_path):
    """Open a video file and return (cap, fps, total_frames, duration) or None on failure."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  ⚠️  Could not open — skipping.")
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        print(f"  ⚠️  Bad FPS/frame count (possibly corrupt) — skipping.")
        cap.release()
        return None

    duration = total_frames / fps
    print(f"  ✅ FPS: {fps:.2f}  |  Frames: {total_frames}  |  Duration: {duration:.1f}s")
    return cap, fps, total_frames, duration


def extract_frames(cap, fps, duration, output_folder, skip_seconds):
    """Read through video and save one frame every skip_seconds."""
    os.makedirs(output_folder, exist_ok=True)

    frame_count     = 0
    saved_count     = 0
    last_saved_time = -skip_seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count      += 1
        current_time_sec  = frame_count / fps

        if current_time_sec >= last_saved_time + skip_seconds:
            last_saved_time = current_time_sec
            saved_count    += 1

            minutes  = int(current_time_sec // 60)
            seconds  = int(current_time_sec % 60)
            time_str = f"{minutes:02d}m_{seconds:02d}s"

            filename = os.path.join(output_folder, f"frame_{saved_count:04d}_{time_str}.jpg")
            cv2.imwrite(filename, frame)
            print(f"  💾 [{saved_count:04d}] {time_str}  (video frame #{frame_count})")

        # Progress ping every 60 seconds of video
        if frame_count % max(1, int(fps) * 60) == 0:
            print(f"  ⏳ Progress: {current_time_sec:.1f}s / {duration:.1f}s  ({saved_count} saved so far)")

    return saved_count


def process_video(video_index, total_videos, video_filename, videos_folder, output_root, skip_seconds):
    """Process a single video: open → extract frames → release."""
    video_path    = os.path.join(videos_folder, video_filename)
    video_name    = os.path.splitext(video_filename)[0]
    output_folder = os.path.join(output_root, video_name)

    print(f"\n[{video_index}/{total_videos}] 🎬 Processing: {video_filename}")

    result = open_video(video_path)
    if result is None:
        return 0

    cap, fps, total_frames, duration = result
    saved_count = extract_frames(cap, fps, duration, output_folder, skip_seconds)
    cap.release()

    print(f"  ✅ Done — {saved_count} frames saved → {output_folder}")
    return saved_count


def process_all_videos(videos_folder, output_root, skip_seconds):
    """Main pipeline: validate → collect → process all videos."""
    validate_folders(videos_folder, output_root)
    video_files = collect_videos(videos_folder)

    total_saved_all = 0

    for video_index, video_filename in enumerate(video_files, start=1):
        saved = process_video(
            video_index, len(video_files),
            video_filename,
            videos_folder, output_root,
            skip_seconds
        )
        total_saved_all += saved

    print("\n" + "=" * 60)
    print(f"🏁 All videos processed!")
    print(f"   Videos             : {len(video_files)}")
    print(f"   Total frames saved : {total_saved_all}")
    print(f"   Output root        : {output_root}")


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    process_all_videos(videos_folder, output_root, skip_seconds)