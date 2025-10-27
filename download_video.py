import yt_dlp
import math
import os
import subprocess

# ------------------ SETTINGS ------------------
video_url = "https://www.youtube.com/watch?v=nt3D26lrkho&list=PLJKyZ_NuOhJQzif2-6-Kq9OiOj_UjJWvi5"
segment_duration = 60  # seconds per clip
output_folder = "videos_fragments"
# ------------------------------------------------


def download_videos(url, output_dir):
    """Download video(s) or playlist using yt-dlp."""
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "format": "bestvideo[height<=720]+bestaudio/best",
        "merge_output_format": "mp4",
    }

    downloaded_files = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # If it's a playlist
        if "entries" in info:
            for entry in info["entries"]:
                if entry:
                    filename = ydl.prepare_filename(entry)
                    filename = os.path.splitext(filename)[0] + ".mp4"
                    downloaded_files.append((entry["title"], filename))
        else:
            filename = ydl.prepare_filename(info)
            filename = os.path.splitext(filename)[0] + ".mp4"
            downloaded_files.append((info["title"], filename))

    return downloaded_files


def get_video_duration(filename):
    """Get duration (seconds) using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", filename
            ],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {filename}: {e}")
        return None


def get_audio_codec(filename):
    """Return audio codec name (e.g., 'opus', 'aac', 'mp3')."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1", filename
            ],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def split_video(filename, output_dir, segment_duration):
    """Split a local video into N-second segments using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_duration(filename)
    if not duration:
        print(f"Skipping {filename}, could not determine duration.")
        return

    num_segments = math.ceil(duration / segment_duration)
    print(f"\nSplitting '{filename}' into {num_segments} segments...")

    audio_codec = get_audio_codec(filename)
    print(f"Detected audio codec: {audio_codec or 'unknown'}")

    for i in range(num_segments):
        start_time = i * segment_duration
        output_name = f"segment_{i+1:03d}.mp4"
        output_path = os.path.join(output_dir, output_name)

        # Build ffmpeg command
        if audio_codec and audio_codec.lower() == "opus":
            # Convert audio to AAC for MP4 compatibility
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-t", str(segment_duration),
                "-i", filename, "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", output_path
            ]
        else:
            # Copy both streams directly
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-t", str(segment_duration),
                "-i", filename, "-c", "copy", output_path
            ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  → Created {output_name}")

    # Delete the original full-length file
    try:
        os.remove(filename)
        print(f"🗑️  Deleted original video: {filename}")
    except Exception as e:
        print(f"Could not delete {filename}: {e}")

    print(f"✅ Finished splitting '{filename}'.")


def main():
    print("📥 Downloading video(s)...")
    videos = download_videos(video_url, output_folder)

    for title, filepath in videos:
        title_safe = "".join(c if c.isalnum() or c in " -_()" else "_" for c in title)
        video_output_dir = os.path.join(output_folder, title_safe)
        split_video(filepath, video_output_dir, segment_duration)

    print("\n🎉 All downloads and segmentations complete!")


if __name__ == "__main__":
    main()

