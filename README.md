# Vehicle Counting System

Python-based vehicle counting using YOLO detection and Kalman Filter tracking. Supports both line-crossing and zone-based counting for accurate vehicle counts in video footage.

## Features

- YOLO vehicle detection with Kalman Filter tracking
- **Zone-based counting** (recommended): Handles multiple simultaneous vehicles
- **Line-based counting**: Traditional line-crossing detection
- Relative/absolute coordinate support (auto-scales to any resolution)
- Annotated output videos with JSON reports
- Robust error handling and edge case management

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a video (YOLO model downloads automatically on first run)
python main.py --video path/to/video.mp4 --config example_config.yaml
```

## Reproducing Zone-Based Counting Results

To reproduce the improved zone-based counting on segment_001.mp4:

```bash
# Run with zone-based counting configuration
python main.py --video "videos_fragments/Relaxing highway traffic/segment_001.mp4" --config segment_001_zone_config.yaml
```

**Expected Results:**
- 80 vehicles detected (vs 52 with line-based counting)
- 54% accuracy improvement over line-crossing method
- Output: `output/zone_test/segment_001_annotated.mp4`

See `SEGMENT_001_COMPARISON.md` for detailed analysis.

## Usage

```bash
# Single video
python main.py --video path/to/video.mp4

# Directory of videos
python main.py --video-dir videos_fragments/

# Custom configuration
python main.py --video video.mp4 --config my_config.yaml

# Custom output directory
python main.py --video video.mp4 --output-dir results/
```

**Common Options:**
- `--config`: Configuration file (default: `config.yaml`)
- `--output-dir`: Output directory
- `--no-output-video`: Skip video generation (report only)
- `--frame-skip N`: Process every Nth frame for faster processing

## Configuration

Create a YAML configuration file (see `example_config.yaml` for full template):

```yaml
detection:
  yolo_model_path: "yolov8n.pt"
  confidence_threshold: 0.5

tracking:
  max_track_age: 30
  min_track_hits: 3
  iou_threshold: 0.3

counting:
  mode: "zone"  # "zone" or "line"
  
  counting_zones:
    - name: "Main Zone"
      top_left: [0.2, 0.4]      # Relative coords (0.0-1.0)
      bottom_right: [0.8, 0.6]
      relative: true

output:
  output_video: true
  output_directory: "output"
```

## Counting Modes

### Zone-Based (Recommended)
Counts vehicles entering/exiting a rectangular area. Best for dense traffic and simultaneous crossings.

```yaml
counting:
  mode: "zone"
  counting_zones:
    - name: "Highway Zone"
      top_left: [0.2, 0.45]      # 20% from left, 45% from top
      bottom_right: [0.8, 0.55]  # 80% from left, 55% from top
      relative: true
```

**Adjusting Zone Size and Position:**

```yaml
# Horizontal band across middle (highway traffic)
top_left: [0.1, 0.45]      # Start 10% from left, 45% from top
bottom_right: [0.9, 0.55]  # End 90% from left, 55% from top

# Vertical band on left side (side road)
top_left: [0.1, 0.2]       # Start 10% from left, 20% from top
bottom_right: [0.3, 0.8]   # End 30% from left, 80% from top

# Small zone in center (intersection)
top_left: [0.4, 0.4]       # Start 40% from left, 40% from top
bottom_right: [0.6, 0.6]   # End 60% from left, 60% from top
```

**Coordinate Guide:**
- `[0.0, 0.0]` = Top-left corner
- `[1.0, 1.0]` = Bottom-right corner
- `[0.5, 0.5]` = Center of frame
- Relative coords auto-scale to any video resolution

### Line-Based (Traditional)
Counts vehicles crossing a line. Good for sparse traffic.

```yaml
counting:
  mode: "line"
  counting_lines:
    - name: "Lane 1"
      start_point: [100, 400]  # Absolute pixels
      end_point: [700, 400]
```

## Output

**Annotated Video:** `output/[video_name]_annotated.mp4`
- Bounding boxes, track IDs, direction arrows
- Counting zones/lines with real-time counts

**JSON Report:** `output/reports/[video_name]_report.json`
```json
{
  "success": true,
  "total_frames": 1800,
  "processing_fps": 18.0,
  "counts": {
    "Highway Zone": {
      "total": 80,
      "entering": 80,
      "exiting": 79,
      "inside": 2
    }
  }
}
```

## YOLO Models

Models download automatically on first run. For better accuracy or speed:

```yaml
detection:
  yolo_model_path: "yolov8n.pt"  # Fastest
  # yolo_model_path: "yolov8m.pt"  # Balanced
  # yolo_model_path: "yolov8x.pt"  # Most accurate
```

Download from: https://github.com/ultralytics/ultralytics

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Project Structure

```
├── detection/          # YOLO vehicle detection
├── tracking/           # Kalman Filter tracking
├── counting/           # Zone/line counting logic
├── visualization/      # Video annotation
├── utils/              # Helper functions
├── main.py             # CLI entry point
└── config.yaml         # Configuration
```

## Documentation

- `example_config.yaml` - Full configuration template
- `ZONE_COUNTING_UPGRADE.md` - Zone counting implementation details
- `SEGMENT_001_COMPARISON.md` - Performance comparison (line vs zone)
