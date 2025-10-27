# Zone-Based Counting Upgrade

## Overview

This upgrade adds a new **zone-based counting mode** to address the issue where multiple vehicles crossing a line simultaneously could be missed by the traditional line-crossing detection.

## Problem Solved

**Issue**: When multiple vehicles cross a counting line at the same time, the line-based detection could miss some counts due to:
- Race conditions in line-crossing detection
- Simultaneous track updates
- Overlapping bounding boxes

**Solution**: Zone-based counting monitors a rectangular area and counts vehicles as they enter/exit the zone, handling multiple simultaneous vehicles correctly.

## New Features

### 1. Zone-Based Counting Mode
- Counts vehicles entering/exiting rectangular zones
- Handles multiple simultaneous vehicles reliably
- Tracks vehicles currently inside the zone
- Provides entering, exiting, total, and net counts

### 2. Dual Mode Support
The system now supports both counting modes:
- **Line mode**: Traditional line-crossing detection (backward compatible)
- **Zone mode**: New zone-based counting (recommended)

### 3. New Files Created
- `counting/counting_zone.py`: CountingZone data class
- `counting/zone_counter.py`: ZoneCounter implementation
- `test_zone_config.yaml`: Example configuration for zone mode

### 4. Enhanced Features
- Relative coordinate support for both lines and zones
- Auto-detection of coordinate type (relative vs absolute)
- Automatic scaling to different video resolutions
- Visual rendering of zones with transparency

## Configuration

### Enable Zone Mode

```yaml
counting:
  mode: "zone"  # Change from "line" to "zone"
  
  counting_zones:
    - name: "Main Counting Zone"
      top_left: [0.2, 0.4]      # 20% from left, 40% from top
      bottom_right: [0.8, 0.6]  # 80% from left, 60% from top
      relative: true            # Scales to any resolution
```

### Keep Line Mode (Default)

```yaml
counting:
  mode: "line"  # Traditional line-crossing
  
  counting_lines:
    - name: "Highway Lane 1"
      start_point: [100, 400]
      end_point: [700, 400]
```

## Usage

### Quick Test

```bash
# Test with zone-based counting
python main.py --video your_video.mp4 --config test_zone_config.yaml
```

### Output Differences

**Line Mode Output:**
```
Highway Lane 1: 45 vehicles (↑23 ↓22)
```

**Zone Mode Output:**
```
Main Counting Zone: 45 vehicles entered (→45 ←12, inside:8, net:33)
```

Where:
- `→45`: 45 vehicles entered the zone
- `←12`: 12 vehicles exited the zone
- `inside:8`: 8 vehicles currently in the zone
- `net:33`: Net count (45 - 12 = 33)

## When to Use Each Mode

### Use Line Mode When:
- Traffic is sparse (vehicles don't overlap)
- You need directional information (up/down, left/right)
- You have limited space for counting area
- Backward compatibility is required

### Use Zone Mode When:
- Multiple vehicles may cross simultaneously
- Traffic is dense or congested
- You want to track vehicles currently in an area
- You need more robust counting

## Technical Details

### Zone Detection Algorithm

1. For each frame, determine which tracks are inside each zone
2. Compare with previous frame to detect:
   - **Entering**: Tracks now inside but weren't before
   - **Exiting**: Tracks were inside but aren't now
3. Update counts and track sets
4. Only count each vehicle once (using `tracks_counted` set)

### Advantages Over Line Detection

- **No race conditions**: All tracks are evaluated independently
- **Simultaneous handling**: Multiple vehicles entering at once are all counted
- **Robust to noise**: Small position variations don't affect counts
- **State tracking**: Knows which vehicles are currently inside

## Backward Compatibility

All existing configurations continue to work:
- Default mode is "line" if not specified
- Existing line configurations work unchanged
- No breaking changes to API or output format

## Testing

Test both modes with the same video:

```bash
# Line mode
python main.py --video test.mp4 --config config.yaml

# Zone mode  
python main.py --video test.mp4 --config test_zone_config.yaml
```

Compare the counts to verify zone mode handles simultaneous crossings better.

## Migration Guide

To migrate from line-based to zone-based counting:

1. **Identify your counting line position**
   - Example: Horizontal line at y=400 from x=100 to x=700

2. **Convert to a zone around the line**
   ```yaml
   # Old line configuration
   counting_lines:
     - name: "Lane 1"
       start_point: [100, 400]
       end_point: [700, 400]
   
   # New zone configuration (50 pixels above/below line)
   counting_zones:
     - name: "Lane 1"
       top_left: [100, 350]
       bottom_right: [700, 450]
   ```

3. **Update mode in config**
   ```yaml
   counting:
     mode: "zone"  # Add this line
   ```

4. **Test and adjust zone size** if needed

## Performance

Zone-based counting has similar performance to line-based:
- Slightly more memory (tracks inside sets)
- Same computational complexity O(n*m) where n=tracks, m=zones
- No noticeable performance difference in practice
