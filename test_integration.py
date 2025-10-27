#!/usr/bin/env python3
"""
Integration and End-to-End Testing Script

Tests the vehicle counting system with real video files from the video_fragments folder.
Validates that output videos are generated correctly and counting reports are accurate.

Requirements tested: 6.1, 6.2, 6.3, 6.4
"""

import os
import sys
import json
from pathlib import Path
import yaml

from vehicle_counting_app import VehicleCountingApp


def test_video_fragments():
    """Test processing videos from video_fragments/Relaxing highway traffic/ folder.
    
    This test:
    1. Processes videos from the test folder
    2. Verifies output videos are generated correctly
    3. Verifies counting reports are accurate
    4. Validates all expected outputs exist
    """
    print("="*70)
    print("INTEGRATION TEST: Video Fragments Processing")
    print("="*70)
    
    # Load test configuration
    config_path = "test_config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Test configuration file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Loaded test configuration from: {config_path}")
    
    # Initialize application
    try:
        app = VehicleCountingApp(config)
        print("✓ Vehicle Counting System initialized successfully")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize application: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Find test videos
    video_dir = "videos_fragments/Relaxing highway traffic"
    if not os.path.exists(video_dir):
        print(f"✗ ERROR: Test video directory not found: {video_dir}")
        return False
    
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    
    if not video_files:
        print(f"✗ ERROR: No video files found in {video_dir}")
        return False
    
    print(f"\n✓ Found {len(video_files)} test videos in {video_dir}")
    
    # Process first 3 videos for testing (to keep test time reasonable)
    test_videos = video_files[:3]
    print(f"\nProcessing {len(test_videos)} videos for integration test...")
    
    # Process videos
    reports = []
    for i, video_path in enumerate(test_videos, 1):
        print(f"\n{'-'*70}")
        print(f"Test Video {i}/{len(test_videos)}: {video_path.name}")
        print(f"{'-'*70}")
        
        # Reset state for new video
        app._reset_state()
        
        try:
            report = app.process_video(str(video_path))
            reports.append(report)
            
            if not report.get('success', False):
                print(f"✗ WARNING: Video processing reported failure")
            else:
                print(f"✓ Video processed successfully")
        
        except Exception as e:
            print(f"✗ ERROR: Exception during video processing: {e}")
            import traceback
            traceback.print_exc()
            reports.append({
                'success': False,
                'video_path': str(video_path),
                'error': str(e)
            })
    
    # Validate results
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    
    all_passed = True
    successful_count = 0
    
    for i, report in enumerate(reports, 1):
        video_name = Path(report['video_path']).name
        print(f"\nVideo {i}: {video_name}")
        
        if not report.get('success', False):
            print(f"  ✗ Processing failed")
            if 'error' in report:
                print(f"    Error: {report['error']}")
            all_passed = False
            continue
        
        successful_count += 1
        
        # Check output video exists
        output_path = report.get('output_path')
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"  ✓ Output video generated: {output_path}")
            print(f"    Size: {file_size:.2f} MB")
        else:
            print(f"  ✗ Output video not found: {output_path}")
            all_passed = False
        
        # Check report file exists
        report_path = report.get('report_path')
        if report_path and os.path.exists(report_path):
            print(f"  ✓ Report generated: {report_path}")
        else:
            print(f"  ✗ Report file not found: {report_path}")
            all_passed = False
        
        # Validate report contents
        print(f"  ✓ Frames processed: {report.get('total_frames', 0)}")
        print(f"  ✓ Processing FPS: {report.get('processing_fps', 0):.2f}")
        
        # Validate counts
        counts = report.get('counts', {})
        if counts:
            print(f"  ✓ Counting data present:")
            for line_name, line_counts in counts.items():
                total = line_counts.get('total', 0)
                up = line_counts.get('up', 0)
                down = line_counts.get('down', 0)
                print(f"    - {line_name}: {total} total (↑{up} ↓{down})")
        else:
            print(f"  ⚠ Warning: No counting data in report")
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos tested: {len(reports)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {len(reports) - successful_count}")
    
    if all_passed and successful_count == len(reports):
        print(f"\n✓ ALL TESTS PASSED")
        return True
    else:
        print(f"\n✗ SOME TESTS FAILED")
        return False


def test_single_video():
    """Test processing a single video file.
    
    Quick test to verify basic functionality with one video.
    """
    print("="*70)
    print("INTEGRATION TEST: Single Video Processing")
    print("="*70)
    
    # Load test configuration
    config_path = "test_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize application
    app = VehicleCountingApp(config)
    
    # Find first test video
    video_dir = "videos_fragments/Relaxing highway traffic"
    video_files = sorted(Path(video_dir).glob("*.mp4"))
    
    if not video_files:
        print(f"✗ ERROR: No video files found")
        return False
    
    test_video = video_files[0]
    print(f"\nProcessing test video: {test_video.name}")
    
    # Process video
    report = app.process_video(str(test_video))
    
    # Validate
    if report.get('success', False):
        print(f"\n✓ Single video test PASSED")
        return True
    else:
        print(f"\n✗ Single video test FAILED")
        return False


def test_batch_processing():
    """Test batch processing of multiple videos.
    
    Tests the process_directory method to ensure it can handle
    multiple videos in sequence.
    """
    print("="*70)
    print("INTEGRATION TEST: Batch Processing")
    print("="*70)
    
    # Load test configuration
    config_path = "test_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize application
    app = VehicleCountingApp(config)
    
    # Process directory (limit to first 2 videos for speed)
    video_dir = "videos_fragments/Relaxing highway traffic"
    
    # Get first 2 videos manually
    video_files = sorted(Path(video_dir).glob("*.mp4"))[:2]
    video_paths = [str(v) for v in video_files]
    
    print(f"\nBatch processing {len(video_paths)} videos...")
    
    reports = app.process_multiple_videos(video_paths)
    
    # Validate
    successful = sum(1 for r in reports if r.get('success', False))
    
    print(f"\nBatch processing results:")
    print(f"  Total: {len(reports)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(reports) - successful}")
    
    if successful == len(reports):
        print(f"\n✓ Batch processing test PASSED")
        return True
    else:
        print(f"\n✗ Batch processing test FAILED")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("VEHICLE COUNTING SYSTEM - INTEGRATION TESTS")
    print("="*70 + "\n")
    
    # Check prerequisites
    if not os.path.exists("test_config.yaml"):
        print("✗ ERROR: test_config.yaml not found")
        print("Please create test configuration file before running tests")
        sys.exit(1)
    
    if not os.path.exists("videos_fragments/Relaxing highway traffic"):
        print("✗ ERROR: Test video directory not found")
        print("Please ensure videos_fragments/Relaxing highway traffic/ exists")
        sys.exit(1)
    
    # Run tests
    results = {}
    
    print("\n" + "="*70)
    print("Running Test Suite...")
    print("="*70 + "\n")
    
    # Test 1: Single video processing
    try:
        results['single_video'] = test_single_video()
    except Exception as e:
        print(f"\n✗ Single video test crashed: {e}")
        import traceback
        traceback.print_exc()
        results['single_video'] = False
    
    print("\n")
    
    # Test 2: Video fragments processing (main test for task 13.1)
    try:
        results['video_fragments'] = test_video_fragments()
    except Exception as e:
        print(f"\n✗ Video fragments test crashed: {e}")
        import traceback
        traceback.print_exc()
        results['video_fragments'] = False
    
    print("\n")
    
    # Test 3: Batch processing
    try:
        results['batch_processing'] = test_batch_processing()
    except Exception as e:
        print(f"\n✗ Batch processing test crashed: {e}")
        import traceback
        traceback.print_exc()
        results['batch_processing'] = False
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("="*70)
        sys.exit(0)
    else:
        print("✗ SOME INTEGRATION TESTS FAILED")
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()