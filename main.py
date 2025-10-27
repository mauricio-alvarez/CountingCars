#!/usr/bin/env python3
"""
Vehicle Counting System - Main Entry Point

Command-line interface for processing videos with vehicle detection,
tracking, and counting capabilities.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml

from vehicle_counting_app import VehicleCountingApp


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Vehicle Counting System - Count vehicles in video footage using YOLO and Kalman Filter tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video with default config
  python main.py --video path/to/video.mp4
  
  # Process a video with custom config
  python main.py --video path/to/video.mp4 --config my_config.yaml
  
  # Process all videos in a directory
  python main.py --video-dir videos_fragments/
  
  # Process with custom output directory
  python main.py --video video.mp4 --output-dir results/
  
  # Disable visualization features for faster processing
  python main.py --video video.mp4 --no-show-ids --no-show-directions
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--video', '-v',
        type=str,
        help='Path to input video file'
    )
    input_group.add_argument(
        '--video-dir', '-d',
        type=str,
        help='Path to directory containing video files (processes all videos)'
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for annotated videos and reports (overrides config)'
    )
    parser.add_argument(
        '--no-output-video',
        action='store_true',
        help='Disable output video generation (only generate report)'
    )
    parser.add_argument(
        '--no-output-report',
        action='store_true',
        help='Disable report generation (only generate video)'
    )
    
    # Visualization toggles
    viz_group = parser.add_argument_group('visualization options')
    viz_group.add_argument(
        '--no-show-detections',
        action='store_true',
        help='Hide detection bounding boxes'
    )
    viz_group.add_argument(
        '--no-show-ids',
        action='store_true',
        help='Hide track ID numbers'
    )
    viz_group.add_argument(
        '--no-show-directions',
        action='store_true',
        help='Hide direction arrows'
    )
    viz_group.add_argument(
        '--show-trails',
        action='store_true',
        help='Show track history trails'
    )
    viz_group.add_argument(
        '--no-show-counting-lines',
        action='store_true',
        help='Hide counting lines'
    )
    
    # Detection options
    detection_group = parser.add_argument_group('detection options')
    detection_group.add_argument(
        '--model',
        type=str,
        help='Path to YOLO model file (overrides config)'
    )
    detection_group.add_argument(
        '--confidence',
        type=float,
        help='Detection confidence threshold 0.0-1.0 (overrides config)'
    )
    
    # Processing options
    processing_group = parser.add_argument_group('processing options')
    processing_group.add_argument(
        '--frame-skip',
        type=int,
        help='Process every Nth frame (0=all frames, overrides config)'
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line argument overrides to configuration.
    
    Args:
        config: Base configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Output directory override
    if args.output_dir:
        if 'output' not in config:
            config['output'] = {}
        config['output']['output_directory'] = args.output_dir
    
    # Output toggles
    if args.no_output_video:
        if 'output' not in config:
            config['output'] = {}
        config['output']['output_video'] = False
    
    if args.no_output_report:
        if 'output' not in config:
            config['output'] = {}
        config['output']['output_report'] = False
    
    # Visualization toggles
    if 'visualization' not in config:
        config['visualization'] = {}
    
    if args.no_show_detections:
        config['visualization']['show_detections'] = False
    
    if args.no_show_ids:
        config['visualization']['show_track_ids'] = False
    
    if args.no_show_directions:
        config['visualization']['show_directions'] = False
    
    if args.show_trails:
        config['visualization']['show_trails'] = True
    
    if args.no_show_counting_lines:
        config['visualization']['show_counting_lines'] = False
    
    # Detection options
    if args.model:
        if 'detection' not in config:
            config['detection'] = {}
        config['detection']['yolo_model_path'] = args.model
    
    if args.confidence is not None:
        if 'detection' not in config:
            config['detection'] = {}
        config['detection']['confidence_threshold'] = args.confidence
    
    # Processing options
    if args.frame_skip is not None:
        if 'video' not in config:
            config['video'] = {}
        config['video']['frame_skip'] = args.frame_skip
    
    return config


def main():
    """Main entry point for the vehicle counting system."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please create a configuration file or use --config to specify a different path")
        print(f"See example_config.yaml for reference")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Apply command-line overrides
    config = apply_cli_overrides(config, args)
    
    # Initialize application
    try:
        app = VehicleCountingApp(config)
        print("Vehicle Counting System initialized successfully")
    except Exception as e:
        print(f"Error initializing application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Process video(s)
    try:
        if args.video:
            # Process single video
            if not os.path.exists(args.video):
                print(f"Error: Video file not found: {args.video}")
                sys.exit(1)
            
            print(f"\n{'='*60}")
            print(f"Processing single video: {args.video}")
            print(f"{'='*60}\n")
            
            report = app.process_video(args.video)
            
            if report.success:
                print(f"\n✓ Video processing completed successfully")
                sys.exit(0)
            else:
                print(f"\n✗ Video processing failed")
                sys.exit(1)
        
        elif args.video_dir:
            # Process directory of videos
            if not os.path.exists(args.video_dir):
                print(f"Error: Directory not found: {args.video_dir}")
                sys.exit(1)
            
            if not os.path.isdir(args.video_dir):
                print(f"Error: Path is not a directory: {args.video_dir}")
                sys.exit(1)
            
            print(f"\n{'='*60}")
            print(f"Processing videos in directory: {args.video_dir}")
            print(f"{'='*60}\n")
            
            reports = app.process_directory(args.video_dir)
            
            if not reports:
                print(f"\n✗ No videos found or processed")
                sys.exit(1)
            
            # Check if any succeeded
            successful = sum(1 for r in reports if r.success)
            
            if successful > 0:
                print(f"\n✓ Batch processing completed: {successful}/{len(reports)} successful")
                sys.exit(0)
            else:
                print(f"\n✗ All video processing failed")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
