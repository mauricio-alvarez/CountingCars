"""Main application for vehicle counting system."""

import os
import time
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np

from detection.vehicle_detector import VehicleDetector
from tracking.kalman_tracker import KalmanTracker
from utils.direction_analyzer import DirectionAnalyzer
from counting.vehicle_counter import VehicleCounter
from counting.counting_line import CountingLine
from counting.zone_counter import ZoneCounter
from counting.counting_zone import CountingZone
from visualization.visualizer import Visualizer
from utils.report import CountingReport, LineReport


class VehicleCountingApp:
    """Main application orchestrating vehicle detection, tracking, and counting.
    
    Coordinates all components of the vehicle counting system and manages
    video processing pipeline from input to output.
    
    Attributes:
        detector: VehicleDetector instance for detecting vehicles
        tracker: KalmanTracker instance for tracking vehicles
        direction_analyzer: DirectionAnalyzer for calculating movement directions
        counter: VehicleCounter for counting line crossings
        visualizer: Visualizer for rendering annotations
        config: Configuration dictionary
    """
    
    def __init__(self, config: Dict):
        """Initialize the vehicle counting application.
        
        Args:
            config: Configuration dictionary with all settings
        """
        self.config = config
        
        # Initialize detection module
        detection_config = config.get('detection', {})
        self.detector = VehicleDetector(
            model_path=detection_config.get('yolo_model_path', 'yolov8n.pt'),
            confidence_threshold=detection_config.get('confidence_threshold', 0.5)
        )
        
        # Initialize tracking module (frame dimensions will be set when processing video)
        tracking_config = config.get('tracking', {})
        self.tracker = KalmanTracker(
            max_age=tracking_config.get('max_track_age', 30),
            min_hits=tracking_config.get('min_track_hits', 3),
            iou_threshold=tracking_config.get('iou_threshold', 0.3),
            low_confidence_max_age=tracking_config.get('low_confidence_max_age', 15)
        )
        
        # Initialize direction analyzer
        direction_config = config.get('direction', {})
        self.direction_analyzer = DirectionAnalyzer(
            history_length=direction_config.get('history_length', 5)
        )
        
        # Initialize counting configuration (will be converted to absolute coordinates when processing video)
        counting_config = config.get('counting', {})
        
        # Determine counting mode: 'line' or 'zone'
        self.counting_mode = counting_config.get('mode', 'line')
        
        # Initialize counting lines
        counting_lines_data = counting_config.get('counting_lines', [])
        self.counting_lines_config = []
        if counting_lines_data:
            for line_data in counting_lines_data:
                line = CountingLine.from_config(line_data)
                self.counting_lines_config.append(line)
        
        # Initialize counting zones
        counting_zones_data = counting_config.get('counting_zones', [])
        self.counting_zones_config = []
        if counting_zones_data:
            for zone_data in counting_zones_data:
                zone = CountingZone.from_config(zone_data)
                self.counting_zones_config.append(zone)
        
        # Counter will be initialized with absolute coordinates when processing video
        self.counter = None
        self.zone_counter = None
        
        # Initialize visualizer
        viz_config = config.get('visualization', {})
        self.visualizer = Visualizer(
            show_ids=viz_config.get('show_track_ids', True),
            show_directions=viz_config.get('show_directions', True),
            show_trails=viz_config.get('show_trails', False)
        )
        
        # Statistics
        self.total_frames_processed = 0
        self.processing_start_time = None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame through the complete pipeline.
        
        Runs detection → tracking → direction analysis → counting → visualization
        on the input frame.
        
        Args:
            frame: Input video frame as numpy array (BGR format)
            
        Returns:
            Tuple of (annotated_frame, counts) where:
            - annotated_frame: Frame with visualizations drawn
            - counts: Dictionary of current counts per counting line
        """
        # Step 1: Detection - detect vehicles in frame
        detections = self.detector.detect(frame)
        
        # Step 2: Tracking - update tracks with new detections
        tracks = self.tracker.update(detections)
        
        # Step 3: Direction Analysis - calculate direction vectors for tracks
        for track in tracks:
            direction = self.direction_analyzer.calculate_direction(track)
            track.direction_vector = direction
        
        # Step 4: Counting - check for line crossings or zone entries and update counts
        counts = {}
        if self.counting_mode == 'zone' and self.zone_counter is not None:
            counts = self.zone_counter.update(tracks)
        elif self.counting_mode == 'line' and self.counter is not None:
            counts = self.counter.update(tracks)
        else:
            # Counter not initialized (no video processed yet)
            counts = {}
        
        # Step 5: Visualization - draw annotations on frame
        annotated_frame = frame.copy()
        
        viz_config = self.config.get('visualization', {})
        
        # Draw detections if enabled
        if viz_config.get('show_detections', True):
            annotated_frame = self.visualizer.draw_detections(annotated_frame, detections)
        
        # Draw tracks
        annotated_frame = self.visualizer.draw_tracks(annotated_frame, tracks)
        
        # Draw counting lines or zones if enabled
        if viz_config.get('show_counting_lines', True):
            if self.counting_mode == 'zone' and self.zone_counter is not None:
                annotated_frame = self.visualizer.draw_counting_zones(
                    annotated_frame,
                    self.zone_counter.counting_zones
                )
            elif self.counting_mode == 'line' and self.counter is not None:
                annotated_frame = self.visualizer.draw_counting_lines(
                    annotated_frame,
                    self.counter.counting_lines
                )
        
        self.total_frames_processed += 1
        
        return annotated_frame, counts

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> CountingReport:
        """Process a complete video file through the pipeline.
        
        Reads video file, processes each frame, writes annotated output video,
        and generates counting report.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path for output video (auto-generated if None)
            
        Returns:
            CountingReport object containing processing statistics and counts
        """
        print(f"Processing video: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            error_msg = f"File not found: {video_path}"
            print(f"Error: {error_msg}")
            return CountingReport(
                video_path=video_path,
                total_frames=0,
                processing_fps=0.0,
                success=False,
                error=error_msg
            )
        
        # Check if file is readable
        if not os.path.isfile(video_path):
            error_msg = f"Path is not a file: {video_path}"
            print(f"Error: {error_msg}")
            return CountingReport(
                video_path=video_path,
                total_frames=0,
                processing_fps=0.0,
                success=False,
                error=error_msg
            )
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            error_msg = f"Could not open video file (unsupported format or corrupted): {video_path}"
            print(f"Error: {error_msg}")
            return CountingReport(
                video_path=video_path,
                total_frames=0,
                processing_fps=0.0,
                success=False,
                error=error_msg
            )
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_properties = {
            'width': frame_width,
            'height': frame_height,
            'fps': fps,
            'total_frames': total_frames
        }
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")
        
        # Set frame dimensions in tracker for boundary detection
        self.tracker.frame_width = frame_width
        self.tracker.frame_height = frame_height
        
        # Initialize counting based on mode
        if self.counting_mode == 'zone':
            # Convert counting zones to absolute coordinates
            counting_zones_absolute = []
            for zone in self.counting_zones_config:
                absolute_zone = zone.to_absolute_coordinates(frame_width, frame_height)
                counting_zones_absolute.append(absolute_zone)
            
            # Initialize zone counter
            self.zone_counter = ZoneCounter(counting_zones_absolute)
            self.counter = None
            
            print(f"Initialized {len(counting_zones_absolute)} counting zone(s) [ZONE MODE]")
        else:
            # Convert counting lines to absolute coordinates
            counting_lines_absolute = []
            for line in self.counting_lines_config:
                absolute_line = line.to_absolute_coordinates(frame_width, frame_height)
                counting_lines_absolute.append(absolute_line)
            
            # Initialize line counter
            self.counter = VehicleCounter(counting_lines_absolute)
            self.zone_counter = None
            
            print(f"Initialized {len(counting_lines_absolute)} counting line(s) [LINE MODE]")
        
        # Setup output video writer if enabled
        output_config = self.config.get('output', {})
        video_writer = None
        write_failed = False
        
        if output_config.get('output_video', True):
            if output_path is None:
                # Auto-generate output path
                output_dir = output_config.get('output_directory', 'output')
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    print(f"Warning: Could not create output directory {output_dir}: {e}")
                    print("Continuing without video output...")
                    write_failed = True
                
                video_name = Path(video_path).stem
                output_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
            
            if not write_failed:
                # Create output directory if needed
                try:
                    output_dir = os.path.dirname(output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    print(f"Warning: Could not create output directory: {e}")
                    print("Continuing without video output...")
                    write_failed = True
            
            if not write_failed:
                # Initialize video writer with primary codec
                codec = output_config.get('video_codec', 'mp4v')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                if not video_writer.isOpened():
                    print(f"Warning: Could not open video writer with codec '{codec}'")
                    
                    # Try fallback codecs
                    fallback_codecs = ['avc1', 'H264', 'XVID', 'MJPG']
                    for fallback_codec in fallback_codecs:
                        if fallback_codec == codec:
                            continue
                        
                        print(f"Trying fallback codec: {fallback_codec}")
                        fourcc = cv2.VideoWriter_fourcc(*fallback_codec)
                        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                        
                        if video_writer.isOpened():
                            print(f"Successfully initialized video writer with codec '{fallback_codec}'")
                            break
                    
                    if not video_writer.isOpened():
                        print(f"Warning: All codec attempts failed. Continuing without video output...")
                        video_writer = None
                        write_failed = True
                else:
                    print(f"Output video: {output_path}")
        
        # Processing loop
        self.processing_start_time = time.time()
        frame_count = 0
        last_counts = {}
        
        # Get frame skip setting
        video_config = self.config.get('video', {})
        frame_skip = video_config.get('frame_skip', 0)
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if configured
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    continue
                
                # Process frame
                annotated_frame, counts = self.process_frame(frame)
                last_counts = counts
                
                # Write to output video
                if video_writer is not None and not write_failed:
                    try:
                        video_writer.write(annotated_frame)
                    except Exception as e:
                        if not write_failed:
                            print(f"Warning: Failed to write frame {frame_count}: {e}")
                            print("Continuing processing without video output...")
                            write_failed = True
                            if video_writer is not None:
                                video_writer.release()
                                video_writer = None
                
                # Display progress
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - self.processing_start_time
                    processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - "
                          f"Processing FPS: {processing_fps:.1f}")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error report
            elapsed_time = time.time() - self.processing_start_time if self.processing_start_time else 0
            processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            return CountingReport(
                video_path=video_path,
                total_frames=frame_count,
                processing_fps=processing_fps,
                video_properties=video_properties,
                processing_time_seconds=elapsed_time,
                output_path=output_path,
                success=False,
                error=str(e)
            )
        
        finally:
            # Clean up
            cap.release()
            if video_writer is not None:
                video_writer.release()
        
        # Calculate final statistics
        elapsed_time = time.time() - self.processing_start_time
        processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Print completion summary
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average processing FPS: {processing_fps:.2f}")
        
        # Print final counts
        print(f"\nFinal counts:")
        if last_counts:
            if self.counting_mode == 'zone':
                for zone_name, zone_counts in last_counts.items():
                    total = zone_counts.get('total', 0)
                    entering = zone_counts.get('entering', 0)
                    exiting = zone_counts.get('exiting', 0)
                    inside = zone_counts.get('inside', 0)
                    net = zone_counts.get('net', 0)
                    print(f"  {zone_name}: {total} vehicles entered (→{entering} ←{exiting}, inside:{inside}, net:{net})")
            else:
                for line_name, line_counts in last_counts.items():
                    total = line_counts.get('total', 0)
                    up = line_counts.get('up', 0)
                    down = line_counts.get('down', 0)
                    print(f"  {line_name}: {total} vehicles (↑{up} ↓{down})")
        else:
            print("  No vehicles counted")
        print(f"{'='*60}\n")
        
        # Generate report using the new method
        report = self.generate_report(
            video_path=video_path,
            output_path=output_path,
            frame_count=frame_count,
            elapsed_time=elapsed_time,
            processing_fps=processing_fps,
            video_properties=video_properties,
            counts=last_counts
        )
        
        # Save report if enabled
        if output_config.get('output_report', True):
            report_path = self._save_report_json(report, video_path)
            print(f"Report saved: {report_path}")
        
        return report

    def process_multiple_videos(self, video_paths: List[str]) -> List[CountingReport]:
        """Process multiple video files in batch.
        
        Processes each video sequentially and collects reports for all videos.
        Resets tracker state between videos.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            List of CountingReport objects, one per video
        """
        print(f"Batch processing {len(video_paths)} videos...")
        
        reports = []
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(video_paths)}")
            print(f"{'='*60}")
            
            # Reset state for new video
            self._reset_state()
            
            # Process video
            report = self.process_video(video_path)
            reports.append(report)
            
            # Check for errors
            if not report.success:
                print(f"Warning: Video processing failed for {video_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"{'='*60}")
        print(f"Total videos processed: {len(reports)}")
        successful = sum(1 for r in reports if r.success)
        print(f"Successful: {successful}")
        print(f"Failed: {len(reports) - successful}")
        
        return reports
    
    def process_directory(self, directory_path: str, 
                         extensions: List[str] = None) -> List[CountingReport]:
        """Process all video files in a directory.
        
        Scans directory for video files and processes them in batch.
        
        Args:
            directory_path: Path to directory containing video files
            extensions: List of video file extensions to process 
                       (default: ['.mp4', '.avi', '.mov', '.mkv'])
            
        Returns:
            List of CountingReport objects, one per video
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Find all video files in directory
        video_paths = []
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Error: Directory not found: {directory_path}")
            return []
        
        if not directory.is_dir():
            print(f"Error: Path is not a directory: {directory_path}")
            return []
        
        # Search for video files
        for ext in extensions:
            video_paths.extend(directory.glob(f"*{ext}"))
            video_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        # Convert to strings and sort
        video_paths = sorted([str(p) for p in video_paths])
        
        if not video_paths:
            print(f"No video files found in {directory_path}")
            print(f"Searched for extensions: {extensions}")
            return []
        
        print(f"Found {len(video_paths)} video files in {directory_path}")
        
        # Process all videos
        return self.process_multiple_videos(video_paths)
    
    def _reset_state(self) -> None:
        """Reset tracker and counter state for processing a new video."""
        # Reset tracker
        self.tracker.tracks = []
        self.tracker.next_track_id = 1
        self.tracker.frame_width = None
        self.tracker.frame_height = None
        
        # Reset counters (will be re-initialized with new video dimensions)
        if self.counter is not None:
            self.counter.reset_all_counts()
        if self.zone_counter is not None:
            self.zone_counter.reset_all_counts()
        self.counter = None
        self.zone_counter = None
        
        # Reset statistics
        self.total_frames_processed = 0
        self.processing_start_time = None
    
    def generate_report(
        self,
        video_path: str,
        output_path: Optional[str],
        frame_count: int,
        elapsed_time: float,
        processing_fps: float,
        video_properties: Dict,
        counts: Dict[str, Dict[str, int]]
    ) -> CountingReport:
        """Generate a CountingReport from processing statistics.
        
        Collects all statistics from video processing and creates a structured
        report with line-by-line counting details.
        
        Args:
            video_path: Path to input video file
            output_path: Path to output video file (if generated)
            frame_count: Total number of frames processed
            elapsed_time: Total processing time in seconds
            processing_fps: Average processing speed
            video_properties: Dictionary with video metadata
            counts: Dictionary of counts per line
            
        Returns:
            CountingReport object with all statistics
        """
        # Create LineReport objects for each counting line
        line_reports = {}
        for line_name, line_counts in counts.items():
            line_report = LineReport(
                line_name=line_name,
                total_count=line_counts.get('total', 0),
                count_by_direction={
                    'up': line_counts.get('up', 0),
                    'down': line_counts.get('down', 0)
                },
                count_by_vehicle_type={}  # Could be extended to track vehicle types
            )
            line_reports[line_name] = line_report
        
        # Create and return CountingReport
        report = CountingReport(
            video_path=video_path,
            total_frames=frame_count,
            processing_fps=processing_fps,
            counting_lines=line_reports,
            video_properties=video_properties,
            processing_time_seconds=elapsed_time,
            output_path=output_path,
            success=True
        )
        
        return report
    
    def _save_report_json(self, report: CountingReport, video_path: str) -> str:
        """Save CountingReport to JSON file.
        
        Args:
            report: CountingReport object to save
            video_path: Path to video file (used for naming report)
            
        Returns:
            Path to saved report file
        """
        output_config = self.config.get('output', {})
        output_dir = output_config.get('output_directory', 'output')
        
        # Create reports subdirectory
        reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate report filename
        video_name = Path(video_path).stem
        report_path = os.path.join(reports_dir, f"{video_name}_report.json")
        
        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save report to {report_path}: {e}")
        
        return report_path
