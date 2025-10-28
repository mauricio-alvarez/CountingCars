"""Report data classes for vehicle counting system."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LineReport:
    """Report for a single counting line.
    
    Contains counting statistics for one counting line including
    directional counts and vehicle type breakdowns.
    
    Attributes:
        line_name: Name of the counting line
        total_count: Total number of vehicles that crossed the line
        count_by_direction: Dictionary with 'up' and 'down' counts
        count_by_vehicle_type: Dictionary mapping vehicle types to counts
    """
    line_name: str
    total_count: int
    count_by_direction: Dict[str, int] = field(default_factory=dict)
    count_by_vehicle_type: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the line report
        """
        return {
            'line_name': self.line_name,
            'total_count': self.total_count,
            'count_by_direction': self.count_by_direction,
            'count_by_vehicle_type': self.count_by_vehicle_type
        }


@dataclass
class CountingReport:
    """Complete report for video processing.
    
    Contains all statistics from processing a video including frame counts,
    performance metrics, and counting results for all lines.
    
    Attributes:
        video_path: Path to the processed video file
        total_frames: Total number of frames processed
        processing_fps: Average processing speed in frames per second
        counting_lines: Dictionary mapping line names to LineReport objects
        video_properties: Optional dictionary with video metadata
        processing_time_seconds: Optional total processing time
        output_path: Optional path to output video file
        success: Whether processing completed successfully
        error: Optional error message if processing failed
    """
    video_path: str
    total_frames: int
    processing_fps: float
    counting_lines: Dict[str, LineReport] = field(default_factory=dict)
    video_properties: Optional[Dict] = None
    processing_time_seconds: Optional[float] = None
    output_path: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the counting report
        """
        report_dict = {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'processing_fps': self.processing_fps,
            'success': self.success,
            'counting_lines': {
                name: line_report.to_dict() 
                for name, line_report in self.counting_lines.items()
            }
        }
        
        if self.video_properties:
            report_dict['video_properties'] = self.video_properties
        
        if self.processing_time_seconds is not None:
            report_dict['processing_time_seconds'] = self.processing_time_seconds
        
        if self.output_path:
            report_dict['output_path'] = self.output_path
        
        if self.error:
            report_dict['error'] = self.error
        
        return report_dict
