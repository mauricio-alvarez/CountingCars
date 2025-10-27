"""Visualizer class for rendering tracking information on video frames."""

from typing import List, Tuple
import cv2
import numpy as np

from detection.detection import Detection
from tracking.track import Track
from counting.counting_line import CountingLine
from counting.counting_zone import CountingZone


class Visualizer:
    """Handles visualization of detections, tracks, and counting lines on video frames.
    
    Provides methods to render bounding boxes, track IDs, direction arrows,
    and counting lines with counts on video frames for debugging and verification.
    
    Attributes:
        show_ids: Whether to display track IDs
        show_directions: Whether to display direction arrows
        show_trails: Whether to display track history trails
    """
    
    # Color definitions (BGR format for OpenCV)
    COLOR_DETECTION = (0, 255, 0)  # Green for detections
    COLOR_TENTATIVE = (0, 165, 255)  # Orange for tentative tracks
    COLOR_CONFIRMED = (0, 255, 0)  # Green for confirmed tracks
    COLOR_DELETED = (0, 0, 255)  # Red for deleted tracks
    COLOR_COUNTING_LINE = (255, 0, 255)  # Magenta for counting lines
    COLOR_COUNTING_ZONE = (255, 165, 0)  # Orange for counting zones
    COLOR_DIRECTION_ARROW = (255, 255, 0)  # Cyan for direction arrows
    COLOR_TEXT_BG = (0, 0, 0)  # Black background for text
    COLOR_TEXT_FG = (255, 255, 255)  # White foreground for text
    
    def __init__(self, show_ids: bool = True, show_directions: bool = True, show_trails: bool = False):
        """Initialize Visualizer with configuration options.
        
        Args:
            show_ids: Whether to display track IDs on bounding boxes
            show_directions: Whether to display direction arrows for tracks
            show_trails: Whether to display track history trails
        """
        self.show_ids = show_ids
        self.show_directions = show_directions
        self.show_trails = show_trails
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes for detections on the frame.
        
        Args:
            frame: Input video frame (will be modified in place)
            detections: List of Detection objects to visualize
            
        Returns:
            Modified frame with detection bounding boxes drawn
        """
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.COLOR_DETECTION, 2)
            
            # Draw confidence and class label
            label = f"{detection.class_name} {detection.confidence:.2f}"
            self._draw_label(frame, label, (x, y - 10))
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw track bounding boxes with IDs and direction arrows.
        
        Args:
            frame: Input video frame (will be modified in place)
            tracks: List of Track objects to visualize
            
        Returns:
            Modified frame with tracks, IDs, and direction arrows drawn
        """
        for track in tracks:
            # Get track color based on state
            if track.state == 'tentative':
                color = self.COLOR_TENTATIVE
            elif track.state == 'confirmed':
                color = self.COLOR_CONFIRMED
            elif track.state == 'deleted':
                color = self.COLOR_DELETED
            else:
                color = self.COLOR_CONFIRMED
            
            # Get current bounding box
            bbox = track.get_current_bbox()
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw track ID if enabled
            if self.show_ids:
                label = f"ID: {track.track_id}"
                self._draw_label(frame, label, (x, y - 10), color)
            
            # Draw direction arrow if enabled
            if self.show_directions and track.direction_vector != (0.0, 0.0):
                self._draw_direction_arrow(frame, bbox, track.direction_vector)
            
            # Draw track trail if enabled
            if self.show_trails and len(track.position_history) > 1:
                self._draw_track_trail(frame, track.position_history, color)
        
        return frame
    
    def draw_counting_lines(self, frame: np.ndarray, lines: List[CountingLine]) -> np.ndarray:
        """Draw counting lines with current count values.
        
        Args:
            frame: Input video frame (will be modified in place)
            lines: List of CountingLine objects to visualize
            
        Returns:
            Modified frame with counting lines and counts drawn
        """
        for line in lines:
            start_x, start_y = line.start_point
            end_x, end_y = line.end_point
            
            # Draw the counting line
            cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                    self.COLOR_COUNTING_LINE, 3)
            
            # Draw circles at endpoints
            cv2.circle(frame, (start_x, start_y), 5, self.COLOR_COUNTING_LINE, -1)
            cv2.circle(frame, (end_x, end_y), 5, self.COLOR_COUNTING_LINE, -1)
            
            # Calculate midpoint for label placement
            mid_x = (start_x + end_x) // 2
            mid_y = (start_y + end_y) // 2
            
            # Create count label
            total_count = line.get_total_count()
            label = f"{line.name}: {total_count}"
            if line.count_up > 0 or line.count_down > 0:
                label += f" (↑{line.count_up} ↓{line.count_down})"
            
            # Draw label with background
            self._draw_label(frame, label, (mid_x, mid_y - 20), self.COLOR_COUNTING_LINE)
        
        return frame
    
    def draw_counting_zones(self, frame: np.ndarray, zones: List[CountingZone]) -> np.ndarray:
        """Draw counting zones with current count values.
        
        Args:
            frame: Input video frame (will be modified in place)
            zones: List of CountingZone objects to visualize
            
        Returns:
            Modified frame with counting zones and counts drawn
        """
        for zone in zones:
            top_left_x, top_left_y = int(zone.top_left[0]), int(zone.top_left[1])
            bottom_right_x, bottom_right_y = int(zone.bottom_right[0]), int(zone.bottom_right[1])
            
            # Draw the counting zone rectangle (semi-transparent)
            overlay = frame.copy()
            cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
                         self.COLOR_COUNTING_ZONE, -1)
            # Blend with original frame for transparency
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Draw zone border
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
                         self.COLOR_COUNTING_ZONE, 3)
            
            # Calculate position for label
            label_x = top_left_x + 10
            label_y = top_left_y + 30
            
            # Create count label
            total_count = zone.get_total_count()
            inside_count = len(zone.tracks_inside)
            label = f"{zone.name}: {total_count} entered"
            if zone.count_entering > 0 or zone.count_exiting > 0:
                label += f" (→{zone.count_entering} ←{zone.count_exiting})"
            label += f" | Inside: {inside_count}"
            
            # Draw label with background
            self._draw_label(frame, label, (label_x, label_y), self.COLOR_COUNTING_ZONE)
        
        return frame
    
    def _draw_direction_arrow(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                             direction: Tuple[float, float], arrow_length: int = 40) -> None:
        """Draw direction arrow from track center.
        
        Args:
            frame: Input video frame (will be modified in place)
            bbox: Bounding box as (x, y, width, height)
            direction: Direction vector as (dx, dy)
            arrow_length: Length of the arrow in pixels
        """
        x, y, w, h = bbox
        
        # Calculate center of bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate arrow endpoint
        dx, dy = direction
        end_x = int(center_x + dx * arrow_length)
        end_y = int(center_y + dy * arrow_length)
        
        # Draw arrow
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                       self.COLOR_DIRECTION_ARROW, 2, tipLength=0.3)
    
    def _draw_track_trail(self, frame: np.ndarray, position_history: List[Tuple[int, int, int, int]], 
                         color: Tuple[int, int, int]) -> None:
        """Draw track history trail.
        
        Args:
            frame: Input video frame (will be modified in place)
            position_history: List of bounding boxes representing track history
            color: Color for the trail
        """
        if len(position_history) < 2:
            return
        
        # Extract center points from position history
        centers = [(x + w // 2, y + h // 2) for x, y, w, h in position_history]
        
        # Draw lines connecting centers
        for i in range(len(centers) - 1):
            # Fade older points
            alpha = (i + 1) / len(centers)
            trail_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, centers[i], centers[i + 1], trail_color, 2)
    
    def _draw_label(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int] = None) -> None:
        """Draw text label with background for readability.
        
        Args:
            frame: Input video frame (will be modified in place)
            text: Text to display
            position: Position for the label as (x, y)
            color: Optional color for the text (defaults to white)
        """
        x, y = position
        
        # Use default text color if not specified
        if color is None:
            color = self.COLOR_TEXT_FG
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Ensure label stays within frame bounds
        y = max(text_height + 5, y)
        x = max(0, min(x, frame.shape[1] - text_width - 10))
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (x - 2, y - text_height - 2), 
                     (x + text_width + 2, y + baseline + 2), 
                     self.COLOR_TEXT_BG, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
