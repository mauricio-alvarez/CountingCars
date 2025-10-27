"""VehicleCounter class for detecting line crossings and maintaining counts."""

from typing import List, Dict, Optional, Tuple
from counting.counting_line import CountingLine
from tracking.track import Track


class VehicleCounter:
    """Manages vehicle counting across multiple counting lines.
    
    Detects when tracked vehicles cross configured counting lines and
    maintains counts per line with directional information.
    
    Attributes:
        counting_lines: List of CountingLine objects to monitor
        previous_positions: Dict mapping track_id to previous center position
    """
    
    def __init__(self, counting_lines: List[CountingLine]):
        """Initialize counter with configured counting lines.
        
        Args:
            counting_lines: List of CountingLine objects defining where to count
        """
        self.counting_lines = counting_lines
        self.previous_positions: Dict[int, Tuple[float, float]] = {}
    
    def update(self, tracks: List[Track]) -> Dict[str, Dict[str, int]]:
        """Update counts based on current tracks.
        
        Checks each track against each counting line to detect crossings.
        Updates counts and returns current count status.
        
        Args:
            tracks: List of active Track objects
            
        Returns:
            Dictionary mapping line names to count dictionaries with keys:
            'up', 'down', 'total'
        """
        # Get current positions for all tracks
        current_positions = {}
        for track in tracks:
            if track.state == 'confirmed':  # Only count confirmed tracks
                bbox = track.get_current_bbox()
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                current_positions[track.track_id] = (center_x, center_y)
        
        # Check for line crossings
        for track in tracks:
            if track.state != 'confirmed':
                continue
                
            track_id = track.track_id
            current_pos = current_positions.get(track_id)
            previous_pos = self.previous_positions.get(track_id)
            
            if current_pos and previous_pos:
                # Check crossing for each counting line
                for line in self.counting_lines:
                    self._check_line_crossing(track, line, previous_pos, current_pos)
        
        # Update previous positions for next frame
        self.previous_positions = current_positions
        
        # Return current counts
        return self._get_counts()
    
    def _check_line_crossing(
        self,
        track: Track,
        line: CountingLine,
        previous_pos: Tuple[float, float],
        current_pos: Tuple[float, float]
    ) -> Optional[str]:
        """Check if track crossed the counting line and update counts.
        
        Uses line segment intersection to detect crossing and cross product
        to determine direction.
        
        Args:
            track: The Track object being checked
            line: The CountingLine to check against
            previous_pos: Track center position in previous frame (x, y)
            current_pos: Track center position in current frame (x, y)
            
        Returns:
            Direction string ('up' or 'down') if crossing detected, None otherwise
        """
        # Skip if this track already crossed this line
        if track.track_id in line.crossed_tracks:
            return None
        
        # Check if line segments intersect
        if not self._segments_intersect(
            previous_pos, current_pos,
            line.start_point, line.end_point
        ):
            return None
        
        # Determine crossing direction using cross product
        direction = self._get_crossing_direction(
            previous_pos, current_pos,
            line.start_point, line.end_point
        )
        
        # Update counts
        if direction == 'up':
            line.count_up += 1
        elif direction == 'down':
            line.count_down += 1
        
        # Mark track as crossed to prevent double-counting
        line.crossed_tracks.add(track.track_id)
        
        return direction
    
    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> bool:
        """Check if two line segments intersect.
        
        Uses the orientation method to determine if segments (p1,p2) and (p3,p4)
        intersect.
        
        Args:
            p1: First point of first segment (x, y)
            p2: Second point of first segment (x, y)
            p3: First point of second segment (x, y)
            p4: Second point of second segment (x, y)
            
        Returns:
            True if segments intersect, False otherwise
        """
        def orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
            """Calculate orientation of ordered triplet (a, b, c).
            
            Returns:
                0 if collinear, 1 if clockwise, 2 if counterclockwise
            """
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else 2
        
        def on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
            """Check if point b lies on segment ac (assuming collinear)."""
            return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                    min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases (collinear points)
        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True
        
        return False
    
    def _get_crossing_direction(
        self,
        previous_pos: Tuple[float, float],
        current_pos: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> str:
        """Determine the direction of line crossing using cross product.
        
        The cross product determines which side of the line the movement is toward.
        
        Args:
            previous_pos: Track position before crossing (x, y)
            current_pos: Track position after crossing (x, y)
            line_start: Start point of counting line (x, y)
            line_end: End point of counting line (x, y)
            
        Returns:
            'up' if crossing from bottom/right to top/left, 'down' otherwise
        """
        # Vector from line start to line end
        line_vec_x = line_end[0] - line_start[0]
        line_vec_y = line_end[1] - line_start[1]
        
        # Vector from previous to current position
        movement_vec_x = current_pos[0] - previous_pos[0]
        movement_vec_y = current_pos[1] - previous_pos[1]
        
        # Cross product: positive means crossing "up/left", negative means "down/right"
        cross_product = line_vec_x * movement_vec_y - line_vec_y * movement_vec_x
        
        return 'up' if cross_product > 0 else 'down'
    
    def _get_counts(self) -> Dict[str, Dict[str, int]]:
        """Get current counts for all counting lines.
        
        Returns:
            Dictionary mapping line names to count dictionaries
        """
        counts = {}
        for line in self.counting_lines:
            counts[line.name] = {
                'up': line.count_up,
                'down': line.count_down,
                'total': line.get_total_count()
            }
        return counts
    
    def reset_all_counts(self) -> None:
        """Reset counts for all counting lines."""
        for line in self.counting_lines:
            line.reset_counts()
        self.previous_positions.clear()
