"""Direction analysis utilities for vehicle tracking."""

from typing import Tuple, List
import numpy as np
from tracking.track import Track


class DirectionAnalyzer:
    """Analyzes movement direction of tracked vehicles.
    
    Uses position history to calculate direction vectors and compare
    movement directions between tracks. Useful for intersection handling
    and visualization.
    
    Attributes:
        history_length: Number of frames to use for direction calculation
    """
    
    def __init__(self, history_length: int = 5, use_smoothing: bool = True):
        """Initialize the direction analyzer.
        
        Args:
            history_length: Number of recent positions to use for direction
                          calculation (default: 5)
            use_smoothing: Whether to apply median filtering for erratic movement (default: True)
        """
        self.history_length = history_length
        self.use_smoothing = use_smoothing
    
    def calculate_direction(self, track: Track) -> Tuple[float, float]:
        """Calculate direction vector from track's position history.
        
        Uses linear regression on the position history for robust direction
        estimation. Falls back to simple vector calculation if insufficient
        data for regression. Applies median filtering for erratic movement
        if smoothing is enabled.
        
        Args:
            track: Track object with position history
            
        Returns:
            Normalized direction vector as (dx, dy). Returns (0.0, 0.0) if
            insufficient history or stationary vehicle.
        """
        # Handle insufficient history
        if len(track.position_history) < 2:
            return (0.0, 0.0)
        
        # Use last N positions or all available
        positions = list(track.position_history)[-self.history_length:]
        
        if len(positions) < 2:
            return (0.0, 0.0)
        
        # Calculate center points from bounding boxes
        centers = [(x + w/2, y + h/2) for x, y, w, h in positions]
        
        # Apply smoothing for erratic movement if enabled
        if self.use_smoothing and len(centers) >= 3:
            centers = self._apply_median_smoothing(centers)
        
        # Use linear regression for robust direction estimation
        if len(centers) >= 3:
            # Extract x and y coordinates
            x_coords = np.array([c[0] for c in centers])
            y_coords = np.array([c[1] for c in centers])
            
            # Time indices (frame numbers)
            t = np.arange(len(centers))
            
            # Linear regression: fit line to x(t) and y(t)
            # Using polyfit with degree 1 (linear)
            try:
                # Fit x = a*t + b
                x_coeffs = np.polyfit(t, x_coords, 1)
                # Fit y = c*t + d
                y_coeffs = np.polyfit(t, y_coords, 1)
                
                # Direction is the slope (velocity)
                dx = x_coeffs[0]
                dy = y_coeffs[0]
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to simple vector calculation
                dx = centers[-1][0] - centers[0][0]
                dy = centers[-1][1] - centers[0][1]
        else:
            # Simple vector calculation for 2 points
            dx = centers[-1][0] - centers[0][0]
            dy = centers[-1][1] - centers[0][1]
        
        # Normalize the direction vector
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Handle stationary or very slow movement
        if magnitude < 1e-6:
            return (0.0, 0.0)
        
        return (float(dx / magnitude), float(dy / magnitude))
    
    def _apply_median_smoothing(self, centers: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply median filtering to smooth erratic movement.
        
        Uses a sliding window of size 3 to apply median filtering to the
        position sequence, reducing the impact of sudden jumps or noise.
        
        Args:
            centers: List of (x, y) center positions
            
        Returns:
            Smoothed list of center positions
        """
        if len(centers) < 3:
            return centers
        
        smoothed = []
        
        # First point remains unchanged
        smoothed.append(centers[0])
        
        # Apply median filter to middle points
        for i in range(1, len(centers) - 1):
            # Get window of 3 points
            window_x = [centers[i-1][0], centers[i][0], centers[i+1][0]]
            window_y = [centers[i-1][1], centers[i][1], centers[i+1][1]]
            
            # Calculate median
            median_x = float(np.median(window_x))
            median_y = float(np.median(window_y))
            
            smoothed.append((median_x, median_y))
        
        # Last point remains unchanged
        smoothed.append(centers[-1])
        
        return smoothed
    
    def get_angle(self, direction: Tuple[float, float]) -> float:
        """Convert direction vector to angle in degrees.
        
        Calculates the angle from the positive x-axis (right direction)
        in the range [0, 360) degrees.
        
        Args:
            direction: Direction vector as (dx, dy)
            
        Returns:
            Angle in degrees [0, 360). Returns 0.0 for zero vector.
        """
        dx, dy = direction
        
        # Handle zero vector
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        
        # Calculate angle using atan2 (returns radians in range [-pi, pi])
        angle_rad = np.arctan2(dy, dx)
        
        # Convert to degrees and normalize to [0, 360)
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360.0
        
        return float(angle_deg)
    
    def compare_directions(self, dir1: Tuple[float, float], 
                          dir2: Tuple[float, float]) -> float:
        """Calculate angle difference between two direction vectors.
        
        Computes the absolute angle difference in the range [0, 180] degrees.
        This is useful for determining if two vehicles are moving in similar
        or different directions.
        
        Args:
            dir1: First direction vector as (dx, dy)
            dir2: Second direction vector as (dx, dy)
            
        Returns:
            Angle difference in degrees [0, 180]. Returns 0.0 if either
            direction is a zero vector.
        """
        # Get angles for both directions
        angle1 = self.get_angle(dir1)
        angle2 = self.get_angle(dir2)
        
        # Calculate absolute difference
        diff = abs(angle1 - angle2)
        
        # Normalize to [0, 180] (take the smaller angle)
        if diff > 180:
            diff = 360 - diff
        
        return float(diff)
