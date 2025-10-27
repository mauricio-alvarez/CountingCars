"""CountingZone class for zone-based vehicle counting."""

from dataclasses import dataclass, field
from typing import Tuple, Set, List


@dataclass
class CountingZone:
    """Represents a rectangular zone for counting vehicles.
    
    Counts vehicles as they enter or exit the zone. More reliable than
    line-based counting when multiple vehicles cross simultaneously.
    
    Supports both absolute pixel coordinates and relative coordinates (0.0-1.0).
    
    Attributes:
        name: Human-readable name for this counting zone
        top_left: Top-left corner of zone (x, y)
        bottom_right: Bottom-right corner of zone (x, y)
        count_entering: Count of vehicles entering the zone
        count_exiting: Count of vehicles exiting the zone
        tracks_inside: Set of track IDs currently inside the zone
        tracks_counted: Set of track IDs that have been counted (entered the zone)
        is_relative: Whether coordinates are relative (0.0-1.0) or absolute pixels
    """
    name: str
    top_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    count_entering: int = 0
    count_exiting: int = 0
    tracks_inside: Set[int] = field(default_factory=set)
    tracks_counted: Set[int] = field(default_factory=set)
    is_relative: bool = False
    
    def get_total_count(self) -> int:
        """Get total count of vehicles that entered the zone.
        
        Returns:
            Total number of vehicles that entered
        """
        return self.count_entering
    
    def get_net_count(self) -> int:
        """Get net count (entering - exiting).
        
        Returns:
            Net count (positive means more entering than exiting)
        """
        return self.count_entering - self.count_exiting
    
    def reset_counts(self) -> None:
        """Reset all counts and tracked vehicles."""
        self.count_entering = 0
        self.count_exiting = 0
        self.tracks_inside.clear()
        self.tracks_counted.clear()
    
    def is_point_inside(self, x: float, y: float) -> bool:
        """Check if a point is inside the zone.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside zone, False otherwise
        """
        return (self.top_left[0] <= x <= self.bottom_right[0] and
                self.top_left[1] <= y <= self.bottom_right[1])
    
    def to_absolute_coordinates(self, frame_width: int, frame_height: int) -> 'CountingZone':
        """Convert relative coordinates to absolute pixel coordinates.
        
        If coordinates are already absolute, returns a copy of self.
        
        Args:
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
            
        Returns:
            New CountingZone with absolute coordinates
        """
        if not self.is_relative:
            # Already absolute, return copy
            return CountingZone(
                name=self.name,
                top_left=self.top_left,
                bottom_right=self.bottom_right,
                count_entering=self.count_entering,
                count_exiting=self.count_exiting,
                tracks_inside=self.tracks_inside.copy(),
                tracks_counted=self.tracks_counted.copy(),
                is_relative=False
            )
        
        # Convert relative to absolute
        top_left_x = int(self.top_left[0] * frame_width)
        top_left_y = int(self.top_left[1] * frame_height)
        bottom_right_x = int(self.bottom_right[0] * frame_width)
        bottom_right_y = int(self.bottom_right[1] * frame_height)
        
        return CountingZone(
            name=self.name,
            top_left=(top_left_x, top_left_y),
            bottom_right=(bottom_right_x, bottom_right_y),
            count_entering=self.count_entering,
            count_exiting=self.count_exiting,
            tracks_inside=self.tracks_inside.copy(),
            tracks_counted=self.tracks_counted.copy(),
            is_relative=False
        )
    
    @staticmethod
    def from_config(config_dict: dict) -> 'CountingZone':
        """Create CountingZone from configuration dictionary.
        
        Automatically detects if coordinates are relative (0.0-1.0) or absolute.
        
        Args:
            config_dict: Dictionary with 'name', 'top_left', 'bottom_right' keys
            
        Returns:
            New CountingZone instance
        """
        name = config_dict['name']
        top_left = tuple(config_dict['top_left'])
        bottom_right = tuple(config_dict['bottom_right'])
        
        # Detect if coordinates are relative (all values between 0 and 1)
        is_relative = config_dict.get('relative', None)
        
        if is_relative is None:
            # Auto-detect based on coordinate values
            all_coords = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
            is_relative = all(0.0 <= coord <= 1.0 for coord in all_coords)
        
        return CountingZone(
            name=name,
            top_left=top_left,
            bottom_right=bottom_right,
            is_relative=is_relative
        )
