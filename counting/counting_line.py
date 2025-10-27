"""CountingLine data class for vehicle counting."""

from dataclasses import dataclass, field
from typing import Tuple, Set


@dataclass
class CountingLine:
    """Represents a virtual line in the video frame for counting vehicles.
    
    When a vehicle track crosses this line, it triggers a count increment.
    The line tracks which vehicles have already crossed to prevent double-counting.
    
    Supports both absolute pixel coordinates and relative coordinates (0.0-1.0).
    
    Attributes:
        name: Human-readable name for this counting line
        start_point: Starting coordinate of the line (x, y)
        end_point: Ending coordinate of the line (x, y)
        count_up: Count of vehicles crossing in one direction
        count_down: Count of vehicles crossing in opposite direction
        crossed_tracks: Set of track IDs that have already crossed this line
        is_relative: Whether coordinates are relative (0.0-1.0) or absolute pixels
    """
    name: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    count_up: int = 0
    count_down: int = 0
    crossed_tracks: Set[int] = field(default_factory=set)
    is_relative: bool = False
    
    def get_total_count(self) -> int:
        """Get total count across both directions.
        
        Returns:
            Sum of count_up and count_down
        """
        return self.count_up + self.count_down
    
    def reset_counts(self) -> None:
        """Reset all counts and crossed tracks."""
        self.count_up = 0
        self.count_down = 0
        self.crossed_tracks.clear()
    
    def to_absolute_coordinates(self, frame_width: int, frame_height: int) -> 'CountingLine':
        """Convert relative coordinates to absolute pixel coordinates.
        
        If coordinates are already absolute, returns a copy of self.
        
        Args:
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
            
        Returns:
            New CountingLine with absolute coordinates
        """
        if not self.is_relative:
            # Already absolute, return copy
            return CountingLine(
                name=self.name,
                start_point=self.start_point,
                end_point=self.end_point,
                count_up=self.count_up,
                count_down=self.count_down,
                crossed_tracks=self.crossed_tracks.copy(),
                is_relative=False
            )
        
        # Convert relative to absolute
        start_x = int(self.start_point[0] * frame_width)
        start_y = int(self.start_point[1] * frame_height)
        end_x = int(self.end_point[0] * frame_width)
        end_y = int(self.end_point[1] * frame_height)
        
        return CountingLine(
            name=self.name,
            start_point=(start_x, start_y),
            end_point=(end_x, end_y),
            count_up=self.count_up,
            count_down=self.count_down,
            crossed_tracks=self.crossed_tracks.copy(),
            is_relative=False
        )
    
    @staticmethod
    def from_config(config_dict: dict) -> 'CountingLine':
        """Create CountingLine from configuration dictionary.
        
        Automatically detects if coordinates are relative (0.0-1.0) or absolute.
        
        Args:
            config_dict: Dictionary with 'name', 'start_point', 'end_point' keys
            
        Returns:
            New CountingLine instance
        """
        name = config_dict['name']
        start_point = tuple(config_dict['start_point'])
        end_point = tuple(config_dict['end_point'])
        
        # Detect if coordinates are relative (all values between 0 and 1)
        is_relative = config_dict.get('relative', None)
        
        if is_relative is None:
            # Auto-detect based on coordinate values
            all_coords = [start_point[0], start_point[1], end_point[0], end_point[1]]
            is_relative = all(0.0 <= coord <= 1.0 for coord in all_coords)
        
        return CountingLine(
            name=name,
            start_point=start_point,
            end_point=end_point,
            is_relative=is_relative
        )
