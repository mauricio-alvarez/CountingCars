"""ZoneCounter class for zone-based vehicle counting."""

from typing import List, Dict, Tuple
from counting.counting_zone import CountingZone
from tracking.track import Track


class ZoneCounter:
    """Manages vehicle counting using zones instead of lines.
    
    Counts vehicles as they enter or exit defined zones. More reliable
    than line-based counting when multiple vehicles cross simultaneously.
    
    Attributes:
        counting_zones: List of CountingZone objects to monitor
    """
    
    def __init__(self, counting_zones: List[CountingZone]):
        """Initialize counter with configured counting zones.
        
        Args:
            counting_zones: List of CountingZone objects defining where to count
        """
        self.counting_zones = counting_zones
    
    def update(self, tracks: List[Track]) -> Dict[str, Dict[str, int]]:
        """Update counts based on current tracks.
        
        Checks each track against each counting zone to detect entries/exits.
        Updates counts and returns current count status.
        
        Args:
            tracks: List of active Track objects
            
        Returns:
            Dictionary mapping zone names to count dictionaries with keys:
            'entering', 'exiting', 'total', 'net', 'inside'
        """
        # Get current track IDs and positions
        current_track_ids = set()
        track_positions = {}
        
        for track in tracks:
            if track.state == 'confirmed':  # Only count confirmed tracks
                current_track_ids.add(track.track_id)
                bbox = track.get_current_bbox()
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                track_positions[track.track_id] = (center_x, center_y)
        
        # Check each zone
        for zone in self.counting_zones:
            # Find tracks currently inside the zone
            tracks_now_inside = set()
            for track_id, (x, y) in track_positions.items():
                if zone.is_point_inside(x, y):
                    tracks_now_inside.add(track_id)
            
            # Detect entering vehicles (now inside but weren't before)
            entering = tracks_now_inside - zone.tracks_inside
            for track_id in entering:
                if track_id not in zone.tracks_counted:
                    zone.count_entering += 1
                    zone.tracks_counted.add(track_id)
            
            # Detect exiting vehicles (were inside but aren't now)
            exiting = zone.tracks_inside - tracks_now_inside
            for track_id in exiting:
                if track_id in zone.tracks_counted:
                    zone.count_exiting += 1
            
            # Update zone's current tracks
            zone.tracks_inside = tracks_now_inside
        
        # Return current counts
        return self._get_counts()
    
    def _get_counts(self) -> Dict[str, Dict[str, int]]:
        """Get current counts for all counting zones.
        
        Returns:
            Dictionary mapping zone names to count dictionaries
        """
        counts = {}
        for zone in self.counting_zones:
            counts[zone.name] = {
                'entering': zone.count_entering,
                'exiting': zone.count_exiting,
                'total': zone.get_total_count(),
                'net': zone.get_net_count(),
                'inside': len(zone.tracks_inside)
            }
        return counts
    
    def reset_all_counts(self) -> None:
        """Reset counts for all counting zones."""
        for zone in self.counting_zones:
            zone.reset_counts()
