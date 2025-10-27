"""Kalman Filter-based tracker with Hungarian algorithm for data association."""

from typing import List, Dict, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection.detection import Detection
from .track import Track
from .iou import calculate_iou
from utils.direction_analyzer import DirectionAnalyzer


class KalmanTracker:
    """Tracks vehicles across frames using Kalman Filter and IoU matching.
    
    Uses the Hungarian algorithm for optimal detection-to-track assignment
    and maintains tracks through occlusions using Kalman predictions.
    
    Attributes:
        max_age: Maximum frames a track can exist without detection before deletion
        min_hits: Minimum detection matches required to confirm a track
        iou_threshold: Minimum IoU for matching detection to track
        tracks: List of active Track objects
        next_track_id: Counter for assigning unique track IDs
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3,
                 low_confidence_max_age: int = 15, frame_width: int = None, frame_height: int = None):
        """Initialize the Kalman tracker.
        
        Args:
            max_age: Maximum frames without detection before track deletion (default: 30)
            min_hits: Minimum hits to confirm a track (default: 3)
            iou_threshold: Minimum IoU for detection-track matching (default: 0.3)
            low_confidence_max_age: Maximum frames to continue tracking with low confidence (default: 15)
            frame_width: Video frame width for boundary detection (optional)
            frame_height: Video frame height for boundary detection (optional)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.low_confidence_max_age = low_confidence_max_age
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.direction_analyzer = DirectionAnalyzer(history_length=5)
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections.
        
        Performs the following steps:
        1. Predict positions for all active tracks
        2. Associate detections with tracks using IoU and Hungarian algorithm
        3. Update matched tracks with detections
        4. Create new tracks for unmatched detections
        5. Mark old tracks for deletion
        
        Args:
            detections: List of Detection objects from current frame
            
        Returns:
            List of active Track objects (confirmed tracks only)
        """
        # Predict next positions for all tracks
        for track in self.tracks:
            track.predict()
            track.age += 1
            track.time_since_update += 1
        
        # Associate detections to tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            new_track = Track.create(self.next_track_id, detections[detection_idx])
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Mark tracks for deletion if they haven't been updated for too long
        tracks_to_keep = []
        for track in self.tracks:
            # Check if track has exited the frame
            if self._is_track_outside_frame(track):
                track.state = 'deleted'
                continue
            
            # Determine max age based on track confidence
            # For tracks with low confidence or detection gaps, use shorter max age
            max_age_threshold = self.max_age
            
            # If track has been without detection for a while, check if it's a low confidence case
            if track.time_since_update > self.low_confidence_max_age:
                # This could be due to low confidence detections or lighting issues
                # Use the low_confidence_max_age threshold
                max_age_threshold = self.low_confidence_max_age
            
            # Delete tracks that are too old
            if track.time_since_update > max_age_threshold:
                track.state = 'deleted'
            else:
                tracks_to_keep.append(track)
        
        self.tracks = tracks_to_keep
        
        # Return only confirmed tracks
        return [track for track in self.tracks if track.state == 'confirmed']
    
    def _associate_detections_to_tracks(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to existing tracks using IoU matching.
        
        Uses the Hungarian algorithm (linear_sum_assignment) for optimal assignment
        based on IoU cost matrix. Applies directional heuristics during intersections
        to maintain correct track associations.
        
        Args:
            detections: List of Detection objects from current frame
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks) where:
            - matches: List of (track_idx, detection_idx) pairs
            - unmatched_detections: List of detection indices with no match
            - unmatched_tracks: List of track indices with no match
        """
        if len(self.tracks) == 0:
            # No tracks exist, all detections are unmatched
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            # No detections, all tracks are unmatched
            return [], [], list(range(len(self.tracks)))
        
        # Detect intersections between tracks
        intersection_pairs = self._detect_intersections()
        
        # Build IoU cost matrix
        # Rows = tracks, Columns = detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        
        for t, track in enumerate(self.tracks):
            predicted_bbox = track.predict()
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = calculate_iou(predicted_bbox, detection.bbox)
        
        # Convert IoU to cost (higher IoU = lower cost)
        cost_matrix = 1.0 - iou_matrix
        
        # Apply intersection handling: adjust costs based on directional similarity
        for track_idx1, track_idx2 in intersection_pairs:
            track1 = self.tracks[track_idx1]
            track2 = self.tracks[track_idx2]
            
            # Check if either track has high IoU with any detection (> 0.5)
            high_iou_detections = set()
            for d in range(len(detections)):
                if iou_matrix[track_idx1, d] > 0.5 or iou_matrix[track_idx2, d] > 0.5:
                    high_iou_detections.add(d)
            
            if high_iou_detections:
                # Apply directional heuristic
                intersection_info = self._handle_intersection(track1, track2)
                
                if intersection_info['prioritize_prediction']:
                    # Prioritize Kalman prediction over IoU matching
                    # Increase cost for ambiguous matches to favor prediction continuity
                    for d in high_iou_detections:
                        # Calculate direction similarity between track and detection
                        # For tracks with similar directions, penalize cross-matching
                        
                        # Get predicted positions
                        pred_bbox1 = track1.predict()
                        pred_bbox2 = track2.predict()
                        det_bbox = detections[d].bbox
                        
                        # Calculate distances from predictions to detection
                        dist1 = self._bbox_distance(pred_bbox1, det_bbox)
                        dist2 = self._bbox_distance(pred_bbox2, det_bbox)
                        
                        # Adjust costs to favor closer prediction
                        # Add penalty proportional to distance difference
                        if dist1 < dist2:
                            # Detection is closer to track1's prediction
                            # Increase cost for track2 matching this detection
                            cost_matrix[track_idx2, d] += 0.3
                        else:
                            # Detection is closer to track2's prediction
                            # Increase cost for track1 matching this detection
                            cost_matrix[track_idx1, d] += 0.3
        
        # Use Hungarian algorithm for optimal assignment
        # linear_sum_assignment minimizes the total cost
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by IoU threshold
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            iou = iou_matrix[track_idx, detection_idx]
            if iou >= self.iou_threshold:
                # Valid match
                matches.append((track_idx, detection_idx))
                unmatched_detections.remove(detection_idx)
                unmatched_tracks.remove(track_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _bbox_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Euclidean distance between centers of two bounding boxes.
        
        Args:
            bbox1: First bounding box as (x, y, width, height)
            bbox2: Second bounding box as (x, y, width, height)
            
        Returns:
            Euclidean distance between box centers
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate centers
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        return float(distance)
    
    def _detect_intersections(self) -> List[Tuple[int, int]]:
        """Detect intersections between tracks based on IoU.
        
        Identifies pairs of tracks that have overlapping bounding boxes with
        IoU greater than 0.3, indicating a potential intersection event.
        
        Returns:
            List of (track_idx1, track_idx2) pairs representing intersecting tracks
        """
        intersection_pairs = []
        
        # Check all pairs of tracks for intersection
        for i in range(len(self.tracks)):
            for j in range(i + 1, len(self.tracks)):
                track1 = self.tracks[i]
                track2 = self.tracks[j]
                
                # Get current bounding boxes
                bbox1 = track1.get_current_bbox()
                bbox2 = track2.get_current_bbox()
                
                # Calculate IoU
                iou = calculate_iou(bbox1, bbox2)
                
                # If IoU > 0.3, consider it an intersection
                if iou > 0.3:
                    intersection_pairs.append((i, j))
        
        return intersection_pairs
    
    def _is_track_outside_frame(self, track: Track) -> bool:
        """Check if a track has exited the frame boundaries.
        
        Args:
            track: Track to check
            
        Returns:
            True if track is completely outside frame boundaries, False otherwise
        """
        # If frame dimensions not set, cannot determine if outside
        if self.frame_width is None or self.frame_height is None:
            return False
        
        # Get current bounding box
        bbox = track.get_current_bbox()
        x, y, w, h = bbox
        
        # Calculate bounding box boundaries
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
        
        # Define margin for considering track as "exited" (10% of frame size)
        margin_x = self.frame_width * 0.1
        margin_y = self.frame_height * 0.1
        
        # Check if track is completely outside frame with margin
        if x_max < -margin_x or x_min > self.frame_width + margin_x:
            return True
        if y_max < -margin_y or y_min > self.frame_height + margin_y:
            return True
        
        return False
    
    def _handle_intersection(self, track1: Track, track2: Track) -> Dict[str, any]:
        """Apply directional heuristic for intersection resolution.
        
        Analyzes the direction vectors of two intersecting tracks to determine
        if they should maintain separate identities or if one should be prioritized.
        
        Args:
            track1: First intersecting track
            track2: Second intersecting track
            
        Returns:
            Dictionary with intersection handling information:
            - 'maintain_separate': bool, True if tracks should keep separate IDs
            - 'angle_difference': float, angle difference in degrees
            - 'prioritize_prediction': bool, True if Kalman prediction should be prioritized
        """
        # Calculate direction vectors for both tracks
        dir1 = self.direction_analyzer.calculate_direction(track1)
        dir2 = self.direction_analyzer.calculate_direction(track2)
        
        # Compare angle difference between directions
        angle_diff = self.direction_analyzer.compare_directions(dir1, dir2)
        
        # Decision logic based on angle difference
        if angle_diff > 45.0:
            # Tracks moving in significantly different directions
            # Maintain separate track IDs
            return {
                'maintain_separate': True,
                'angle_difference': angle_diff,
                'prioritize_prediction': False
            }
        else:
            # Tracks moving in similar directions
            # Prioritize Kalman prediction over IoU matching
            return {
                'maintain_separate': False,
                'angle_difference': angle_diff,
                'prioritize_prediction': True
            }
