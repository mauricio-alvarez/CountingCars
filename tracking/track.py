"""Track data class with Kalman Filter for vehicle tracking."""

from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, Deque
import cv2
import numpy as np

from detection.detection import Detection


@dataclass
class Track:
    """Represents a tracked vehicle across multiple frames.
    
    Uses Kalman Filter for position prediction and maintains position history
    for direction analysis.
    
    Attributes:
        track_id: Unique identifier for this track
        kalman_filter: OpenCV Kalman Filter for state prediction
        position_history: Deque of recent positions (last 10)
        age: Number of frames since track creation
        hits: Number of times this track matched a detection
        time_since_update: Frames since last detection match
        state: Track state ('tentative', 'confirmed', 'deleted')
        direction_vector: Cached direction vector for visualization (dx, dy)
    """
    track_id: int
    kalman_filter: cv2.KalmanFilter
    position_history: Deque[Tuple[int, int, int, int]] = field(default_factory=lambda: deque(maxlen=10))
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = 'tentative'
    direction_vector: Tuple[float, float] = (0.0, 0.0)
    
    @staticmethod
    def create(track_id: int, detection: Detection) -> 'Track':
        """Create a new track from a detection.
        
        Args:
            track_id: Unique identifier for the new track
            detection: Initial detection to initialize the track
            
        Returns:
            New Track instance initialized with the detection
        """
        # Initialize Kalman Filter with 8-state vector [x, y, w, h, vx, vy, vw, vh]
        kf = cv2.KalmanFilter(8, 4)
        
        # State transition matrix (A)
        # Position updates with velocity: x_new = x + vx, y_new = y + vy, etc.
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix (H) - we only measure position [x, y, w, h]
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        # Process noise covariance (Q) - moderate noise for acceleration
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Measurement noise covariance (R) - based on detection uncertainty
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        # Error covariance matrix (P) - initial uncertainty
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        # Initialize state with detection position and zero velocity
        x, y, w, h = detection.bbox
        kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        
        # Create track instance
        track = Track(
            track_id=track_id,
            kalman_filter=kf,
            position_history=deque(maxlen=10),
            age=0,
            hits=1,
            time_since_update=0,
            state='tentative'
        )
        
        # Add initial position to history
        track.position_history.append(detection.bbox)
        
        return track
    
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next position using Kalman Filter.
        
        Returns:
            Predicted bounding box as (x, y, width, height)
        """
        # Perform Kalman prediction
        predicted_state = self.kalman_filter.predict()
        
        # Extract position from state vector
        x = int(predicted_state[0])
        y = int(predicted_state[1])
        w = int(predicted_state[2])
        h = int(predicted_state[3])
        
        return (x, y, w, h)
    
    def update(self, detection: Detection) -> None:
        """Update Kalman Filter with new detection.
        
        Args:
            detection: New detection to update the track with
        """
        # Convert detection to measurement vector
        x, y, w, h = detection.bbox
        measurement = np.array([x, y, w, h], dtype=np.float32).reshape(4, 1)
        
        # Update Kalman Filter
        self.kalman_filter.correct(measurement)
        
        # Update track metadata
        self.hits += 1
        self.time_since_update = 0
        
        # Add position to history
        self.position_history.append(detection.bbox)
        
        # Update direction vector for visualization
        self.direction_vector = self.get_direction_vector()
        
        # Update state to confirmed if enough hits
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def get_direction_vector(self) -> Tuple[float, float]:
        """Calculate direction vector from position history.
        
        Uses the last 5 positions (or fewer if not available) to calculate
        the movement direction.
        
        Returns:
            Normalized direction vector as (dx, dy), or (0, 0) if insufficient history
        """
        if len(self.position_history) < 2:
            return (0.0, 0.0)
        
        # Use last 5 positions or all available
        positions = list(self.position_history)[-5:]
        
        # Calculate center points
        centers = [(x + w/2, y + h/2) for x, y, w, h in positions]
        
        # Calculate direction from first to last position
        dx = centers[-1][0] - centers[0][0]
        dy = centers[-1][1] - centers[0][1]
        
        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude < 1e-6:  # Stationary or very slow
            return (0.0, 0.0)
        
        return (dx / magnitude, dy / magnitude)
    
    def get_current_bbox(self) -> Tuple[int, int, int, int]:
        """Get the current bounding box (last known or predicted).
        
        Returns:
            Current bounding box as (x, y, width, height)
        """
        if self.position_history:
            return self.position_history[-1]
        else:
            return self.predict()
