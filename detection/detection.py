"""Detection data class for vehicle detection results."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """Represents a single vehicle detection in a frame.
    
    Attributes:
        bbox: Bounding box coordinates as (x, y, width, height)
        confidence: Detection confidence score (0.0 to 1.0)
        class_id: Numeric class identifier
        class_name: Human-readable class name (e.g., 'car', 'truck')
    """
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    class_id: int
    class_name: str
