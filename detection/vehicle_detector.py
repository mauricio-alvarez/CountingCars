"""Vehicle detector using YOLO object detection."""

from typing import List
import numpy as np
from ultralytics import YOLO

from .detection import Detection


class VehicleDetector:
    """Detects vehicles in video frames using YOLO.
    
    This class wraps the YOLO model and provides vehicle-specific detection
    functionality with configurable confidence thresholds and class filtering.
    """
    
    # Vehicle class names that we want to detect
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """Initialize the vehicle detector.
        
        Args:
            model_path: Path to the YOLO model weights file
            confidence_threshold: Minimum confidence score for detections (default: 0.5)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect vehicles in a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of Detection objects for vehicles found in the frame
        """
        # Run YOLO inference on the frame
        results = self.model(frame, verbose=False)
        
        detections = []
        
        # Process results from the first (and only) image
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get detection data
                box = boxes.xyxy[i].cpu().numpy()  # Bounding box in xyxy format
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = result.names[class_id]
                
                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Filter to vehicle classes only
                if class_name not in self.VEHICLE_CLASSES:
                    continue
                
                # Convert from xyxy to xywh format
                x1, y1, x2, y2 = box
                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                # Create Detection object
                detection = Detection(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
        
        return detections
