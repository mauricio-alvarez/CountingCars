"""Intersection over Union (IoU) calculation utility."""

from typing import Tuple


def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: First bounding box as (x, y, width, height)
        bbox2: Second bounding box as (x, y, width, height)
        
    Returns:
        IoU value between 0.0 and 1.0, or 0.0 for invalid/non-overlapping boxes
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Handle invalid boxes (zero or negative dimensions)
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0
    
    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Check if there is no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Handle edge case where union is zero (shouldn't happen with valid boxes)
    if union_area <= 0:
        return 0.0
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou
