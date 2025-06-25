"""
Utility functions for fixed aspect ratio crop calculations.

This module provides pure functions for parsing and calculating fixed aspect ratio
crops, designed to be independent of application state and fully testable in isolation.
"""

import re
from typing import Optional, Tuple


def parse_aspect_ratio(ratio_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse and validate an aspect ratio string in W:H format.
    
    Args:
        ratio_str: Aspect ratio string in "W:H" format (e.g., "16:9", "4:3", "1:1")
        
    Returns:
        Tuple of (width, height) integers if valid, None if invalid
        
    Validation Rules:
        - Must be in strict "W:H" format with positive integers only
        - Calculated ratio (width/height) must be between 0.1 and 100.0 inclusive
        - Rejects malformed strings like "16:", ":9", "16/9", decimal ratios
        
    Examples:
        >>> parse_aspect_ratio("16:9")
        (16, 9)
        >>> parse_aspect_ratio("1:1")
        (1, 1)
        >>> parse_aspect_ratio("invalid")
        None
    """
    if not isinstance(ratio_str, str):
        return None
        
    # Use strict regex pattern to match only "W:H" format with integers
    pattern = r'^(\d+):(\d+)$'
    match = re.match(pattern, ratio_str.strip())
    
    if not match:
        return None
        
    try:
        width = int(match.group(1))
        height = int(match.group(2))
        
        # Validate positive integers
        if width <= 0 or height <= 0:
            return None
            
        # Calculate ratio and validate bounds (0.1 to 100.0 inclusive)
        ratio = width / height
        if ratio < 0.1 or ratio > 100.0:
            return None
            
        return (width, height)
        
    except (ValueError, ZeroDivisionError):
        return None


def calculate_fixed_aspect_ratio_bbox(
    original_bbox: Tuple[int, int, int, int], 
    image_dims: Tuple[int, int], 
    aspect_ratio: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Calculate fixed aspect ratio bbox that CONTAINS entire source (adds black bars if needed).
    
    Args:
        original_bbox: Original bounding box as (x1, y1, x2, y2)
        image_dims: Image dimensions as (width, height)
        aspect_ratio: Target aspect ratio as (width, height) integers
        
    Returns:
        New bounding box as (x1, y1, x2, y2) with EXACT fixed aspect ratio
        
    Algorithm (Expand with Black Bars):
        1. Find minimum rectangle with target aspect ratio that CONTAINS entire source
        2. NEVER truncate/crop source content
        3. Add black bars (expand canvas) if needed to accommodate ideal frame
        4. Position optimally within expanded canvas
        
    Examples:
        >>> calculate_fixed_aspect_ratio_bbox((100, 100, 478, 1215), (676, 1280), (1, 1))
        # Source: 378×1115 → needs 1115×1115 frame
        # Image: 676×1280 → add black bars to accommodate 1115×1115
        # Result: 1115×1115 frame positioned to contain source (may extend beyond image)
    """
    x1, y1, x2, y2 = original_bbox
    image_width, image_height = image_dims
    aspect_w, aspect_h = aspect_ratio
    
    # Validate inputs
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox: x2 must be > x1 and y2 must be > y1")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image dimensions: must be positive")
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError("Invalid aspect ratio: must be positive integers")
    
    # Calculate source bbox dimensions
    source_width = x2 - x1
    source_height = y2 - y1
    
    # Calculate target aspect ratio
    target_ratio = aspect_w / aspect_h
    
    # EXPAND WITH BLACK BARS ALGORITHM:
    # Find minimum dimensions with target aspect ratio that CONTAINS entire source
    # Add black bars if needed - NEVER crop source
    
    # Simple logic: 
    # final_width = max(source_width, source_height × ratio)
    # final_height = max(source_height, source_width / ratio)
    final_width = max(source_width, int(source_height * target_ratio))
    final_height = max(source_height, int(source_width / target_ratio))
    
    # Verify containment (should always be true with correct algorithm)
    if final_width < source_width or final_height < source_height:
        # Fallback: ensure minimum containment
        final_width = max(final_width, source_width)
        final_height = max(final_height, source_height)
        # Recalculate to maintain ratio
        if target_ratio >= 1.0:
            final_height = int(final_width / target_ratio)
        else:
            final_width = int(final_height * target_ratio)
    
    # Position optimally to contain source (may extend beyond image bounds)
    # Calculate source center
    source_center_x = (x1 + x2) // 2
    source_center_y = (y1 + y2) // 2
    
    # Ideal position: center the target frame on source center
    ideal_x1 = source_center_x - final_width // 2
    ideal_y1 = source_center_y - final_height // 2
    
    # Calculate final coordinates (may extend beyond image bounds - that's OK!)
    # Black bars will be added during image processing to accommodate
    new_x1 = ideal_x1
    new_y1 = ideal_y1
    new_x2 = new_x1 + final_width
    new_y2 = new_y1 + final_height
    
    return (new_x1, new_y1, new_x2, new_y2)


def validate_bbox(bbox: Tuple[int, int, int, int], image_dims: Tuple[int, int]) -> bool:
    """
    Validate that a bounding box has valid coordinates and is within image bounds.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        image_dims: Image dimensions as (width, height)
        
    Returns:
        True if bbox is valid, False otherwise
        
    Validation Checks:
        - x2 > x1 and y2 > y1 (non-zero area)
        - All coordinates are non-negative
        - bbox is entirely within image bounds
        - All coordinates are integers
        
    Examples:
        >>> validate_bbox((10, 10, 50, 30), (100, 100))
        True
        >>> validate_bbox((50, 30, 10, 10), (100, 100))
        False  # x2 < x1, y2 < y1
        >>> validate_bbox((10, 10, 110, 30), (100, 100))
        False  # x2 > image_width
    """
    try:
        x1, y1, x2, y2 = bbox
        image_width, image_height = image_dims
        
        # Check coordinate types (should be integers)
        if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
            return False
        if not all(isinstance(dim, (int, float)) for dim in [image_width, image_height]):
            return False
            
        # Convert to integers for validation
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        image_width, image_height = int(image_width), int(image_height)
        
        # Check basic validity
        if x2 <= x1 or y2 <= y1:
            return False
            
        # Check bounds
        if x1 < 0 or y1 < 0:
            return False
        if x2 > image_width or y2 > image_height:
            return False
            
        # Check image dimensions are positive
        if image_width <= 0 or image_height <= 0:
            return False
            
        return True
        
    except (ValueError, TypeError, IndexError):
        return False


def normalize_bbox(
    bbox: Tuple[int, int, int, int], 
    image_dims: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Normalize a bounding box to ensure valid coordinates within image bounds.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        image_dims: Image dimensions as (width, height)
        
    Returns:
        Normalized bounding box as (x1, y1, x2, y2)
        
    Normalization Operations:
        - Clamp coordinates to image boundaries [0, width] and [0, height]
        - Fix coordinate ordering (ensure x2 > x1, y2 > y1)
        - Ensure minimum bbox size of 1x1
        - Convert to integers
        
    Examples:
        >>> normalize_bbox((-10, -5, 50, 30), (100, 100))
        (0, 0, 50, 30)  # Negative coords clamped to 0
        >>> normalize_bbox((10, 10, 110, 130), (100, 100))
        (10, 10, 100, 100)  # Out-of-bounds coords clamped
        >>> normalize_bbox((50, 30, 10, 10), (100, 100))
        (10, 10, 50, 30)  # Coordinates swapped to fix ordering
    """
    try:
        x1, y1, x2, y2 = bbox
        image_width, image_height = image_dims
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        image_width, image_height = int(image_width), int(image_height)
        
        # Ensure positive image dimensions
        image_width = max(1, image_width)
        image_height = max(1, image_height)
        
        # Fix coordinate ordering if needed
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(x1 + 1, min(x2, image_width))
        y2 = max(y1 + 1, min(y2, image_height))
        
        # Ensure minimum 1x1 size
        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1
            
        # Final bounds check (in case minimum size pushes us out of bounds)
        if x2 > image_width:
            x2 = image_width
            x1 = max(0, x2 - 1)
        if y2 > image_height:
            y2 = image_height
            y1 = max(0, y2 - 1)
            
        return (x1, y1, x2, y2)
        
    except (ValueError, TypeError, IndexError):
        # Return a safe default bbox if normalization fails
        return (0, 0, min(1, int(image_dims[0])), min(1, int(image_dims[1]))) 