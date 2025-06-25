"""Image processing for crop and resize operations.

This module implements the ImageProcessor class that handles pixel-level image
manipulations including cropping, padding, resizing, and face restoration.
"""

import os
import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from PIL import Image

from ..data.config import OutputImageConfig
from ..utils.logging import get_logger
from .crop_utils import parse_aspect_ratio, calculate_fixed_aspect_ratio_bbox

if TYPE_CHECKING:
    from ..models.face_restorer import FaceRestorer


class ImageProcessor:
    """Handles pixel-level image processing operations."""

    def __init__(self, config: OutputImageConfig):
        """Initialize image processor.

        Args:
            config: Output image configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        self.logger.debug(f"ImageProcessor initialized with crop_ratio: {config.crop_ratio}")
        self.logger.debug(f"ImageProcessor initialized with enable_pose_cropping: {config.enable_pose_cropping}")
        
        # Initialize FaceRestorer with lazy loading for face restoration
        self._face_restorer: Optional["FaceRestorer"] = None
        self._face_restorer_initialized = False

    def _get_face_restorer(self) -> Optional["FaceRestorer"]:
        """Get FaceRestorer instance with lazy loading.
        
        Returns:
            FaceRestorer instance if face restoration is enabled, None otherwise
        """
        if not self.config.face_restoration_enabled:
            return None
            
        if not self._face_restorer_initialized:
            try:
                from ..models.face_restorer import create_face_restorer
                self._face_restorer = create_face_restorer()
                self.logger.debug("FaceRestorer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize FaceRestorer: {e}")
                self.logger.info("Face restoration will be disabled for this session")
                self._face_restorer = None
            finally:
                self._face_restorer_initialized = True
                
        return self._face_restorer

    def expand_bbox_to_aspect_ratio(
        self,
        bbox: Tuple[int, int, int, int],
        image: np.ndarray,
        padding: float = 0.0
    ) -> Tuple[int, int, int, int]:
        """Expand bbox to configured aspect ratio with optional padding.
        
        This method applies padding first, then converts to the target aspect ratio
        if one is configured. If no aspect ratio is configured, it just applies padding.
        
        Args:
            bbox: Original bounding box as (x1, y1, x2, y2)
            image: Source image for dimension calculation
            padding: Padding factor as proportion of bbox size (0.0 to 1.0)
            
        Returns:
            Final bounding box as (x1, y1, x2, y2)
        """
        image_height, image_width = image.shape[:2]
        image_dims = (image_width, image_height)

        # Step 1: Apply padding if specified
        if padding > 0.0:
            padded_bbox = self._apply_padding_to_bbox(bbox, padding, image_dims)
            self.logger.debug(f"Applied padding {padding}: {bbox} -> {padded_bbox}")
        else:
            padded_bbox = bbox
            self.logger.debug(f"No padding applied: {bbox}")
            
        # Step 2: Convert to fixed aspect ratio if configured
        if self.config.crop_ratio is not None:
            # Handle "any" case - skip aspect ratio calculation, use variable aspect ratio
            if self.config.crop_ratio == "any":
                self.logger.debug("crop_ratio='any': using variable aspect ratio with padding")
                return padded_bbox
            
            # Handle W:H format aspect ratios
            aspect_ratio = parse_aspect_ratio(self.config.crop_ratio)
            if aspect_ratio is not None:
                try:
                    final_bbox = calculate_fixed_aspect_ratio_bbox(
                        padded_bbox, image_dims, aspect_ratio
                    )

                    return final_bbox
                except Exception as e:
                    error_msg = f"Failed to calculate fixed aspect ratio bbox: {e}"
                    self.logger.warning(error_msg)
                    return padded_bbox
            else:
                warning_msg = f"Invalid crop_ratio format: {self.config.crop_ratio}"
                self.logger.warning(warning_msg)
        else:
            no_ratio_msg = "No crop_ratio configured, using padded bbox"
            self.logger.debug(no_ratio_msg)
        
        return padded_bbox

    def _apply_padding_to_bbox(
        self, 
        bbox: Tuple[int, int, int, int], 
        padding: float, 
        image_dims: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Apply padding to a bounding box with bounds checking.
        
        Args:
            bbox: Original bounding box as (x1, y1, x2, y2)
            padding: Padding factor as proportion of bbox size (0.0 to 1.0)
            image_dims: Image dimensions as (width, height)
            
        Returns:
            Padded bounding box as (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        image_width, image_height = image_dims

        # Calculate padding
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        padding_x = int(bbox_width * padding)
        padding_y = int(bbox_height * padding)

        # Apply padding with bounds checking
        padded_x1 = max(0, x1 - padding_x)
        padded_y1 = max(0, y1 - padding_y)
        padded_x2 = min(image_width, x2 + padding_x)
        padded_y2 = min(image_height, y2 + padding_y)

        return (padded_x1, padded_y1, padded_x2, padded_y2)

    def crop_and_resize(
        self, 
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        padding: float = 0.0,
        use_face_restoration: bool = False
    ) -> np.ndarray:
        """Crop region from image with optional padding and aspect ratio conversion.

        Args:
            image: Source image as numpy array
            bbox: Original bounding box as (x1, y1, x2, y2) - may extend beyond image
            padding: Padding factor as proportion of bbox size (0.0 to 1.0). 
                    When 0.0, crops directly without padding.
            use_face_restoration: Whether to apply face restoration if enabled

        Returns:
            Cropped and potentially upscaled/restored image as numpy array
        """
        # Use helper method to get final bbox (handles padding + aspect ratio conversion)
        final_bbox = self.expand_bbox_to_aspect_ratio(bbox, image, padding)
        crop_x1, crop_y1, crop_x2, crop_y2 = final_bbox

        # Handle case where bbox extends beyond image bounds - add transparent bars
        image_height, image_width = image.shape[:2]
        
        # Calculate how much the bbox extends beyond image
        extend_left = max(0, -crop_x1)
        extend_top = max(0, -crop_y1) 
        extend_right = max(0, crop_x2 - image_width)
        extend_bottom = max(0, crop_y2 - image_height)
        
        if extend_left > 0 or extend_top > 0 or extend_right > 0 or extend_bottom > 0:
            # Need to add transparent bars - expand the image canvas
            # Convert image to RGBA if it's not already
            if image.shape[2] == 3:  # RGB
                # Add alpha channel with full opacity
                alpha_channel = np.full((image_height, image_width, 1), 255, dtype=image.dtype)
                image_rgba = np.concatenate([image, alpha_channel], axis=2)
            else:  # Already RGBA
                image_rgba = image
            
            # Create expanded canvas with transparent background (RGBA with alpha=0)
            expanded_width = image_width + extend_left + extend_right
            expanded_height = image_height + extend_top + extend_bottom
            expanded_image = np.zeros((expanded_height, expanded_width, 4), dtype=image.dtype)
            
            # Copy original image into expanded canvas
            expanded_image[extend_top:extend_top + image_height, extend_left:extend_left + image_width] = image_rgba
            
            # Adjust bbox coordinates to expanded image space
            crop_x1 += extend_left
            crop_y1 += extend_top
            crop_x2 += extend_left
            crop_y2 += extend_top
            
            # Use expanded image for cropping
            source_image = expanded_image
        else:
            # No extension needed, use original image
            source_image = image

        # Crop the region from (possibly expanded) image
        cropped = source_image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Determine minimum dimension: use resize value if configured, otherwise default_crop_size
        min_dimension = self.config.resize if self.config.resize is not None else self.config.default_crop_size

        # Check if upscaling is needed
        crop_height, crop_width = cropped.shape[:2]
        needs_upscaling = crop_height < min_dimension and crop_width < min_dimension

        # Apply face restoration if enabled and requested
        if use_face_restoration and needs_upscaling:
            face_restorer = self._get_face_restorer()
            if face_restorer is not None:
                try:
                    # Calculate target size for face restoration - use more conservative approach
                    # Only upscale to the minimum needed size, not excessively large
                    target_size = min(min_dimension, self.config.default_crop_size)  # Cap at default_crop_size for performance
                    
                    # Apply GFPGAN face restoration with configured strength
                    restored_image = face_restorer.restore_face(
                        image=cropped,
                        target_size=target_size,
                        strength=self.config.face_restoration_strength
                    )
                    
                    self.logger.debug(
                        f"GFPGAN face restoration applied: {crop_width}x{crop_height} -> target {target_size}px "
                        f"(strength: {self.config.face_restoration_strength:.2f})"
                    )
                    return restored_image
                    
                except Exception as e:
                    self.logger.warning(f"Face restoration failed, falling back to Lanczos: {e}")
                    # Fall through to standard Lanczos processing

        # Standard upscaling logic (fallback or when face restoration not used)
        if needs_upscaling:
            # Calculate scale factor to ensure at least one dimension equals min_dimension
            scale_factor = min_dimension / max(crop_width, crop_height)
            new_width = int(crop_width * scale_factor)
            new_height = int(crop_height * scale_factor)

            # Convert to PIL for high-quality Lanczos upscaling
            pil_image = Image.fromarray(cropped)
            upscaled_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            cropped = np.array(upscaled_pil)

            self.logger.debug(
                f"Lanczos upscaling applied: {crop_width}x{crop_height} -> {new_width}x{new_height} "
                f"(scale: {scale_factor:.2f}, target: {min_dimension}px)"
            )

        return cropped 