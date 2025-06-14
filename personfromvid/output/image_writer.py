"""Image writing and processing for output generation.

This module implements the ImageWriter class that handles high-quality image
processing and file I/O operations for generating the final output images.
"""

import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps
import cv2

from ..data.config import OutputImageConfig
from ..data.frame_data import FrameData
from ..data.detection_results import FaceDetection
from ..data.context import ProcessingContext
from ..utils.logging import get_logger
from ..utils.exceptions import ImageWriteError
from .naming_convention import NamingConvention


class ImageWriter:
    """Handles image processing and file writing for output generation."""

    def __init__(self, context: ProcessingContext):
        """Initialize image writer.

        Args:
            context: ProcessingContext with unified pipeline data
        """
        self.config = context.config.output.image
        self.output_directory = context.output_directory
        self.video_base_name = context.video_base_name
        self.logger = get_logger(__name__)
        self.naming = NamingConvention(context=context)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._validate_config()

    def save_frame_outputs(
        self,
        frame: FrameData,
        pose_categories: List[str],
        head_angle_categories: List[str],
    ) -> List[str]:
        """Save all output images for a frame.

        Args:
            frame: Frame data to process
            pose_categories: List of pose categories this frame is selected for
            head_angle_categories: List of head angle categories this frame is selected for

        Returns:
            List of output file paths that were created
        """
        output_files = []

        try:
            # Load the source image
            source_image = self._load_frame_image(frame)

            # Save full frame images for pose categories
            if self.config.full_frame_enabled and pose_categories:
                for category in pose_categories:
                    rank = frame.selections.selection_rank or 1
                    filename = self.naming.get_full_frame_filename(
                        frame, category, rank, self.config.format
                    )
                    output_path = self.naming.get_full_output_path(filename)

                    # Apply resize if configured
                    resized_image = self._apply_resize(source_image)
                    self._save_image(resized_image, output_path)
                    output_files.append(str(output_path))
                    self.logger.debug(f"Saved full frame: {filename}")

            # Save face crop images for head angle categories
            if (
                self.config.face_crop_enabled
                and head_angle_categories
                and frame.face_detections
            ):
                best_face = frame.get_best_face()
                if best_face:
                    face_crop = self._crop_face(source_image, best_face)

                    for category in head_angle_categories:
                        rank = frame.selections.selection_rank or 1
                        filename = self.naming.get_face_crop_filename(
                            frame, category, rank, self.config.format
                        )
                        output_path = self.naming.get_full_output_path(filename)

                        # Face crops already handle resize logic in _crop_face, so save directly
                        self._save_image(face_crop, output_path)
                        output_files.append(str(output_path))
                        self.logger.debug(f"Saved face crop: {filename}")

            # Update frame's output files list
            frame.selections.output_files.extend(output_files)

            return output_files

        except Exception as e:
            error_msg = f"Failed to save outputs for frame {frame.frame_id}: {e}"
            self.logger.error(error_msg)
            raise ImageWriteError(error_msg) from e

    def _load_frame_image(self, frame: FrameData) -> np.ndarray:
        """Load frame image from cache or file.

        Args:
            frame: Frame data containing the file path and cached image.

        Returns:
            Image as a numpy array in RGB format.
        """
        if frame.image is None:
            raise ImageWriteError(
                f"Frame file not found or failed to load: {frame.file_path}"
            )

        try:
            # The cached frame.image is in BGR format, convert to RGB for PIL
            image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            return image_rgb

        except Exception as e:
            raise ImageWriteError(
                f"Error converting frame image {frame.frame_id} to RGB: {e}"
            ) from e

    def _crop_face(
        self, image: np.ndarray, face_detection: FaceDetection
    ) -> np.ndarray:
        """Crop face from image with padding and upscale to minimum size.

        Args:
            image: Source image as numpy array
            face_detection: Face detection with bounding box

        Returns:
            Cropped and potentially upscaled face image as numpy array
        """
        x1, y1, x2, y2 = face_detection.bbox

        # Calculate padding
        face_width = x2 - x1
        face_height = y2 - y1
        padding_x = int(face_width * self.config.face_crop_padding)
        padding_y = int(face_height * self.config.face_crop_padding)

        # Apply padding with bounds checking
        img_height, img_width = image.shape[:2]
        crop_x1 = max(0, x1 - padding_x)
        crop_y1 = max(0, y1 - padding_y)
        crop_x2 = min(img_width, x2 + padding_x)
        crop_y2 = min(img_height, y2 + padding_y)

        # Crop the image
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Determine minimum dimension: use resize value if configured, otherwise default to 512
        min_dimension = self.config.resize if self.config.resize is not None else 512
        
        # Upscale if both dimensions are smaller than the minimum dimension
        crop_height, crop_width = cropped.shape[:2]

        if crop_height < min_dimension and crop_width < min_dimension:
            # Calculate scale factor to ensure at least one dimension equals min_dimension
            scale_factor = min_dimension / max(crop_width, crop_height)
            new_width = int(crop_width * scale_factor)
            new_height = int(crop_height * scale_factor)

            # Convert to PIL for high-quality Lanczos upscaling
            pil_image = Image.fromarray(cropped)
            upscaled_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            cropped = np.array(upscaled_pil)

            self.logger.debug(
                f"Upscaled face crop from {crop_width}x{crop_height} to {new_width}x{new_height} "
                f"(scale: {scale_factor:.2f}, target: {min_dimension}px)"
            )

        # Store crop region in frame metadata for reference
        crop_key = f"face_crop_{face_detection.confidence:.2f}"
        # Note: This modifies the frame data, which might be used elsewhere
        # In a production system, we might want to avoid this side effect

        return cropped

    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save image to file with appropriate format and quality settings.

        Args:
            image: Image as numpy array (RGB format)
            output_path: Path to save the image
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)

            # Ensure RGB mode
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Save with format-specific settings
            if self.config.format.lower() == "png":
                pil_image.save(
                    output_path, format="PNG", optimize=self.config.png.optimize
                )
            elif self.config.format.lower() in ["jpg", "jpeg"]:
                pil_image.save(
                    output_path,
                    format="JPEG",
                    quality=self.config.jpeg.quality,
                    optimize=True,
                )
            else:
                raise ImageWriteError(f"Unsupported image format: {self.config.format}")

            self.logger.debug(
                f"Saved image: {output_path} ({self.config.format.upper()})"
            )

        except Exception as e:
            raise ImageWriteError(f"Failed to save image {output_path}: {e}") from e

    def _validate_config(self) -> None:
        """Validate the output configuration."""
        if self.config.format.lower() not in ["png", "jpg", "jpeg"]:
            raise ValueError(f"Unsupported output format: {self.config.format}")

        if not (0.0 <= self.config.face_crop_padding <= 1.0):
            raise ValueError(
                f"Face crop padding must be between 0.0 and 1.0, got: {self.config.face_crop_padding}"
            )

        if self.config.format.lower() in ["jpg", "jpeg"]:
            if not (70 <= self.config.jpeg.quality <= 100):
                raise ValueError(
                    f"JPEG quality must be between 70 and 100, got: {self.config.jpeg.quality}"
                )

        if self.config.resize is not None:
            if not (256 <= self.config.resize <= 4096):
                raise ValueError(
                    f"Resize dimension must be between 256 and 4096, got: {self.config.resize}"
                )

    def get_output_statistics(self) -> dict:
        """Get statistics about output generation.

        Returns:
            Dictionary with output statistics
        """
        return {
            "output_directory": str(self.output_directory),
            "video_base_name": self.video_base_name,
            "format": self.config.format,
            "face_crop_enabled": self.config.face_crop_enabled,
            "full_frame_enabled": self.config.full_frame_enabled,
            "face_crop_padding": self.config.face_crop_padding,
            "resize": self.config.resize,
        }

    def _apply_resize(self, image: np.ndarray) -> np.ndarray:
        """Apply resize to image based on configuration.

        Args:
            image: Image as numpy array

        Returns:
            Resized image as numpy array
        """
        if self.config.resize is None:
            return image
            
        height, width = image.shape[:2]
        max_dimension = self.config.resize
        
        # Only resize if image is larger than the target size
        if max(width, height) > max_dimension:
            # Calculate scale factor to ensure the larger dimension equals max_dimension
            scale_factor = max_dimension / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Convert to PIL for high-quality Lanczos downscaling
            pil_image = Image.fromarray(image)
            resized_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            resized_image = np.array(resized_pil)
            
            self.logger.debug(
                f"Resized image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.2f})"
            )
            return resized_image
        else:
            return image
