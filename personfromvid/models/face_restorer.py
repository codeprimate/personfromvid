"""Face restoration using GFPGAN for enhanced quality.

This module provides face restoration capabilities using GFPGAN models
for high-quality face enhancement at native resolution.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Suppress specific deprecation warnings from GFPGAN dependencies for better UX
# These are not actionable by users and come from external libraries
warnings.filterwarnings('ignore', message='.*torchvision.transforms.functional_tensor.*deprecated.*')
warnings.filterwarnings('ignore', message='.*torchvision.transforms.functional_tensor.*removed.*')
warnings.filterwarnings('ignore', message='.*pretrained.*deprecated.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Arguments other than a weight enum.*deprecated.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*scipy.ndimage.filters.*deprecated.*', category=DeprecationWarning)

from ..data.config import DeviceType, get_default_config
from ..models.model_manager import get_model_manager
from ..models.model_configs import ModelConfigs
from ..utils.exceptions import FaceRestorationError

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "auto"
DEFAULT_MODEL_NAME = "gfpgan_v1_4"


class FaceRestorer:
    """Face restoration using GFPGAN models.

    This class provides high-quality face restoration using GFPGAN
    with support for:
    - Native resolution face enhancement (no upscaling)
    - Device-aware initialization (CPU/GPU/AUTO)
    - Lazy model loading with caching
    - Comprehensive error handling with fallback
    - Memory management and cleanup

    Examples:
        Basic usage:
        >>> restorer = FaceRestorer()
        >>> enhanced_face = restorer.restore_face(face_image, strength=0.8)

        GPU initialization:
        >>> restorer = FaceRestorer(device="cuda")
        >>> enhanced_face = restorer.restore_face(face_image, strength=1.0)

        Custom model and configuration:
        >>> restorer = FaceRestorer("gfpgan_v1_4", device="cpu", config=custom_config)
        >>> enhanced_face = restorer.restore_face(face_image, strength=0.5)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        config: Optional["Config"] = None,
    ):
        """Initialize face restorer with specified model.

        Args:
            model_name: Name of the GFPGAN model to use
            device: Computation device ("cpu", "cuda", or "auto")
            config: Application configuration object (optional)

        Raises:
            FaceRestorationError: If model loading fails or device is unsupported
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)

        # Store config or get default
        if config is None:
            self.config = get_default_config()
        else:
            self.config = config

        # Get model configuration
        self.model_config = ModelConfigs.get_model(model_name)
        if not self.model_config:
            raise FaceRestorationError(f"Unknown face restoration model: {model_name}")

        # Validate device support
        device_type = DeviceType.CPU if self.device == "cpu" else DeviceType.GPU
        if not self.model_config.is_device_supported(device_type):
            raise FaceRestorationError(
                f"Model {model_name} does not support device {self.device}"
            )

        # Download and cache model
        self.model_manager = get_model_manager()
        self.model_path = self.model_manager.ensure_model_available(model_name)

        # Initialize model inference engine (lazy loading)
        self._gfpgan_restorer = None

        logger.info(f"Initialized FaceRestorer with model {model_name} on {self.device}")

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            # Check for CUDA availability
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the GFPGAN model for inference.

        Raises:
            FaceRestorationError: If model loading fails
        """
        if self._gfpgan_restorer is not None:
            return

        try:
            # Import GFPGAN dependencies
            try:
                from gfpgan import GFPGANer
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            except ImportError as e:
                raise FaceRestorationError(
                    "GFPGAN not installed. Install with: pip install gfpgan"
                ) from e

            # Ensure GFPGAN auxiliary models cache directory exists
            gfpgan_cache_dir = self.model_manager.cache_dir / "gfpgan_auxiliary"
            gfpgan_cache_dir.mkdir(exist_ok=True)

            # Store original FaceRestoreHelper.__init__ method
            original_init = FaceRestoreHelper.__init__

            def patched_init(
                self, 
                upscale_factor, 
                face_size=512, 
                crop_ratio=(1, 1), 
                det_model='retinaface_resnet50', 
                save_ext='png', 
                template_3points=False, 
                pad_blur=False, 
                use_parse=False, 
                device=None, 
                model_rootpath=None
            ):
                # Override model_rootpath to use our cache directory
                if model_rootpath == 'gfpgan/weights' or model_rootpath is None:
                    model_rootpath = str(gfpgan_cache_dir)
                
                # Call original init with our cache directory
                return original_init(
                    self, upscale_factor, face_size, crop_ratio, 
                    det_model, save_ext, template_3points, 
                    pad_blur, use_parse, device, model_rootpath
                )

            # Temporarily monkey-patch FaceRestoreHelper.__init__
            FaceRestoreHelper.__init__ = patched_init

            try:
                # Initialize GFPGANer (now using our patched FaceRestoreHelper)
                self._gfpgan_restorer = GFPGANer(
                    model_path=str(self.model_path),
                    upscale=1,  # No upscaling, just enhancement
                    arch='clean',  # Use clean architecture
                    channel_multiplier=2,
                    bg_upsampler=None,  # No background upsampling
                    device=self.device
                )

                logger.debug(f"Successfully loaded GFPGAN model on {self.device}")
                logger.debug(f"GFPGAN auxiliary models cached in: {gfpgan_cache_dir}")

            finally:
                # Restore original FaceRestoreHelper.__init__
                FaceRestoreHelper.__init__ = original_init

        except Exception as e:
            raise FaceRestorationError(
                f"Failed to load GFPGAN model {self.model_name}: {str(e)}"
            ) from e

    def restore_face(
        self,
        image: np.ndarray,
        target_size: int = None,
        strength: float = 0.8
    ) -> np.ndarray:
        """Restore face using GFPGAN with configurable strength blending.

        This method enhances face quality at native resolution without upscaling,
        then optionally resizes to target size while preserving aspect ratio.

        Args:
            image: Input face image as numpy array (H, W, C) in RGB format
            target_size: Optional target output size in pixels (for longest dimension)
            strength: Restoration strength (0.0=no effect, 1.0=full restoration)

        Returns:
            Enhanced face image as numpy array (H, W, C) in RGB format

        Raises:
            FaceRestorationError: If restoration fails (will log error and return original)
        """
        if image is None or image.size == 0:
            raise FaceRestorationError("Input image is empty or None")

        # Validate strength parameter
        strength = max(0.0, min(1.0, float(strength)))

        try:
            # Convert numpy to correct format for GFPGAN (expects BGR)
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image

            # Convert RGB to BGR for GFPGAN
            image_bgr = image_uint8[:, :, ::-1]

            # If strength is 0, return original image (optionally resized)
            if strength == 0.0:
                result = image_uint8
            else:
                # Apply GFPGAN restoration
                self._load_model()
                
                # GFPGAN enhancement
                _, _, restored_img = self._gfpgan_restorer.enhance(
                    image_bgr, 
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )

                # Convert back to RGB
                restored_img_rgb = restored_img[:, :, ::-1]

                # Apply strength blending
                if strength == 1.0:
                    result = restored_img_rgb
                else:
                    blended = (strength * restored_img_rgb.astype(np.float32) + 
                              (1.0 - strength) * image_uint8.astype(np.float32))
                    result = np.clip(blended, 0, 255).astype(np.uint8)

            # Apply target size if specified (preserving aspect ratio)
            if target_size is not None:
                result = self._resize_to_target(result, target_size)

            return result

        except Exception as e:
            # Log error but don't crash - return original image as fallback
            logger.error(f"Face restoration failed: {str(e)}")
            logger.info("Falling back to original image")
            
            try:
                # Fallback to original image (optionally resized)
                if target_size is not None:
                    return self._resize_to_target(image_uint8, target_size)
                else:
                    return image_uint8
            except Exception as fallback_error:
                raise FaceRestorationError(
                    f"Both restoration and fallback failed: {str(e)}, fallback: {str(fallback_error)}"
                ) from e

    def _resize_to_target(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size preserving aspect ratio.

        Args:
            image: Input image as numpy array
            target_size: Target size for longest dimension

        Returns:
            Resized image as numpy array
        """
        try:
            from PIL import Image
            
            height, width = image.shape[:2]
            
            # Calculate new dimensions preserving aspect ratio
            scale_factor = target_size / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize using PIL
            pil_image = Image.fromarray(image)
            resized_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            return np.array(resized_pil)
            
        except Exception as e:
            logger.warning(f"Resize failed, returning original: {e}")
            return image

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_path": str(self.model_path),
            "model_loaded": self._gfpgan_restorer is not None,
            "model_config": {
                "supported_devices": [d.value for d in self.model_config.supported_devices],
                "input_size": self.model_config.input_size,
                "description": self.model_config.description,
            }
        }

    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, '_gfpgan_restorer') and self._gfpgan_restorer is not None:
            try:
                # Clean up CUDA memory if on GPU
                if self.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
            except Exception:
                pass  # Ignore cleanup errors


def create_face_restorer(
    model_name: Optional[str] = None,
    device: str = "auto",
    config: Optional["Config"] = None,
) -> FaceRestorer:
    """Create a face restorer instance with default settings.

    Args:
        model_name: Name of the GFPGAN model (defaults to gfpgan_v1_4)
        device: Computation device ("cpu", "cuda", or "auto")
        config: Application configuration object (optional)

    Returns:
        Initialized FaceRestorer instance

    Raises:
        FaceRestorationError: If initialization fails
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    return FaceRestorer(model_name=model_name, device=device, config=config) 