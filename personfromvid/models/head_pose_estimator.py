"""Head pose estimation inference using HopeNet and 6DRepNet models.

This module provides head pose estimation capabilities using state-of-the-art models
like HopeNet and 6DRepNet. It supports both CPU and GPU inference with automatic model
downloading and caching.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import cv2
import math
import time

from ..data.detection_results import HeadPoseResult, FaceDetection
from ..utils.exceptions import ModelInferenceError, HeadPoseEstimationError
from .model_configs import ModelConfigs, ModelFormat
from .model_manager import get_model_manager
from ..analysis.head_angle_classifier import HeadAngleClassifier

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_DEVICE = "cpu"
DEFAULT_YAW_THRESHOLD = 22.5
DEFAULT_PITCH_THRESHOLD = 22.5
DEFAULT_PROFILE_YAW_THRESHOLD = 67.5
DEFAULT_MAX_ROLL = 30.0
# More lenient thresholds for forward-facing detection
DEFAULT_FORWARD_YAW_THRESHOLD = 45.0  # ±45 degrees is more realistic for "front-facing"
DEFAULT_FORWARD_PITCH_THRESHOLD = 30.0  # ±30 degrees for up/down variation


class HeadPoseEstimator:
    """Head pose estimation inference using HopeNet or 6DRepNet models.

    This class provides high-performance head pose estimation with support for:
    - HopeNet models (Pickle format)
    - 6DRepNet models (ONNX format)
    - Batch processing for improved efficiency
    - Angle normalization and validation
    - Cardinal direction classification
    - CPU and GPU acceleration

    Examples:
        Basic usage:
        >>> estimator = HeadPoseEstimator("hopenet_alpha1")
        >>> head_pose = estimator.estimate_head_pose(face_image)

        Batch processing:
        >>> head_poses = estimator.estimate_batch([face1, face2, face3])

        Custom angle thresholds:
        >>> estimator.set_angle_thresholds(yaw=25.0, pitch=25.0)
        >>> head_pose = estimator.estimate_head_pose(face_image)
    """

    def __init__(
        self,
        model_name: str,
        device: str = DEFAULT_DEVICE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        config: Optional["Config"] = None,
    ):
        """Initialize head pose estimator with specified model.

        Args:
            model_name: Name of the head pose estimation model to use
            device: Computation device ("cpu", "cuda", or "auto")
            confidence_threshold: Minimum confidence threshold for estimates
            config: Application configuration object (optional)

        Raises:
            HeadPoseEstimationError: If model loading fails or device is unsupported
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.confidence_threshold = confidence_threshold
        
        # Store config or get default
        if config is None:
            from ..data.config import get_default_config
            self.config = get_default_config()
        else:
            self.config = config

        # Angle thresholds for direction classification (degrees)
        self.yaw_threshold = DEFAULT_YAW_THRESHOLD
        self.pitch_threshold = DEFAULT_PITCH_THRESHOLD
        self.profile_yaw_threshold = DEFAULT_PROFILE_YAW_THRESHOLD
        self.max_roll = DEFAULT_MAX_ROLL

        # More lenient thresholds for forward-facing detection
        self.forward_yaw_threshold = DEFAULT_FORWARD_YAW_THRESHOLD
        self.forward_pitch_threshold = DEFAULT_FORWARD_PITCH_THRESHOLD

        # Get model configuration
        self.model_config = ModelConfigs.get_model(model_name)
        if not self.model_config:
            raise HeadPoseEstimationError(
                f"Unknown head pose estimation model: {model_name}"
            )

        # Validate device support
        from ..data.config import DeviceType

        device_type = DeviceType.CPU if self.device == "cpu" else DeviceType.GPU
        if not self.model_config.is_device_supported(device_type):
            raise HeadPoseEstimationError(
                f"Model {model_name} does not support device {device}"
            )

        # Download and cache model
        self.model_manager = get_model_manager()
        self.model_path = self.model_manager.ensure_model_available(model_name)

        # Initialize model inference engine
        self._model = None
        self._input_size = self.model_config.input_size
        self._model_format = self.model_config.files[0].format

        # Initialize head angle classifier for direction classification
        self._head_angle_classifier = HeadAngleClassifier()
        # Sync thresholds from classifier
        self.yaw_threshold = self._head_angle_classifier.yaw_threshold
        self.pitch_threshold = self._head_angle_classifier.pitch_threshold
        self.profile_yaw_threshold = self._head_angle_classifier.profile_yaw_threshold
        self.max_roll = self._head_angle_classifier.max_roll

        logger.info(
            f"Initialized HeadPoseEstimator with model {model_name} on {device}"
        )

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
        """Load the head pose estimation model for inference.

        Raises:
            HeadPoseEstimationError: If model loading fails
        """
        if self._model is not None:
            return

        try:
            if self._model_format == ModelFormat.PICKLE:
                self._load_hopenet_model()
            elif self._model_format == ModelFormat.ONNX:
                self._load_onnx_model()
            elif (
                self._model_format == ModelFormat.PYTORCH
                or self._model_format == ModelFormat.SAFETENSORS
            ):
                self._load_pytorch_model()
            else:
                raise HeadPoseEstimationError(
                    f"Unsupported model format: {self._model_format}"
                )

            logger.debug(f"Successfully loaded {self.model_name} model")

        except Exception as e:
            raise HeadPoseEstimationError(
                f"Failed to load model {self.model_name}: {str(e)}"
            ) from e

    def _load_hopenet_model(self) -> None:
        """Load HopeNet model from pickle file."""
        try:
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms
        except ImportError as e:
            raise HeadPoseEstimationError(
                "PyTorch not installed. Install with: pip install torch torchvision"
            ) from e

        # Load the model state dict
        checkpoint = torch.load(str(self.model_path), map_location=self.device)

        # Create HopeNet model architecture
        self._model = self._create_hopenet_model()

        # Load state dict
        if "state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["state_dict"])
        else:
            self._model.load_state_dict(checkpoint)

        self._model.to(self.device)
        self._model.eval()

        # Set up preprocessing transform
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self._input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.debug(f"HopeNet model loaded on device: {self.device}")

    def _create_hopenet_model(self):
        """Create HopeNet model architecture."""
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
        except ImportError as e:
            raise HeadPoseEstimationError("PyTorch not installed") from e

        # ResNet50-based HopeNet architecture
        class Hopenet(nn.Module):
            def __init__(self, block, layers, num_bins):
                super(Hopenet, self).__init__()
                self.inplanes = 64
                self.conv1 = nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                self.avgpool = nn.AvgPool2d(7, stride=1)

                self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
                self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
                self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

                # Angular softmax
                self.softmax = nn.Softmax(dim=1)

                # Initialize weights
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            def _make_layer(self, block, planes, blocks, stride=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(
                            self.inplanes,
                            planes * block.expansion,
                            kernel_size=1,
                            stride=stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample))
                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes))

                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)

                # Predict angular bins
                yaw = self.fc_yaw(x)
                pitch = self.fc_pitch(x)
                roll = self.fc_roll(x)

                return yaw, pitch, roll

        # Bottleneck block for ResNet50
        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(
                    planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
                )
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(
                    planes, planes * self.expansion, kernel_size=1, bias=False
                )
                self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)

                return out

        # Create HopeNet with ResNet50 backbone
        num_bins = 66  # Standard HopeNet bins
        model = Hopenet(Bottleneck, [3, 4, 6, 3], num_bins)

        return model

    def _load_onnx_model(self) -> None:
        """Load ONNX model using ONNXRuntime."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise HeadPoseEstimationError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            ) from e

        # Set up providers based on device
        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Create inference session
        self._model = ort.InferenceSession(str(self.model_path), providers=providers)

        # Get input/output info
        self._input_name = self._model.get_inputs()[0].name
        self._output_names = [output.name for output in self._model.get_outputs()]

        logger.debug(f"ONNX model loaded with providers: {self._model.get_providers()}")

    def _load_pytorch_model(self) -> None:
        """Load PyTorch model (including safetensors format)."""
        try:
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms
        except ImportError as e:
            raise HeadPoseEstimationError(
                "PyTorch not installed. Install with: pip install torch torchvision"
            ) from e

        # Load the model (supports both .pth and .safetensors)
        if self.model_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file

                state_dict = load_file(str(self.model_path))
            except ImportError:
                raise HeadPoseEstimationError(
                    "safetensors not installed. Install with: pip install safetensors"
                )
            except Exception as e:
                raise HeadPoseEstimationError(
                    f"Failed to load safetensors model from {self.model_path}: {str(e)}"
                ) from e
        else:
            state_dict = torch.load(str(self.model_path), map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Create 6DRepNet model architecture (RepVGG-A0 backbone)
        self._model = self._create_sixdrepnet_model()

        # Load state dict
        self._model.load_state_dict(state_dict)

        self._model.to(self.device)
        self._model.eval()

        # Set up preprocessing transform for 6DRepNet
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self._input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.debug(f"PyTorch 6DRepNet model loaded on device: {self.device}")

    def _create_sixdrepnet_model(self):
        """Create 6DRepNet model architecture with RepVGG-A0 backbone to match X01D model."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError as e:
            raise HeadPoseEstimationError("PyTorch not installed") from e

        # RepVGG Block for the backbone (deployed version)
        class RepVGGBlock(nn.Module):
            def __init__(
                self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
            ):
                super(RepVGGBlock, self).__init__()
                # For deployed model, only need the reparameterized conv
                self.rbr_reparam = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=True
                )
                self.activation = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.activation(self.rbr_reparam(x))

        # 6DRepNet model matching X01D architecture
        class SixDRepNet(nn.Module):
            def __init__(self):
                super(SixDRepNet, self).__init__()

                # Layer naming to match the X01D model weights
                self.layer0 = RepVGGBlock(
                    3, 48, kernel_size=3, stride=2, padding=1
                )  # RepVGG-A0 width

                # Layer 1: 48 -> 48 (2 blocks)
                self.layer1 = nn.Sequential(
                    RepVGGBlock(48, 48, stride=2), RepVGGBlock(48, 48, stride=1)
                )

                # Layer 2: 48 -> 96 (4 blocks)
                self.layer2 = nn.Sequential(
                    RepVGGBlock(48, 96, stride=2),
                    RepVGGBlock(96, 96, stride=1),
                    RepVGGBlock(96, 96, stride=1),
                    RepVGGBlock(96, 96, stride=1),
                )

                # Layer 3: 96 -> 192 (14 blocks)
                layer3_blocks = [RepVGGBlock(96, 192, stride=2)]
                for _ in range(13):
                    layer3_blocks.append(RepVGGBlock(192, 192, stride=1))
                self.layer3 = nn.Sequential(*layer3_blocks)

                # Layer 4: 192 -> 1280 (1 block)
                self.layer4 = nn.Sequential(RepVGGBlock(192, 1280, stride=2))

                # Global average pooling
                self.gap = nn.AdaptiveAvgPool2d(1)

                # 6D rotation representation head
                self.linear_reg = nn.Linear(1280, 6)

            def forward(self, x):
                x = self.layer0(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.gap(x)
                x = x.view(x.size(0), -1)
                x = self.linear_reg(x)
                return x

        # Create the model
        model = SixDRepNet()

        return model

    def estimate_head_pose(self, face_image: np.ndarray) -> HeadPoseResult:
        """Estimate head pose in a single face image.

        Args:
            face_image: Input face image as numpy array (H, W, C) in BGR format

        Returns:
            Head pose result with yaw, pitch, roll angles and direction

        Raises:
            HeadPoseEstimationError: If head pose estimation fails
        """
        if face_image is None or face_image.size == 0:
            raise HeadPoseEstimationError("Input face image is empty or None")

        # Ensure model is loaded
        self._load_model()

        try:
            if self._model_format == ModelFormat.PICKLE:
                return self._estimate_hopenet(face_image)
            elif self._model_format == ModelFormat.ONNX:
                return self._estimate_onnx(face_image)
            elif (
                self._model_format == ModelFormat.PYTORCH
                or self._model_format == ModelFormat.SAFETENSORS
            ):
                return self._estimate_pytorch(face_image)
            else:
                raise HeadPoseEstimationError(
                    f"Unsupported model format: {self._model_format}"
                )

        except Exception as e:
            raise HeadPoseEstimationError(
                f"Head pose estimation failed: {str(e)}"
            ) from e

    def estimate_batch(self, face_images: List[np.ndarray]) -> List[HeadPoseResult]:
        """Estimate head poses in a batch of face images for improved efficiency.

        Args:
            face_images: List of input face images as numpy arrays (H, W, C) in BGR format

        Returns:
            List of head pose results, one per input image

        Raises:
            HeadPoseEstimationError: If batch head pose estimation fails
        """
        if not face_images:
            return []

        # Validate all images
        for i, image in enumerate(face_images):
            if image is None or image.size == 0:
                raise HeadPoseEstimationError(
                    f"Input face image at index {i} is empty or None"
                )

        # Ensure model is loaded
        self._load_model()

        try:
            if self._model_format == ModelFormat.PICKLE:
                return self._estimate_batch_hopenet(face_images)
            elif self._model_format == ModelFormat.ONNX:
                return self._estimate_batch_onnx(face_images)
            elif (
                self._model_format == ModelFormat.PYTORCH
                or self._model_format == ModelFormat.SAFETENSORS
            ):
                return self._estimate_batch_pytorch(face_images)
            else:
                raise HeadPoseEstimationError(
                    f"Unsupported model format: {self._model_format}"
                )

        except Exception as e:
            raise HeadPoseEstimationError(
                f"Batch head pose estimation failed: {str(e)}"
            ) from e

    def _estimate_hopenet(self, face_image: np.ndarray) -> HeadPoseResult:
        """Estimate head pose using HopeNet model."""
        import torch

        # Preprocess image
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        input_tensor = self._transform(rgb_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            yaw, pitch, roll = self._model(input_tensor)

            # Convert logits to angles
            yaw_predicted = self._softmax_to_angle(yaw)
            pitch_predicted = self._softmax_to_angle(pitch)
            roll_predicted = self._softmax_to_angle(roll)

            # Calculate confidence (based on prediction entropy)
            confidence = self._calculate_confidence(yaw, pitch, roll)

        # Convert to degrees and normalize
        yaw_degrees = float(yaw_predicted.cpu().numpy())
        pitch_degrees = float(pitch_predicted.cpu().numpy())
        roll_degrees = float(roll_predicted.cpu().numpy())

        # Normalize angles
        yaw_degrees = self._normalize_angle(yaw_degrees)
        pitch_degrees = self._normalize_angle(pitch_degrees)
        roll_degrees = self._normalize_angle(roll_degrees)

        # Classify direction - This is now handled by HeadAngleClassifier
        # direction = self.angles_to_direction(yaw_degrees, pitch_degrees, roll_degrees)

        return HeadPoseResult(
            yaw=yaw_degrees,
            pitch=pitch_degrees,
            roll=roll_degrees,
            confidence=float(confidence.cpu().numpy()),
            face_id=0,
        )

    def _estimate_onnx(self, face_image: np.ndarray) -> HeadPoseResult:
        """Estimate head pose using ONNX model."""
        # Preprocess image
        input_image = self._preprocess_image_onnx(face_image)

        # Run inference
        outputs = self._model.run(self._output_names, {self._input_name: input_image})

        # Post-process results
        return self._postprocess_onnx_results(outputs)

    def _estimate_batch_hopenet(
        self, face_images: List[np.ndarray]
    ) -> List[HeadPoseResult]:
        """Batch estimation for HopeNet models."""
        import torch

        results = []

        # Process in batches for memory efficiency
        batch_size = self.config.models.batch_size
        for i in range(0, len(face_images), batch_size):
            batch_images = face_images[i : i + batch_size]

            # Preprocess batch
            batch_tensors = []
            for face_image in batch_images:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                tensor = self._transform(rgb_image)
                batch_tensors.append(tensor)

            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Run inference
            with torch.no_grad():
                yaw_batch, pitch_batch, roll_batch = self._model(batch_tensor)

                # Process each result in batch
                for j in range(len(batch_images)):
                    yaw = yaw_batch[j : j + 1]
                    pitch = pitch_batch[j : j + 1]
                    roll = roll_batch[j : j + 1]

                    # Convert to angles
                    yaw_predicted = self._softmax_to_angle(yaw)
                    pitch_predicted = self._softmax_to_angle(pitch)
                    roll_predicted = self._softmax_to_angle(roll)

                    # Calculate confidence
                    confidence = self._calculate_confidence(yaw, pitch, roll)

                    # Convert to degrees and normalize
                    yaw_degrees = self._normalize_angle(
                        float(yaw_predicted.cpu().numpy())
                    )
                    pitch_degrees = self._normalize_angle(
                        float(pitch_predicted.cpu().numpy())
                    )
                    roll_degrees = self._normalize_angle(
                        float(roll_predicted.cpu().numpy())
                    )

                    # Classify direction - This is now handled by HeadAngleClassifier
                    # direction = self.angles_to_direction(yaw_degrees, pitch_degrees, roll_degrees)

                    result = HeadPoseResult(
                        yaw=yaw_degrees,
                        pitch=pitch_degrees,
                        roll=roll_degrees,
                        confidence=float(confidence.cpu().numpy()),
                        face_id=j,
                    )
                    results.append(result)

        return results

    def _estimate_batch_onnx(
        self, face_images: List[np.ndarray]
    ) -> List[HeadPoseResult]:
        """Batch estimation for ONNX models."""
        results = []

        # ONNX models typically process single images, so iterate
        for i, face_image in enumerate(face_images):
            result = self._estimate_onnx(face_image)
            result.face_id = i
            results.append(result)

        return results

    def _estimate_pytorch(self, face_image: np.ndarray) -> HeadPoseResult:
        """Estimate head pose using PyTorch 6DRepNet model."""
        import torch

        # Preprocess image
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        input_tensor = self._transform(rgb_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            # Model outputs 6D rotation representation
            pred_6d = self._model(input_tensor)

            # Convert 6D representation to rotation matrix then to Euler angles
            yaw, pitch, roll = self._convert_6d_to_euler(pred_6d)

            # Calculate confidence (placeholder - could implement actual confidence from model)
            confidence = 0.85

        # Convert to degrees and normalize
        yaw_degrees = self._normalize_angle(float(yaw.cpu().numpy()))
        pitch_degrees = self._normalize_angle(float(pitch.cpu().numpy()))
        roll_degrees = self._normalize_angle(float(roll.cpu().numpy()))

        # Classify direction - This is now handled by HeadAngleClassifier
        # direction = self.angles_to_direction(yaw_degrees, pitch_degrees, roll_degrees)

        return HeadPoseResult(
            yaw=yaw_degrees,
            pitch=pitch_degrees,
            roll=roll_degrees,
            confidence=confidence,
            face_id=0,
        )

    def _estimate_batch_pytorch(
        self, face_images: List[np.ndarray]
    ) -> List[HeadPoseResult]:
        """Batch estimation for PyTorch 6DRepNet models."""
        import torch

        results = []

        # Process in batches for memory efficiency
        batch_size = self.config.models.batch_size
        for i in range(0, len(face_images), batch_size):
            batch_images = face_images[i : i + batch_size]

            # Preprocess batch
            batch_tensors = []
            for face_image in batch_images:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                tensor = self._transform(rgb_image)
                batch_tensors.append(tensor)

            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Run inference
            with torch.no_grad():
                pred_6d_batch = self._model(batch_tensor)

                # Process each result in batch
                for j in range(len(batch_images)):
                    pred_6d = pred_6d_batch[j : j + 1]

                    # Convert to Euler angles
                    yaw, pitch, roll = self._convert_6d_to_euler(pred_6d)

                    # Convert to degrees and normalize
                    yaw_degrees = self._normalize_angle(float(yaw.cpu().numpy()))
                    pitch_degrees = self._normalize_angle(float(pitch.cpu().numpy()))
                    roll_degrees = self._normalize_angle(float(roll.cpu().numpy()))

                    # Classify direction - This is now handled by HeadAngleClassifier
                    # direction = self.angles_to_direction(yaw_degrees, pitch_degrees, roll_degrees)

                    result = HeadPoseResult(
                        yaw=yaw_degrees,
                        pitch=pitch_degrees,
                        roll=roll_degrees,
                        confidence=0.85,
                        face_id=j,
                    )
                    results.append(result)

        return results

    def _convert_6d_to_euler(self, pred_6d):
        """Convert 6D rotation representation to Euler angles."""
        import torch
        import torch.nn.functional as F

        # Reshape 6D prediction to two 3D vectors
        pred_6d = pred_6d.view(-1, 6)
        a1 = pred_6d[:, :3].view(-1, 3, 1)
        a2 = pred_6d[:, 3:].view(-1, 3, 1)

        # Normalize first vector
        b1 = F.normalize(a1, dim=1)

        # Compute second vector orthogonal to first
        b2 = a2 - torch.bmm(b1.transpose(1, 2), a2) * b1
        b2 = F.normalize(b2, dim=1)

        # Compute third vector as cross product
        b3 = torch.cross(b1.squeeze(-1), b2.squeeze(-1), dim=1).unsqueeze(-1)

        # Construct rotation matrix
        R = torch.cat([b1, b2, b3], dim=-1)

        # Convert rotation matrix to Euler angles (ZYX convention)
        # Extract Euler angles from rotation matrix
        sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])

        singular = sy < 1e-6

        # Non-singular case
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        pitch = torch.atan2(-R[:, 2, 0], sy)
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])

        # Singular case (gimbal lock)
        yaw_singular = torch.atan2(-R[:, 0, 1], R[:, 1, 1])
        pitch_singular = torch.atan2(-R[:, 2, 0], sy)
        roll_singular = torch.zeros_like(roll)

        # Select based on singularity
        yaw = torch.where(singular, yaw_singular, yaw)
        pitch = torch.where(singular, pitch_singular, pitch)
        roll = torch.where(singular, roll_singular, roll)

        # Convert to degrees
        yaw = yaw * 180.0 / math.pi
        pitch = pitch * 180.0 / math.pi
        roll = roll * 180.0 / math.pi

        # Return first item if batch size is 1
        return yaw[0], pitch[0], roll[0]

    def _preprocess_image_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model inference."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb_image, self._input_size)

        # Normalize to [0, 1] and convert to float32
        normalized = resized.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        # Add batch dimension and transpose to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

        return input_tensor

    def _postprocess_onnx_results(self, outputs: List[np.ndarray]) -> HeadPoseResult:
        """Post-process ONNX model outputs to extract head pose."""
        # Typical ONNX head pose models output 3 angles directly
        if len(outputs) >= 3:
            yaw = (
                float(outputs[0].item())
                if outputs[0].size == 1
                else float(outputs[0][0])
            )
            pitch = (
                float(outputs[1].item())
                if outputs[1].size == 1
                else float(outputs[1][0])
            )
            roll = (
                float(outputs[2].item())
                if outputs[2].size == 1
                else float(outputs[2][0])
            )
        elif len(outputs) == 1 and outputs[0].shape[-1] == 3:
            # Single output with 3 angles
            angles = outputs[0].flatten()
            yaw, pitch, roll = float(angles[0]), float(angles[1]), float(angles[2])
        else:
            raise HeadPoseEstimationError(
                f"Unexpected ONNX output format: {[o.shape for o in outputs]}"
            )

        # Normalize angles
        yaw = self._normalize_angle(yaw)
        pitch = self._normalize_angle(pitch)
        roll = self._normalize_angle(roll)

        # Calculate confidence (placeholder for ONNX models)
        confidence = 0.8  # Could be improved with model-specific confidence calculation

        # Classify direction - This is now handled by HeadAngleClassifier
        # direction = self.angles_to_direction(yaw, pitch, roll)

        return HeadPoseResult(
            yaw=yaw, pitch=pitch, roll=roll, confidence=confidence, face_id=0
        )

    def _softmax_to_angle(self, predictions):
        """Convert softmax predictions to angle values."""
        import torch
        import torch.nn.functional as F

        # Apply softmax
        softmax_output = F.softmax(predictions, dim=1)

        # Create angle bins
        idx_tensor = torch.arange(
            predictions.size(1), dtype=torch.float32, device=predictions.device
        )
        idx_tensor = idx_tensor * 3 - 99  # Map to [-99, 99] degree range

        # Calculate expected value
        angle = torch.sum(softmax_output * idx_tensor, dim=1)

        return angle

    def _calculate_confidence(self, yaw, pitch, roll):
        """Calculate a combined confidence score for the head pose angles."""
        # Simple confidence based on magnitude of angles (closer to zero is better)
        yaw_conf = max(0, 1 - abs(yaw) / 90)
        pitch_conf = max(0, 1 - abs(pitch) / 90)
        roll_conf = max(0, 1 - abs(roll) / 45)

        # Weighted average
        return yaw_conf * 0.4 + pitch_conf * 0.4 + roll_conf * 0.2

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to be within [-180, 180]."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def angles_to_direction(self, yaw: float, pitch: float, roll: float) -> str:
        """Classify head pose angles into cardinal directions using HeadAngleClassifier.

        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
            roll: Roll angle in degrees

        Returns:
            Direction string from the 9 cardinal directions plus forward_facing
        """
        # First check for forward-facing with lenient thresholds
        if self.is_facing_forward(yaw, pitch, roll):
            # If it's also within strict thresholds, call it "front"
            if abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold:
                return "front"
            else:
                return "forward_facing"

        # Use the dedicated HeadAngleClassifier for standard classification
        return self._head_angle_classifier.classify_head_angle(yaw, pitch, roll)

    def is_valid_orientation(self, roll: float) -> bool:
        """Check if head orientation is valid (not excessively tilted)."""
        return self._head_angle_classifier.is_valid_orientation(roll)

    def is_facing_forward(self, yaw: float, pitch: float, roll: float) -> bool:
        """Check if head pose is facing forward using more lenient thresholds.

        This method uses more realistic thresholds for determining if someone
        is facing generally forward, which is useful for applications where
        strict frontal pose isn't required.

        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
            roll: Roll angle in degrees (optional check)

        Returns:
            True if the head pose is considered facing forward
        """
        # Check if within forward-facing thresholds
        yaw_forward = abs(yaw) <= self.forward_yaw_threshold
        pitch_forward = abs(pitch) <= self.forward_pitch_threshold
        roll_valid = abs(roll) <= self.max_roll  # Still check for extreme tilt

        return yaw_forward and pitch_forward and roll_valid

    def set_forward_facing_thresholds(
        self, yaw: Optional[float] = None, pitch: Optional[float] = None
    ) -> None:
        """Set more lenient thresholds for forward-facing detection.

        Args:
            yaw: Forward-facing yaw threshold in degrees (default: 45.0)
            pitch: Forward-facing pitch threshold in degrees (default: 30.0)
        """
        if yaw is not None:
            self.forward_yaw_threshold = yaw
        if pitch is not None:
            self.forward_pitch_threshold = pitch

        logger.debug(
            f"Updated forward-facing thresholds: yaw={self.forward_yaw_threshold}, pitch={self.forward_pitch_threshold}"
        )

    def set_angle_thresholds(
        self,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        profile_yaw: Optional[float] = None,
        max_roll: Optional[float] = None,
    ) -> None:
        """Set angle thresholds for direction classification.

        Args:
            yaw: Yaw threshold in degrees (default: 22.5)
            pitch: Pitch threshold in degrees (default: 22.5)
            profile_yaw: Profile yaw threshold in degrees (default: 67.5)
            max_roll: Maximum acceptable roll in degrees (default: 30.0)
        """
        if yaw is not None:
            self.yaw_threshold = yaw
        if pitch is not None:
            self.pitch_threshold = pitch
        if profile_yaw is not None:
            self.profile_yaw_threshold = profile_yaw
        if max_roll is not None:
            self.max_roll = max_roll

        # Also update the HeadAngleClassifier thresholds to keep them in sync
        self._head_angle_classifier.set_angle_thresholds(
            yaw=yaw, pitch=pitch, profile_yaw=profile_yaw, max_roll=max_roll
        )

        logger.debug(
            f"Updated angle thresholds: yaw={self.yaw_threshold}, pitch={self.pitch_threshold}, "
            f"profile_yaw={self.profile_yaw_threshold}, max_roll={self.max_roll}"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "model_format": self._model_format.value,
            "device": self.device,
            "input_size": self._input_size,
            "confidence_threshold": self.confidence_threshold,
            "angle_thresholds": {
                "yaw": self.yaw_threshold,
                "pitch": self.pitch_threshold,
                "profile_yaw": self.profile_yaw_threshold,
                "max_roll": self.max_roll,
            },
            "forward_facing_thresholds": {
                "yaw": self.forward_yaw_threshold,
                "pitch": self.forward_pitch_threshold,
            },
        }

    def process_frame_batch(
        self,
        frames_with_faces: List["FrameData"],
        progress_callback: Optional[callable] = None,
        interruption_check: Optional[callable] = None,
    ) -> Tuple[Dict[str, int], int]:
        """Process a batch of frames with head pose estimation at a high level.

        This method handles:
        - Loading images from frame data
        - Extracting face crops from frames
        - Running head pose estimation in batches
        - Updating frame data with head pose results
        - Error handling for individual frames
        - Progress tracking with rate calculation
        - Statistics collection
        - Interruption checking

        Args:
            frames_with_faces: List of FrameData objects containing face detections
            progress_callback: Optional callback for progress updates (called with processed_count)
            interruption_check: Optional callback to check for interruption

        Returns:
            Tuple of (head_angles_by_category, total_head_poses_found)

        Raises:
            HeadPoseEstimationError: If processing fails completely
        """
        if not frames_with_faces:
            return {}, 0

        from pathlib import Path
        import cv2

        total_head_poses_found = 0
        head_angles_by_category = {}
        processed_count = 0
        start_time = time.time()

        # Process frames in batches for memory efficiency
        batch_size = self.config.models.batch_size
        total_frames = len(frames_with_faces)
        total_batches = (total_frames + batch_size - 1) // batch_size

        logger.info(
            f"Starting head pose estimation on {total_frames} frames ({total_batches} batches)"
        )

        for i in range(0, total_frames, batch_size):
            # Check for interruption at the start of each batch
            if interruption_check:
                interruption_check()
                
            batch_num = i // batch_size + 1
            batch_frames = frames_with_faces[i : i + batch_size]

            logger.debug(
                f"Processing head pose batch {batch_num}/{total_batches} ({len(batch_frames)} frames)"
            )

            # Collect face crops from this batch
            batch_face_images = []
            batch_frame_data = []
            batch_face_indices = []

            for frame_data in batch_frames:
                # Check for interruption periodically during frame processing
                if interruption_check and len(batch_face_images) % 10 == 0:
                    interruption_check()
                    
                try:
                    # Load frame image
                    frame_path = frame_data.file_path
                    if not frame_path.exists():
                        continue

                    image = cv2.imread(str(frame_path))
                    if image is None:
                        continue

                    # Extract face crops from this frame
                    face_detections = frame_data.face_detections
                    for face_idx, face_detection in enumerate(face_detections):
                        bbox = face_detection.bbox
                        x1, y1, x2, y2 = bbox

                        # Add padding to face crop
                        padding = 0.3  # 30% padding
                        width = x2 - x1
                        height = y2 - y1
                        pad_x = int(width * padding / 2)
                        pad_y = int(height * padding / 2)

                        # Clamp to image bounds
                        x1_padded = max(0, x1 - pad_x)
                        y1_padded = max(0, y1 - pad_y)
                        x2_padded = min(image.shape[1], x2 + pad_x)
                        y2_padded = min(image.shape[0], y2 + pad_y)

                        # Extract face crop
                        face_crop = image[y1_padded:y2_padded, x1_padded:x2_padded]

                        if face_crop.size > 0:
                            batch_face_images.append(face_crop)
                            batch_frame_data.append(frame_data)
                            batch_face_indices.append(face_idx)

                except Exception as e:
                    logger.warning(
                        f"Failed to process frame {frame_data.file_path}: {e}"
                    )
                    continue

            # Skip batch if no valid face images
            if not batch_face_images:
                processed_count += len(batch_frames)
                if progress_callback:
                    # Calculate current processing rate
                    elapsed = time.time() - start_time
                    current_rate = processed_count / elapsed if elapsed > 0 else 0
                    # Check if progress_callback accepts rate parameter
                    try:
                        progress_callback(processed_count, rate=current_rate)
                    except TypeError:
                        # Fallback to single argument for backwards compatibility
                        progress_callback(processed_count)
                continue

            # Check for interruption before running head pose estimation
            if interruption_check:
                interruption_check()

            # Run head pose estimation on batch of face crops
            try:
                logger.debug(
                    f"Running head pose inference on {len(batch_face_images)} face crops..."
                )
                batch_head_pose_results = self.estimate_batch(batch_face_images)

                batch_head_poses_found = 0
                batch_classifications = {}

                # Process results for each face crop in batch
                for j, (frame_data, face_idx, head_pose_result) in enumerate(
                    zip(batch_frame_data, batch_face_indices, batch_head_pose_results)
                ):
                    # Check for interruption during result processing
                    if interruption_check and j % 5 == 0:
                        interruption_check()
                        
                    if head_pose_result:
                        batch_head_poses_found += 1

                        # Update the face detection with head pose information
                        if (
                            face_idx < len(frame_data.face_detections)
                            and hasattr(frame_data.face_detections[face_idx], "head_pose")
                        ):
                            frame_data.face_detections[face_idx].head_pose = (
                                head_pose_result
                            )

                        # Add to frame's head_poses list
                        frame_data.head_poses.append(head_pose_result)

                        total_head_poses_found += 1

                        # Classify head pose direction using HeadAngleClassifier
                        direction = self._head_angle_classifier.classify_head_angle(
                            head_pose_result.yaw, head_pose_result.pitch, head_pose_result.roll
                        )
                        head_pose_result.direction = direction

                        # Update statistics with the direction classification
                        if direction:
                            head_angles_by_category[direction] = (
                                head_angles_by_category.get(direction, 0) + 1
                            )
                            batch_classifications[direction] = (
                                batch_classifications.get(direction, 0) + 1
                            )

                # Log batch results
                if batch_head_poses_found > 0:
                    logger.debug(
                        f"Batch {batch_num}/{total_batches}: {batch_head_poses_found} head poses found, classifications: {batch_classifications}"
                    )
                else:
                    logger.debug(
                        f"Batch {batch_num}/{total_batches}: No head poses detected"
                    )

            except Exception as e:
                logger.error(
                    f"Head pose estimation failed for batch {batch_num}/{total_batches}: {e}"
                )
                # Continue with next batch rather than failing completely

            # Update progress with rate calculation
            processed_count += len(batch_frames)
            if progress_callback:
                # Calculate current processing rate
                elapsed = time.time() - start_time
                current_rate = processed_count / elapsed if elapsed > 0 else 0
                # Check if progress_callback accepts rate parameter
                try:
                    progress_callback(processed_count, rate=current_rate)
                except TypeError:
                    # Fallback to single argument for backwards compatibility
                    progress_callback(processed_count)

        logger.debug(
            f"Head pose estimation completed: {total_head_poses_found} head poses found in {total_frames} frames"
        )
        logger.debug(f"Final head angle categories: {head_angles_by_category}")

        return head_angles_by_category, total_head_poses_found

    def __del__(self):
        """Cleanup resources when estimator is destroyed."""
        self._model = None


def create_head_pose_estimator(
    model_name: Optional[str] = None,
    device: str = "auto",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    config: Optional["Config"] = None,
) -> HeadPoseEstimator:
    """Factory function to create a head pose estimator.

    Args:
        model_name: Name of the head pose model (None for default)
        device: Computation device ("cpu", "cuda", or "auto")
        confidence_threshold: Minimum confidence threshold
        config: Application configuration object (optional)

    Returns:
        Configured HeadPoseEstimator instance

    Raises:
        HeadPoseEstimationError: If model creation fails
    """
    if model_name is None:
        # Use default from model configs
        defaults = ModelConfigs.get_default_models()
        model_name = defaults.get("head_pose_estimation", "sixdrepnet")

    return HeadPoseEstimator(
        model_name=model_name, device=device, confidence_threshold=confidence_threshold, config=config
    )
