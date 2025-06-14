"""Model configuration and metadata for Person From Vid.

This module defines configuration classes and metadata for all AI models used in the system,
including download URLs, versions, checksums, and device compatibility information.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import hashlib

from ..data.config import DeviceType


class ModelFormat(str, Enum):
    """Supported model formats."""

    ONNX = "onnx"
    PYTORCH = "pt"
    PICKLE = "pkl"
    TENSORFLOW = "tf"
    TORCHSCRIPT = "ts"
    SAFETENSORS = "safetensors"


class ModelProvider(str, Enum):
    """Model sources and providers."""

    ULTRALYTICS = "ultralytics"
    GITHUB = "github"
    DIRECT_URL = "direct_url"


@dataclass
class ModelFile:
    """Represents a single model file within a model."""

    filename: str
    url: str
    sha256_hash: str
    size_bytes: int
    format: ModelFormat
    description: str


@dataclass
class ModelMetadata:
    """Complete metadata for an AI model."""

    name: str
    version: str
    provider: ModelProvider
    files: List[ModelFile]
    supported_devices: List[DeviceType]
    input_size: tuple
    description: str
    license: str
    citation: Optional[str] = None
    requirements: Optional[List[str]] = None

    def get_primary_file(self) -> ModelFile:
        """Get the primary model file (first in list)."""
        if not self.files:
            raise ValueError(f"No files defined for model {self.name}")
        return self.files[0]

    def get_cache_key(self) -> str:
        """Generate a unique cache key for this model version."""
        content = f"{self.name}:{self.version}:{self.provider.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def is_device_supported(self, device: DeviceType) -> bool:
        """Check if the model supports the specified device."""
        return (
            device in self.supported_devices
            or DeviceType.AUTO in self.supported_devices
        )


class ModelConfigs:
    """Registry of all available model configurations."""

    # Face Detection Models
    SCRFD_500M = ModelMetadata(
        name="scrfd_500m",
        version="1.0.0",
        provider=ModelProvider.DIRECT_URL,
        files=[
            ModelFile(
                filename="scrfd_500m.onnx",
                url="https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx",
                sha256_hash="",  # Will be calculated on first download
                size_bytes=0,
                format=ModelFormat.ONNX,
                description="SCRFD 500M - Lightweight face detection model",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(640, 640),
        description="SCRFD 500M - Lightweight face detection model",
        license="Apache-2.0",
        citation="SCRFD: Learning Better Representations for Face Detection",
    )

    SCRFD_2_5G = ModelMetadata(
        name="scrfd_2_5g",
        version="1.0.0",
        provider=ModelProvider.DIRECT_URL,
        files=[
            ModelFile(
                filename="scrfd_2_5g.onnx",
                url="https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx",
                sha256_hash="",  # Will be calculated on first download
                size_bytes=0,
                format=ModelFormat.ONNX,
                description="SCRFD 2.5G - Balanced accuracy and efficiency",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(640, 640),
        description="SCRFD 2.5G - Balanced accuracy and efficiency",
        license="Apache-2.0",
        citation="SCRFD: Learning Better Representations for Face Detection",
    )

    SCRFD_10G = ModelMetadata(
        name="scrfd_10g",
        version="1.0.0",
        provider=ModelProvider.DIRECT_URL,
        files=[
            ModelFile(
                filename="scrfd_10g.onnx",
                url="https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx",
                sha256_hash="",  # Will be calculated on first download
                size_bytes=0,
                format=ModelFormat.ONNX,
                description="SCRFD 10G - High accuracy face detection model",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(640, 640),
        description="SCRFD 10G - High accuracy face detection model",
        license="Apache-2.0",
        citation="SCRFD: Learning Better Representations for Face Detection",
    )

    YOLOFACE_V8N = ModelMetadata(
        name="yolov8n-face",
        version="1.0.0",
        provider=ModelProvider.GITHUB,
        files=[
            ModelFile(
                filename="yolov8n-face-lindevs.pt",
                url="https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt",
                sha256_hash="b038ca653b503453a94f6e12d76feca6840b2a97d7a1322b4498c5e922f29832",
                size_bytes=6291456,  # ~6MB
                format=ModelFormat.PYTORCH,
                description="YOLOv8 Nano face detection model",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(640, 640),
        description="YOLOv8 Nano face detection - Fast and lightweight face detection",
        license="AGPL-3.0",
        requirements=["ultralytics>=8.0.0"],
    )

    YOLOFACE_V8S = ModelMetadata(
        name="yolov8s-face",
        version="1.0.0",
        provider=ModelProvider.GITHUB,
        files=[
            ModelFile(
                filename="yolov8s-face-lindevs.pt",
                url="https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.pt",
                sha256_hash="",  # Will be calculated on first download
                size_bytes=22500000,  # ~22MB
                format=ModelFormat.PYTORCH,
                description="YOLOv8 Small face detection model - higher accuracy",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(640, 640),
        description="YOLOv8 Small face detection - Higher accuracy face detection",
        license="AGPL-3.0",
        requirements=["ultralytics>=8.0.0"],
    )

    # Pose Estimation Models
    YOLOV8N_POSE = ModelMetadata(
        name="yolov8n-pose",
        version="1.0.0",
        provider=ModelProvider.ULTRALYTICS,
        files=[
            ModelFile(
                filename="yolov8n-pose.pt",
                url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
                sha256_hash="",
                size_bytes=6553600,  # ~6.25MB
                format=ModelFormat.PYTORCH,
                description="YOLOv8 Nano pose estimation model",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(640, 640),
        description="YOLOv8 Nano pose estimation - 17-point human pose keypoints",
        license="AGPL-3.0",
        requirements=["ultralytics>=8.0.0"],
    )

    YOLOV8S_POSE = ModelMetadata(
        name="yolov8s-pose",
        version="1.0.0",
        provider=ModelProvider.ULTRALYTICS,
        files=[
            ModelFile(
                filename="yolov8s-pose.pt",
                url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt",
                sha256_hash="",
                size_bytes=23068672,  # ~22MB
                format=ModelFormat.PYTORCH,
                description="YOLOv8 Small pose estimation model - higher accuracy",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(640, 640),
        description="YOLOv8 Small pose estimation - More accurate 17-point human pose keypoints",
        license="AGPL-3.0",
        requirements=["ultralytics>=8.0.0"],
    )

    # Head Pose Estimation Models
    HOPENET_ALPHA1 = ModelMetadata(
        name="hopenet_alpha1",
        version="1.0.0",
        provider=ModelProvider.DIRECT_URL,
        files=[
            ModelFile(
                filename="hopenet_alpha1.pkl",
                url="https://drive.google.com/uc?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvws4nC",
                sha256_hash="",
                size_bytes=46137344,  # ~44MB
                format=ModelFormat.PICKLE,
                description="HopeNet Alpha 1 head pose estimation model",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(224, 224),
        description="HopeNet Alpha 1 head pose estimation - Yaw, pitch, roll angle prediction",
        license="MIT",
        citation="Fine-Grained Head Pose Estimation Without Keypoints",
        requirements=["torch>=1.8.0"],
    )

    SIXDREPNET = ModelMetadata(
        name="sixdrepnet",
        version="1.0.0",
        provider=ModelProvider.DIRECT_URL,
        files=[
            ModelFile(
                filename="sixdrepnet.safetensors",
                url="https://huggingface.co/X01D/6DRepNET-RepVGGA0/resolve/main/model.safetensors",
                sha256_hash="",  # Will be calculated on download
                size_bytes=7340000,  # ~7MB
                format=ModelFormat.PYTORCH,
                description="6DRepNet head pose estimation model (RepVGG-A0 backbone)",
            )
        ],
        supported_devices=[DeviceType.CPU, DeviceType.GPU],
        input_size=(224, 224),
        description="6DRepNet - 6D rotation representation for head pose estimation (RepVGG-A0)",
        license="Apache-2.0",
        citation="6D Rotation Representation For Unconstrained Head Pose Estimation",
    )

    @classmethod
    def get_all_models(cls) -> Dict[str, ModelMetadata]:
        """Get all available model configurations."""
        models = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, ModelMetadata):
                models[attr.name] = attr
        return models

    @classmethod
    def get_model(cls, name: str) -> Optional[ModelMetadata]:
        """Get model configuration by name."""
        models = cls.get_all_models()
        return models.get(name)

    @classmethod
    def get_models_by_type(cls, model_type: str) -> List[ModelMetadata]:
        """Get all models of a specific type (face, pose, head_pose)."""
        all_models = cls.get_all_models()

        type_keywords = {
            "face": ["face", "scrfd", "yoloface"],
            "pose": ["pose"],
            "head_pose": ["hopenet", "sixdrepnet", "head"],
        }

        if model_type not in type_keywords:
            return []

        keywords = type_keywords[model_type]
        matching_models = []

        for model in all_models.values():
            if any(keyword in model.name.lower() for keyword in keywords):
                matching_models.append(model)

        return matching_models

    @classmethod
    def get_default_models(cls) -> Dict[str, str]:
        """Get default model names for each type."""
        return {
            "face_detection": "yolov8s-face",
            "pose_estimation": "yolov8s-pose",
            "head_pose_estimation": "sixdrepnet",
        }

    @classmethod
    def validate_model_config(cls, model_name: str, device: DeviceType) -> bool:
        """Validate that a model configuration is valid for the specified device."""
        model = cls.get_model(model_name)
        if not model:
            return False
        return model.is_device_supported(device)


# Model type mappings for configuration validation
MODEL_TYPE_MAPPING = {
    "face_detection_model": "face",
    "pose_estimation_model": "pose",
    "head_pose_model": "head_pose",
}


def get_model_for_config_key(
    config_key: str, model_name: str
) -> Optional[ModelMetadata]:
    """Get model metadata for a configuration key and model name."""
    return ModelConfigs.get_model(model_name)


def validate_config_models(config_dict: Dict[str, Any]) -> List[str]:
    """Validate all model configurations in a config dictionary.

    Returns:
        List of validation error messages, empty if all valid.
    """
    errors = []
    models_config = config_dict.get("models", {})

    for config_key, model_name in models_config.items():
        if config_key in [
            "face_detection_model",
            "pose_estimation_model",
            "head_pose_model",
        ]:
            model = ModelConfigs.get_model(model_name)
            if not model:
                errors.append(f"Unknown model '{model_name}' for {config_key}")
            elif hasattr(models_config, "device"):
                device = DeviceType(models_config.device)
                if not model.is_device_supported(device):
                    errors.append(
                        f"Model '{model_name}' does not support device '{device.value}'"
                    )

    return errors
