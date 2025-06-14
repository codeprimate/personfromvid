"""Model management system for Person From Vid.

Simple model downloading and caching with automatic availability checking.
"""

import shutil
from pathlib import Path
from typing import Optional, Union, List
from datetime import datetime
import logging
import requests

from tqdm import tqdm

from .model_configs import ModelConfigs, ModelProvider
from ..utils.exceptions import ModelError, ModelNotFoundError, ModelDownloadError


logger = logging.getLogger(__name__)


class ModelManager:
    """Simple model manager focused on downloading and caching."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the model manager.

        Args:
            cache_dir: Directory for model caching. If None, uses default from config.
        """
        if cache_dir is None:
            from ..data.config import get_default_config

            config = get_default_config()
            self.cache_dir = config.storage.cache_directory / "models"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_model_available(self, model_name: str) -> Path:
        """Ensure a model is available, downloading if necessary.

        Args:
            model_name: Name of the model

        Returns:
            Path to the model's primary file
        """
        if self.is_model_cached(model_name):
            return self.get_model_path(model_name)

        return self.download_model(model_name)

    def download_model(self, model_name: str) -> Path:
        """Download a model and return its primary file path.

        Args:
            model_name: Name of the model to download

        Returns:
            Path to the model's primary file

        Raises:
            ModelNotFoundError: If model configuration is not found
            ModelDownloadError: If download fails
        """
        metadata = ModelConfigs.get_model(model_name)
        if not metadata:
            raise ModelNotFoundError(f"Model '{model_name}' not found in configuration")

        model_cache_dir = self.cache_dir / model_name
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading model '{model_name}'...")

        try:
            # Download all files for this model
            for file_info in metadata.files:
                file_path = model_cache_dir / file_info.filename
                self._download_file(file_info, file_path, metadata.provider)

            logger.info(f"Successfully downloaded model '{model_name}'")
            return self.get_model_path(model_name)

        except Exception as e:
            # Clean up on failure
            if model_cache_dir.exists():
                shutil.rmtree(model_cache_dir)
            raise ModelDownloadError(f"Failed to download model '{model_name}': {e}")

    def _download_file(
        self, file_info, file_path: Path, provider: ModelProvider
    ) -> None:
        """Download a single model file."""
        self._download_from_url(file_info, file_path)

    def _download_from_url(self, file_info, file_path: Path) -> None:
        """Download file from direct URL."""
        try:
            response = requests.get(file_info.url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                with tqdm(
                    desc=f"Downloading {file_info.filename}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        except requests.RequestException as e:
            raise ModelDownloadError(f"Failed to download from {file_info.url}: {e}")

    def get_model_path(self, model_name: str) -> Path:
        """Get the path to a cached model's primary file.

        Args:
            model_name: Name of the model

        Returns:
            Path to the model's primary file

        Raises:
            ModelNotFoundError: If model is not cached
        """
        if not self.is_model_cached(model_name):
            raise ModelNotFoundError(f"Model '{model_name}' is not cached")

        metadata = ModelConfigs.get_model(model_name)
        if not metadata:
            raise ModelNotFoundError(f"Model '{model_name}' not found in configuration")

        model_cache_dir = self.cache_dir / model_name
        primary_file = metadata.get_primary_file()
        return model_cache_dir / primary_file.filename

    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is cached locally.

        Args:
            model_name: Name of the model to check

        Returns:
            True if all model files exist
        """
        metadata = ModelConfigs.get_model(model_name)
        if not metadata:
            return False

        model_cache_dir = self.cache_dir / model_name
        if not model_cache_dir.exists():
            return False

        # Check if all required files exist
        for file_info in metadata.files:
            file_path = model_cache_dir / file_info.filename
            if not file_path.exists():
                return False

        return True

    def list_cached_models(self) -> List[str]:
        """Get list of all cached model names."""
        cached_models = []
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir() and self.is_model_cached(model_dir.name):
                cached_models.append(model_dir.name)
        return cached_models

    def get_cache_size(self) -> int:
        """Get total size of model cache in bytes."""
        total_size = 0
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        return total_size

    def clear_cache(self) -> None:
        """Clear the entire model cache."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model cache cleared")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager(cache_dir: Optional[Union[str, Path]] = None) -> ModelManager:
    """Get the global model manager instance.

    Args:
        cache_dir: Cache directory override

    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None or cache_dir is not None:
        _model_manager = ModelManager(cache_dir)
    return _model_manager
