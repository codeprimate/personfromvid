"""Optional model management utilities.

These utilities provide additional model management features that can be used
when needed but are not part of the core ModelManager workflow.
"""

import hashlib
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


def verify_file_integrity(file_path: Path, expected_hash: str) -> bool:
    """Verify SHA256 hash of a file.

    Args:
        file_path: Path to the file to verify
        expected_hash: Expected SHA256 hash

    Returns:
        True if hash matches, False otherwise
    """
    if not file_path.exists():
        return False

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest() == expected_hash
    except Exception as e:
        logger.warning(f"Failed to verify hash for {file_path}: {e}")
        return False


def get_model_download_time(model_dir: Path) -> datetime:
    """Get the download time of a model from directory modification time.

    Args:
        model_dir: Path to the model directory

    Returns:
        Datetime of when the model was downloaded
    """
    if model_dir.exists():
        return datetime.fromtimestamp(model_dir.stat().st_mtime)
    return datetime.now()


def cleanup_old_models(cache_dir: Path, keep_days: int = 30) -> List[str]:
    """Remove models older than specified days.

    Args:
        cache_dir: Model cache directory
        keep_days: Number of days to keep models

    Returns:
        List of removed model names
    """
    from .model_configs import ModelConfigs

    removed_models = []
    cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)

    for model_dir in cache_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Check if model is still in configuration
        if not ModelConfigs.get_model(model_name):
            logger.info(f"Removing obsolete model: {model_name}")
            import shutil

            shutil.rmtree(model_dir)
            removed_models.append(model_name)
            continue

        # Check age
        download_time = get_model_download_time(model_dir)
        if download_time.timestamp() < cutoff_date:
            logger.info(f"Removing old model: {model_name}")
            import shutil

            shutil.rmtree(model_dir)
            removed_models.append(model_name)

    return removed_models


def validate_model_cache(cache_dir: Path) -> Dict[str, bool]:
    """Validate integrity of all cached models using their configured hashes.

    Args:
        cache_dir: Model cache directory

    Returns:
        Dictionary mapping model names to validation status
    """
    from .model_configs import ModelConfigs

    validation_results = {}

    for model_dir in cache_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        metadata = ModelConfigs.get_model(model_name)

        if not metadata:
            validation_results[model_name] = False
            continue

        # Check all files exist and have correct hashes
        is_valid = True
        for file_info in metadata.files:
            file_path = model_dir / file_info.filename
            if not file_path.exists():
                is_valid = False
                break

            if hasattr(file_info, "sha256_hash") and file_info.sha256_hash:
                if not verify_file_integrity(file_path, file_info.sha256_hash):
                    is_valid = False
                    break

        validation_results[model_name] = is_valid

    return validation_results
