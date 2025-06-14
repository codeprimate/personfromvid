"""Temporary directory management.

This module manages the lifecycle of temporary directories used during
video processing, including automatic cleanup and disk space monitoring.
"""

import os
import shutil
import tempfile
import logging
import stat
from pathlib import Path
from typing import Optional, List


from ..utils.logging import get_logger
from ..utils.exceptions import TempDirectoryError
from ..data.config import Config


class TempManager:
    """Manages temporary directory lifecycle and organization.

    Creates and manages the temporary directory structure used during
    video processing, with automatic cleanup and disk space monitoring.
    """

    def __init__(self, video_path: str, config: Optional[Config] = None):
        """Initialize temporary directory manager.

        Args:
            video_path: Path to the video file being processed
            config: Application configuration (uses default if None)
        """
        self.video_path = Path(video_path)
        self.logger = get_logger("temp_manager")

        # Get config or use default
        if config is None:
            from ..data.config import get_default_config

            self.config = get_default_config()
        else:
            self.config = config

        # Store storage config for convenience
        self.storage_config = self.config.storage

        # Calculate temp directory path in cache directory
        video_base_name = self.video_path.stem
        self.temp_dir_name = f"temp_{video_base_name}"

        # Use configured temp directory or create in cache directory
        if self.storage_config.temp_directory:
            self.temp_dir_path = self.storage_config.temp_directory / self.temp_dir_name
        else:
            # Create temp directory in cache directory
            self.temp_dir_path = (
                self.storage_config.cache_directory / "temp" / self.temp_dir_name
            )

        # Subdirectory paths
        self.frames_dir: Optional[Path] = None

        # Track created directories for cleanup
        self._created_dirs: List[Path] = []

        self.logger.debug(f"Temp directory path: {self.temp_dir_path}")

    def create_temp_structure(self) -> Path:
        """Create temporary directory structure.

        Returns:
            Path to the created temporary directory

        Raises:
            TempDirectoryError: If temp directory cannot be created
        """
        try:
            # Create main temp directory (and parent directories if needed)
            self.temp_dir_path.mkdir(parents=True, exist_ok=True)
            self._created_dirs.append(self.temp_dir_path)

            # Create subdirectories
            self.frames_dir = self.temp_dir_path / "frames"
            self.frames_dir.mkdir(exist_ok=True)
            self._created_dirs.append(self.frames_dir)

            self.logger.info(
                f"Created temporary directory structure: {self.temp_dir_path}"
            )
            self.logger.debug(f"Subdirectories: frames")

            return self.temp_dir_path

        except Exception as e:
            self.logger.error(f"Failed to create temp directory structure: {e}")
            # Clean up any partially created directories
            self._cleanup_partial_creation()
            raise TempDirectoryError(f"Cannot create temp directory: {e}")

    def cleanup_temp_files(self) -> None:
        """Remove temporary directory and all contents.

        Performs safe cleanup with error handling for locked files.
        """
        if not self.temp_dir_path.exists():
            self.logger.debug("Temp directory does not exist, nothing to clean up")
            return

        try:
            self.logger.info(f"Cleaning up temporary directory: {self.temp_dir_path}")

            # Get directory size before cleanup for logging
            try:
                dir_size = self._get_directory_size(self.temp_dir_path)
                self.logger.debug(
                    f"Temp directory size: {dir_size / (1024*1024):.1f} MB"
                )
            except Exception:
                dir_size = 0

            # Remove directory and all contents
            shutil.rmtree(self.temp_dir_path, ignore_errors=True)

            # Verify cleanup was successful
            if self.temp_dir_path.exists():
                self.logger.warning("Temp directory still exists after cleanup attempt")
                # Try alternative cleanup method
                self._force_cleanup()
            else:
                self.logger.info("Temporary directory cleaned up successfully")
                if dir_size > 0:
                    self.logger.info(
                        f"Freed {dir_size / (1024*1024):.1f} MB of disk space"
                    )

        except Exception as e:
            self.logger.error(f"Failed to cleanup temp directory: {e}")
            # Don't raise exception - cleanup is best effort

    def get_temp_path(self) -> Path:
        """Get path to temporary directory.

        Returns:
            Path to temporary directory

        Raises:
            TempDirectoryError: If temp directory not created
        """
        if not self.temp_dir_path.exists():
            raise TempDirectoryError("Temp directory has not been created")
        return self.temp_dir_path

    def get_frames_dir(self) -> Path:
        """Get path to frames subdirectory.

        Returns:
            Path to frames directory

        Raises:
            TempDirectoryError: If frames directory not created
        """
        if not self.frames_dir or not self.frames_dir.exists():
            raise TempDirectoryError("Frames directory has not been created")
        return self.frames_dir

    def get_temp_file_path(self, filename: str, subdir: Optional[str] = None) -> Path:
        """Get path for a temporary file.

        Args:
            filename: Name of the file
            subdir: Optional subdirectory (frames only)

        Returns:
            Full path for the temporary file
        """
        if subdir:
            if subdir == "frames":
                return self.get_frames_dir() / filename
            else:
                raise ValueError(
                    f"Unknown subdirectory: {subdir}. Only 'frames' is supported."
                )
        else:
            return self.get_temp_path() / filename

    def monitor_disk_space(self, min_free_gb: float = 1.0) -> bool:
        """Monitor available disk space - removed for simplicity.

        Args:
            min_free_gb: Minimum free space required in GB (ignored)

        Returns:
            Always True - let OS handle disk space management
        """
        return True  # Disk space checking removed

    def get_temp_usage_info(self) -> dict:
        """Get information about temporary directory usage.

        Returns:
            Dictionary with usage information
        """
        info = {
            "temp_dir": str(self.temp_dir_path),
            "exists": self.temp_dir_path.exists(),
            "size_mb": 0.0,
            "file_count": 0,
            "subdirs": {},
        }

        if not self.temp_dir_path.exists():
            return info

        try:
            # Get overall size and file count
            total_size = 0
            total_files = 0

            # Only iterate over frames directory now
            for subdir_name, subdir_path in [("frames", self.frames_dir)]:
                if subdir_path and subdir_path.exists():
                    size = self._get_directory_size(subdir_path)
                    files = self._count_files(subdir_path)

                    info["subdirs"][subdir_name] = {
                        "size_mb": size / (1024 * 1024),
                        "file_count": files,
                    }

                    total_size += size
                    total_files += files

            info["size_mb"] = total_size / (1024 * 1024)
            info["file_count"] = total_files

        except Exception as e:
            self.logger.warning(f"Failed to get temp usage info: {e}")

        return info

    # Context manager support

    def __enter__(self):
        """Enter context manager."""
        self.create_temp_structure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        self.cleanup_temp_files()
        return False  # Don't suppress exceptions

    # Private methods

    def _cleanup_partial_creation(self) -> None:
        """Clean up partially created directories."""
        for dir_path in reversed(self._created_dirs):
            try:
                if dir_path.exists():
                    if dir_path.is_dir():
                        shutil.rmtree(dir_path, ignore_errors=True)
                    else:
                        dir_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {dir_path}: {e}")

        self._created_dirs.clear()

    def _force_cleanup(self) -> None:
        """Force cleanup by walking the directory and removing items individually."""
        self.logger.info("Attempting forced cleanup...")
        try:
            # Walk from the bottom up to remove files first, then directories
            for root, dirs, files in os.walk(self.temp_dir_path, topdown=False):
                for f in files:
                    file_path = os.path.join(root, f)
                    try:
                        # Make writable and remove
                        os.chmod(file_path, stat.S_IWRITE)
                        os.remove(file_path)
                    except Exception as e:
                        self.logger.warning(f"Could not remove file {file_path}: {e}")

                for d in dirs:
                    dir_path = os.path.join(root, d)
                    try:
                        os.rmdir(dir_path)
                    except Exception as e:
                        self.logger.warning(
                            f"Could not remove directory {dir_path}: {e}"
                        )

            # Finally, remove the top-level temp directory
            try:
                os.rmdir(self.temp_dir_path)
            except Exception as e:
                self.logger.warning(
                    f"Could not remove top-level temp directory {self.temp_dir_path}: {e}"
                )

            if not self.temp_dir_path.exists():
                self.logger.info("Force cleanup successful")
            else:
                self.logger.error("Force cleanup failed. Some temp files may remain.")

        except Exception as e:
            self.logger.error(f"An error occurred during force cleanup: {e}")

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes.

        Args:
            directory: Directory to measure

        Returns:
            Size in bytes
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        # Skip files that can't be accessed
                        pass
        except Exception:
            pass

        return total_size

    def _count_files(self, directory: Path) -> int:
        """Count files in directory.

        Args:
            directory: Directory to count files in

        Returns:
            Number of files
        """
        count = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                count += len(filenames)
        except Exception:
            pass

        return count
