"""Pipeline state persistence and resumption.

This module manages pipeline state saving, loading, and validation to enable
resumption from interruptions.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime

from ..data import PipelineState, VideoMetadata, ProcessingContext
from ..utils.logging import get_logger
from ..utils.exceptions import (
    StateManagementError,
    StateLoadError,
    StateSaveError,
    VideoFileError,
)

# Avoid circular imports
if TYPE_CHECKING:
    from .temp_manager import TempManager


class StateManager:
    """Manages pipeline state persistence and resumption.

    Handles reading/writing info JSON files, validating video file integrity,
    and tracking step completion status for graceful resumption.
    """

    def __init__(self, context: ProcessingContext):
        """Initialize state manager.

        Args:
            context: ProcessingContext with unified pipeline data
        """
        self.video_path = context.video_path
        self.temp_manager = context.temp_manager
        self.video_base_name = context.video_base_name
        self.logger = get_logger("state_manager")
        self.state_file_path = (
            self.temp_manager.get_temp_path() / f"{self.video_base_name}_info.json"
        )
        self.logger.debug(f"State file path: {self.state_file_path}")

    def load_state(self) -> Optional[PipelineState]:
        """Load pipeline state from JSON file.

        Returns:
            PipelineState if found and valid, None otherwise

        Raises:
            StateLoadError: If state file exists but cannot be loaded
        """
        if not self.state_file_path.exists():
            self.logger.debug("No existing state file found")
            return None

        try:
            self.logger.info(f"Loading pipeline state from: {self.state_file_path}")

            # Load state from file
            state = PipelineState.load_from_file(self.state_file_path)

            # Validate state integrity
            self._validate_state(state)

            self.logger.info("Pipeline state loaded successfully")
            self.logger.info(f"State created: {state.created_at}")
            self.logger.info(f"Last updated: {state.last_updated}")
            self.logger.info(f"Current step: {state.current_step}")
            self.logger.info(f"Completed steps: {len(state.completed_steps)}")

            return state

        except Exception as e:
            self.logger.error(f"Failed to load pipeline state: {e}")
            raise StateLoadError(f"Cannot load state from {self.state_file_path}: {e}")

    def save_state(self, state: PipelineState) -> None:
        """Save pipeline state to JSON file.

        Args:
            state: Pipeline state to save

        Raises:
            StateSaveError: If state cannot be saved
        """
        try:
            # Update last modified timestamp
            state.last_updated = datetime.now()

            # Create backup of existing file if it exists
            self._create_backup_if_exists()

            # Save state to file
            state.save_to_file(self.state_file_path)

            self.logger.debug(f"Pipeline state saved to: {self.state_file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save pipeline state: {e}")

            # Try to restore backup if save failed
            self._restore_backup_if_exists()

            raise StateSaveError(f"Cannot save state to {self.state_file_path}: {e}")

    def update_step_progress(self, step: str, progress: Dict[str, Any]) -> None:
        """Update progress for a specific step.

        Args:
            step: Step name
            progress: Progress data to update
        """
        # Load current state
        state = self.load_state()
        if not state:
            self.logger.warning("No state to update")
            return

        # Update step progress
        if step in state.step_progress:
            for key, value in progress.items():
                if key == "processed_count":
                    state.step_progress[step].update_progress(value)
                else:
                    state.step_progress[step].set_data(key, value)

        # Save updated state
        self.save_state(state)

    def mark_step_complete(self, step: str) -> None:
        """Mark a step as completed.

        Args:
            step: Step name to mark as complete
        """
        # Load current state
        state = self.load_state()
        if not state:
            self.logger.warning("No state to update")
            return

        # Mark step complete
        state.complete_step(step)

        # Save updated state
        self.save_state(state)

    def can_resume(self) -> bool:
        """Check if processing can be resumed from existing state.

        Returns:
            True if resumable state exists, False otherwise
        """
        try:
            state = self.load_state()
            return state is not None and state.can_resume()
        except Exception:
            return False

    def get_resume_point(self) -> Optional[str]:
        """Get the step to resume processing from.

        Returns:
            Step name to resume from, or None if no resumable state
        """
        try:
            state = self.load_state()
            if state:
                return state.get_resume_point()
            return None
        except Exception:
            return None

    def delete_state(self) -> None:
        """Delete the state file.

        Used when starting fresh processing or after successful completion.
        """
        try:
            if self.state_file_path.exists():
                self.state_file_path.unlink()
                self.logger.info(f"Deleted state file: {self.state_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to delete state file: {e}")

    def get_state_info(self) -> Optional[Dict[str, Any]]:
        """Get basic information about existing state.

        Returns:
            Dictionary with state information, or None if no state exists
        """
        if not self.state_file_path.exists():
            return None

        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "video_file": data.get("video_file"),
                "created_at": data.get("created_at"),
                "last_updated": data.get("last_updated"),
                "current_step": data.get("current_step"),
                "completed_steps": data.get("completed_steps", []),
                "can_resume": len(data.get("completed_steps", [])) > 0,
            }
        except Exception as e:
            self.logger.warning(f"Failed to get state info: {e}")
            return None

    # Private methods

    def _validate_state(self, state: PipelineState) -> None:
        """Validate state integrity and consistency.

        Args:
            state: State to validate

        Raises:
            StateLoadError: If state is invalid
        """
        # Check if video file still exists
        if not self.video_path.exists():
            raise StateLoadError(f"Video file no longer exists: {self.video_path}")

        # Check if video file path matches
        if Path(state.video_file) != self.video_path:
            self.logger.warning(
                f"Video path mismatch: state={state.video_file}, actual={self.video_path}"
            )

        # Validate video file integrity with hash
        try:
            current_hash = self._calculate_video_hash()
            if current_hash != state.video_hash:
                raise StateLoadError(
                    f"Video file has been modified since state was created. "
                    f"Expected hash: {state.video_hash[:16]}..., "
                    f"Actual hash: {current_hash[:16]}..."
                )
            self.logger.debug("Video file integrity verified")

        except Exception as e:
            if isinstance(e, StateLoadError):
                raise
            self.logger.warning(f"Could not verify video file integrity: {e}")

    def _calculate_video_hash(self) -> str:
        """Calculate SHA256 hash of video file.

        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()

        try:
            with open(self.video_path, "rb") as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            raise VideoFileError(f"Cannot read video file for hashing: {e}")

        return hash_sha256.hexdigest()

    def _create_backup_if_exists(self) -> None:
        """Create backup of existing state file."""
        if self.state_file_path.exists():
            backup_path = self.state_file_path.with_suffix(".json.backup")
            try:
                # Read and write to create backup
                with open(self.state_file_path, "r", encoding="utf-8") as src:
                    content = src.read()
                with open(backup_path, "w", encoding="utf-8") as dst:
                    dst.write(content)

                self.logger.debug(f"Created backup: {backup_path}")

            except Exception as e:
                self.logger.warning(f"Failed to create backup: {e}")

    def _restore_backup_if_exists(self) -> None:
        """Restore backup state file if it exists."""
        backup_path = self.state_file_path.with_suffix(".json.backup")

        if backup_path.exists():
            try:
                # Read backup and restore to main file
                with open(backup_path, "r", encoding="utf-8") as src:
                    content = src.read()
                with open(self.state_file_path, "w", encoding="utf-8") as dst:
                    dst.write(content)

                self.logger.info(f"Restored backup state file")

                # Remove backup after successful restore
                backup_path.unlink()

            except Exception as e:
                self.logger.error(f"Failed to restore backup: {e}")

    def cleanup_backups(self) -> None:
        """Clean up backup files."""
        backup_path = self.state_file_path.with_suffix(".json.backup")

        if backup_path.exists():
            try:
                backup_path.unlink()
                self.logger.debug("Cleaned up backup file")
            except Exception as e:
                self.logger.warning(f"Failed to clean up backup: {e}")

    # Context manager support for state operations

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup."""
        if exc_type is None:
            # Success - cleanup backups
            self.cleanup_backups()
        else:
            # Error - try to restore backup
            self._restore_backup_if_exists()

        return False  # Don't suppress exceptions
