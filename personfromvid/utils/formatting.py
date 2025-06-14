"""Formatting utilities for Person From Vid console output.

This module provides structured, visually appealing output formatting
with emojis, separators, and consistent styling across all processing steps.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from rich.console import Console
from rich.progress import (
    Progress,
    TaskID,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

from ..data import VideoMetadata, ProcessingResult
from ..data.constants import get_total_pipeline_steps


# Visual constants
MAIN_SEPARATOR = "═" * 79
STEP_SEPARATOR = "─" * 79

# Emoji mapping for consistent theming
EMOJIS = {
    # Application/General
    "app": "🎬",
    "analysis": "🔍",
    "video": "📹",
    "setup": "🔧",
    "stats": "📊",
    "progress": "🔄",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    # Specific processing
    "face": "👤",
    "body_pose": "🏃",
    "targeting": "🎯",
    "files": "📂",
    "image": "🖼️",
    "timing": "⏱️",
    "celebration": "🎉",
    "trophy": "🏆",
    "folder": "📁",
    "art": "🎨",
}


@dataclass
class StepTiming:
    """Tracks timing information for a processing step."""

    start_time: float
    end_time: Optional[float] = None

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string."""
        duration = self.duration
        if duration < 1.0:
            return f"{duration:.1f}s"
        elif duration < 60:
            return f"{duration:.1f}s"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            return f"{minutes}m {seconds:.1f}s"


class RichFormatter:
    """Main formatter class for structured console output."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize formatter with rich console.

        Args:
            console: Rich console instance (creates new if None)
        """
        self.console = console or Console()
        self.step_timings: Dict[str, StepTiming] = {}
        self.overall_start_time: Optional[float] = None

        # Progress tracking
        self.current_progress: Optional[Progress] = None
        self.current_task: Optional[TaskID] = None

    def print_app_header(self, video_path: str) -> None:
        """Print the application header."""
        filename = Path(video_path).name
        self.console.print(f"\npersonfromvid {video_path}")
        self.console.print()
        self.console.print(f"{EMOJIS['app']} Person From Vid")
        self.console.print(
            "   AI-powered video frame extraction and pose categorization"
        )
        self.console.print()

    def print_system_check(
        self, gpu_available: bool = False, models_ready: bool = True
    ) -> None:
        """Print system check results."""
        self.console.print(f"{EMOJIS['analysis']} System Check")
        self.console.print(f"   {EMOJIS['success']} Python environment ready")

        if gpu_available:
            self.console.print(f"   {EMOJIS['success']} GPU acceleration available")
        else:
            self.console.print(
                f"   {EMOJIS['warning']} GPU acceleration unavailable (CPU fallback active)"
            )

        if models_ready:
            self.console.print(f"   {EMOJIS['success']} Required models accessible")
        else:
            self.console.print(f"   {EMOJIS['warning']} Some models need downloading")

        self.console.print()

    def print_video_analysis(
        self, metadata: VideoMetadata, file_size_mb: float
    ) -> None:
        """Print video analysis information."""
        self.console.print(f"{EMOJIS['video']} Video Analysis")
        self.console.print(
            f"   {EMOJIS['success']} File: {metadata.file_path} ({file_size_mb:.1f} MB)"
        )

        # Format duration
        duration_str = self._format_duration(metadata.duration_seconds)
        resolution_str = f"{metadata.width}×{metadata.height}"

        self.console.print(
            f"   {EMOJIS['success']} Format: {resolution_str}, {metadata.fps:.1f}fps, {duration_str}, {metadata.codec}"
        )
        self.console.print()

    def print_processing_config(self, config_info: Dict[str, Any]) -> None:
        """Print processing configuration."""
        self.console.print(f"{EMOJIS['setup']} Processing Configuration")

        output_dir = config_info.get("output_directory", "Not specified")
        self.console.print(f"   {EMOJIS['folder']} Output Directory: {output_dir}")

        threshold = config_info.get("quality_threshold", 0.3)
        self.console.print(f"   {EMOJIS['targeting']} Quality Threshold: {threshold}")

        formats = config_info.get("output_formats", ["PNG (face crops + full frames)"])
        formats_str = ", ".join(formats)
        self.console.print(f"   {EMOJIS['image']} Output Formats: {formats_str}")

        resume = config_info.get("resume_enabled", True)
        resume_str = "enabled" if resume else "disabled"
        self.console.print(f"   {EMOJIS['progress']} Resume: {resume_str}")

        device = config_info.get("device", "CPU")
        self.console.print(f"   🚀 Device: {device} (auto-detected)")

        self.console.print()
        self.console.print(MAIN_SEPARATOR)
        self.console.print()

    def start_step(self, step_number: int, step_name: str, description: str) -> None:
        """Start a new processing step."""
        total_steps = get_total_pipeline_steps()

        self.console.print(f"Step {step_number}/{total_steps}: {step_name}")
        self.console.print()
        self.console.print(description)

        # Record start time
        self.step_timings[step_name] = StepTiming(start_time=time.time())

        if self.overall_start_time is None:
            self.overall_start_time = time.time()

    def create_progress_bar(
        self, description: str, total: Optional[int] = None
    ) -> Progress:
        """Create a rich progress bar for the current step.

        Args:
            description: Description for the progress bar
            total: Total number of items (None for indeterminate)

        Returns:
            Progress instance with active task
        """
        if total is not None:
            # Determinate progress bar
            progress = Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            )
            task = progress.add_task(description, total=total)
        else:
            # Indeterminate progress bar (spinner)
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                console=self.console,
                transient=True,
            )
            task = progress.add_task(description)

        self.current_progress = progress
        self.current_task = task
        return progress

    def update_progress(self, advance: int = 1, **kwargs) -> None:
        """Update the current progress bar."""
        if self.current_progress and self.current_task is not None:
            self.current_progress.update(self.current_task, advance=advance, **kwargs)

    def complete_step(
        self,
        step_name: str,
        results: Optional[Dict[str, Any]] = None,
        step_state: Optional[Any] = None,
    ) -> None:
        """Complete the current step and show results."""
        # Stop any active progress
        if self.current_progress:
            self.current_progress.stop()
            self.current_progress = None
            self.current_task = None

        # Record end time
        if step_name in self.step_timings:
            self.step_timings[step_name].end_time = time.time()

        # Try to get results from step state if not provided directly
        if not results and step_state:
            try:
                step_progress = step_state.get_step_progress(step_name)
                if (
                    hasattr(step_progress, "data")
                    and "step_results" in step_progress.data
                ):
                    results = step_progress.data["step_results"]
            except:
                pass  # Ignore errors in extracting step results

        # Print results if available
        if results:
            for key, value in results.items():
                if isinstance(value, str):
                    self.console.print(f"{EMOJIS['success']} {value}")
                elif isinstance(value, dict):
                    # Handle nested results
                    for subkey, subvalue in value.items():
                        self.console.print(f"{EMOJIS['stats']} {subkey}: {subvalue}")

        # Print timing
        if step_name in self.step_timings:
            duration = self.step_timings[step_name].duration_formatted
            self.console.print()
            self.console.print(f"{EMOJIS['timing']} Completed in {duration}")

        self.console.print()
        self.console.print(STEP_SEPARATOR)
        self.console.print()

    def print_completion_summary(
        self, result: ProcessingResult, output_path: str
    ) -> None:
        """Print the final completion summary."""
        self.console.print(MAIN_SEPARATOR)
        self.console.print()

        if result.success:
            self.console.print(f"{EMOJIS['celebration']} Processing Complete")
            self.console.print()

            video_name = Path(
                result.video_file if hasattr(result, "video_file") else output_path
            ).name
            self.console.print(
                f"{EMOJIS['celebration']} Successfully processed: {video_name}"
            )
            self.console.print()

            # Results summary
            self.console.print(f"{EMOJIS['stats']} Results Summary")
            self.console.print(f"   • Frames analyzed: {result.total_frames_extracted}")
            self.console.print(f"   • Faces detected: {result.faces_found}")
            self.console.print(
                f"   • Poses categorized: {sum(result.poses_found.values()) + sum(result.head_angles_found.values())}"
            )
            self.console.print(f"   • Files generated: {len(result.output_files)}")
            self.console.print(
                f"   • Total time: {result.processing_time_seconds:.1f}s"
            )
            self.console.print()

            # Top categories
            if result.poses_found or result.head_angles_found:
                self.console.print(f"{EMOJIS['trophy']} Top Categories")

                if result.poses_found:
                    body_top = sorted(
                        result.poses_found.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    body_str = ", ".join(
                        [f"{pose} ({count})" for pose, count in body_top]
                    )
                    self.console.print(f"   • Body: {body_str}")

                if result.head_angles_found:
                    head_top = sorted(
                        result.head_angles_found.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    head_str = ", ".join(
                        [f"{angle} ({count})" for angle, count in head_top]
                    )
                    self.console.print(f"   • Head: {head_str}")

                self.console.print()

            self.console.print(f"{EMOJIS['folder']} Output: {output_path}")
        else:
            self.console.print(f"{EMOJIS['error']} Processing Failed")
            self.console.print()
            if result.error_message:
                self.console.print(f"Error: {result.error_message}")
            self.console.print()

    def print_error(self, message: str, step_name: Optional[str] = None) -> None:
        """Print an error message."""
        self.console.print(f"{EMOJIS['error']} Error: {message}")

        if step_name and step_name in self.step_timings:
            self.step_timings[step_name].end_time = time.time()

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"{EMOJIS['warning']} {message}")

    def print_info(self, message: str, emoji_key: str = "success") -> None:
        """Print an info message with optional emoji."""
        emoji = EMOJIS.get(emoji_key, EMOJIS["success"])
        self.console.print(f"{emoji} {message}")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"

        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        if minutes < 60:
            return f"{minutes}m {remaining_seconds}s"

        hours = minutes // 60
        remaining_minutes = minutes % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def create_formatter(console: Optional[Console] = None) -> RichFormatter:
    """Create a new RichFormatter instance.

    Args:
        console: Rich console instance (creates new if None)

    Returns:
        Configured RichFormatter instance
    """
    return RichFormatter(console)


# Convenience functions for common formatting operations
def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def format_percentage(value: float, total: float) -> str:
    """Format a percentage with one decimal place."""
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def format_count_summary(items: Dict[str, int], max_items: int = 3) -> str:
    """Format a dictionary of counts into a summary string."""
    if not items:
        return "none"

    sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:max_items]

    return ", ".join([f"{name} ({count})" for name, count in top_items])
