"""Consolidated output formatter for Person From Vid.

This module provides a clean, streamlined output format that eliminates
duplicate content and excessive whitespace while maintaining clear progress
indication and transparency about processing work.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from contextlib import contextmanager
import re

from rich.console import Console
from rich.progress import (
    Progress,
    TaskID,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    ProgressColumn,
)
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.table import Table

from ..data import VideoMetadata, ProcessingResult
from ..data.constants import get_total_pipeline_steps


@dataclass
class StepTiming:
    """Timing information for a processing step."""

    start_time: float
    end_time: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def duration_formatted(self) -> str:
        """Get formatted duration string."""
        duration = self.duration_seconds
        if duration < 1:
            return f"{duration*1000:.0f}ms"
        elif duration < 60:
            return f"{duration:.1f}s"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            return f"{minutes}m {seconds:.1f}s"


class CompactProgressColumn(ProgressColumn):
    """A compact progress column showing percentage."""

    def render(self, task):
        """Render progress as compact percentage."""
        if task.total is None:
            return ""
        percentage = int(task.percentage)
        return f"{percentage:3d}%"


class ConsolidatedFormatter:
    """Consolidated output formatter that eliminates duplicate content."""

    def __init__(self, console: Optional[Console] = None, enable_debug: bool = False):
        """Initialize the consolidated formatter.

        Args:
            console: Rich console instance (creates new if None)
            enable_debug: Whether to show debug-level information
        """
        self.console = console or Console()
        self.enable_debug = enable_debug

        # State tracking
        self.step_timings: Dict[str, StepTiming] = {}
        self.overall_start_time: Optional[float] = None

        # Per-step progress objects
        self.step_progress: Optional[Progress] = None
        self.current_step_task: Optional[TaskID] = None
        self.live_display: Optional[Live] = None

        # Step results for summary
        self.step_results: Dict[str, Dict[str, Any]] = {}

    def start_processing(self, video_path: str, config_info: Dict[str, Any]) -> None:
        """Start the processing session with header and configuration."""
        self.overall_start_time = time.time()

        # Print minimal header
        filename = Path(video_path).name
        self.console.print(f"\n🎬 Person From Vid - Processing: {filename}")

        # System check (compact)
        status_items = []
        if config_info.get("gpu_available", False):
            status_items.append("🚀 GPU")
        else:
            status_items.append("💻 CPU")

        device_info = " • ".join(status_items)
        self.console.print(f"   {device_info}")

        # Video info (single line)
        if "video_metadata" in config_info and config_info["video_metadata"]:
            metadata = config_info["video_metadata"]
            size_mb = config_info.get("file_size_mb", 0)

            # Handle both dict and VideoMetadata object formats
            if isinstance(metadata, dict):
                duration = metadata.get("duration", 0)
                width = metadata.get("width", 0)
                height = metadata.get("height", 0)
                fps = metadata.get("fps", 0)
            else:
                # VideoMetadata object
                duration = metadata.duration
                width = metadata.width
                height = metadata.height
                fps = metadata.fps

            duration_str = self._format_duration(duration)
            self.console.print(
                f"   📹 {width}×{height}, {fps:.1f}fps, {duration_str} ({size_mb:.1f}MB)"
            )

        self.console.print()

    def start_step(self, step_number: int, step_name: str, description: str) -> None:
        """Start a new processing step."""
        self.step_timings[step_name] = StepTiming(start_time=time.time())

        # Print a header for the new step
        self.console.print(f"\n[bold]▶ Step {step_number}: {description}[/bold]")

        if self.enable_debug:
            self.console.print(
                f"[dim]DEBUG: Starting step {step_number}: {step_name}[/dim]"
            )

    @contextmanager
    def step_progress_context(self, description: str, total: Optional[int] = None):
        """Context manager for a step's progress bar."""
        self.step_progress = Progress(
            TextColumn("[bold green]  {task.description}"),
            BarColumn(bar_width=40),
            CompactProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("• {task.fields[rate]}", style="cyan"),
            transient=True,
        )

        self.current_step_task = self.step_progress.add_task(
            description, total=total, rate="0/s" if total else "working..."
        )

        self.live_display = Live(
            self.step_progress, console=self.console, transient=True
        )
        self.live_display.start()

        try:
            yield self.current_step_task
        finally:
            if self.live_display:
                self.live_display.stop()
            self.step_progress = None
            self.current_step_task = None
            self.live_display = None

    def update_step_progress(self, advance: int = 1, **kwargs) -> None:
        """Update current step progress."""
        if self.current_step_task is not None and self.step_progress is not None:
            update_data = kwargs.copy()
            if "rate" in update_data:
                update_data["rate"] = f"{update_data['rate']:.1f}/s"

            self.step_progress.update(
                self.current_step_task, advance=advance, **update_data
            )

    def complete_step(
        self, step_name: str, results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete the current step and print summary."""
        # Record end time
        if step_name in self.step_timings:
            self.step_timings[step_name].end_time = time.time()

        # Store results for final summary
        if results:
            self.step_results[step_name] = results

        # Print bullet points with detailed information before completion
        self._print_step_details(step_name, results)

        # Brief completion message
        duration = (
            self.step_timings[step_name].duration_formatted
            if step_name in self.step_timings
            else "unknown"
        )

        summary = ""
        if results and "summary" in results:
            summary = results["summary"]
        else:
            step_display = step_name.replace("_", " ").title()
            summary = f"{step_display} completed"

        self.console.print(f"  ✅ {summary} ({duration})")

        if self.enable_debug and results:
            for key, value in results.items():
                if key != "summary":
                    self.console.print(f"[dim]   DEBUG: {key}: {value}[/dim]")

    def _print_step_details(
        self, step_name: str, results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Print detailed bullet points for each step."""
        if not results:
            return

        # Extract step-specific details and print as bullet points
        if step_name == "initialization":
            self.console.print("  • Loading face detection, pose, and quality models")

        elif step_name == "frame_extraction":
            # Show detailed extraction information if available
            if "extraction_summary" in results:
                # Extract the frame count from the summary
                match = re.search(
                    r"Extracted (\d+) unique frames", results["extraction_summary"]
                )
                if match:
                    frame_count = match.group(1)
                    self.console.print(f"  • Extracted {frame_count} frames from video")
            elif "total_frames" in results:
                self.console.print(
                    f"  • Extracted {results['total_frames']} frames from video"
                )

            # Show I-frames and temporal samples info
            if "i_frames_info" in results:
                # Extract just the number for cleaner display
                match = re.search(r"I-frames found: (\d+)", results["i_frames_info"])
                if match:
                    i_frames = match.group(1)
                    self.console.print(f"  • I-frames found: {i_frames}")

            if "temporal_info" in results:
                # Extract just the number for cleaner display
                match = re.search(r"Temporal samples: (\d+)", results["temporal_info"])
                if match:
                    temporal = match.group(1)
                    self.console.print(f"  • Temporal samples: {temporal}")

            # Show duplicates removed info from extraction summary
            if "extraction_summary" in results:
                match = re.search(
                    r"\((\d+) duplicates removed\)", results["extraction_summary"]
                )
                if match:
                    duplicates = match.group(1)
                    self.console.print(f"  • Duplicates removed: {duplicates}")

        elif step_name == "face_detection":
            if "faces_found" in results and "frames_with_faces" in results:
                self.console.print(
                    f"  • Found {results['faces_found']} faces in {results['frames_with_faces']} frames"
                )

        elif step_name == "pose_analysis":
            if "poses_found" in results and "head_angles_found" in results:
                self.console.print(
                    f"  • Analyzed poses for {results.get('total_analyzed', 0)} detected faces"
                )

                # Show body poses breakdown
                poses = results["poses_found"]
                if poses:
                    pose_items = [f"{name} ({count})" for name, count in poses.items()]
                    self.console.print(f"  • Body poses found: {', '.join(pose_items)}")

                # Show head poses breakdown
                head_angles = results["head_angles_found"]
                if head_angles:
                    head_items = [
                        f"{name} ({count})" for name, count in head_angles.items()
                    ]
                    self.console.print(f"  • Head poses found: {', '.join(head_items)}")

        elif step_name == "closeup_detection":
            if "total_closeups" in results:
                self.console.print(
                    f"  • Identified {results['total_closeups']} frames as containing a closeup"
                )

        elif step_name == "quality_assessment":
            if "total_assessed" in results and "quality_passed" in results:
                self.console.print(
                    f"  • Assessed sharpness and lighting for {results['total_assessed']} faces"
                )
                self.console.print(
                    f"  • {results['quality_passed']} faces passed quality thresholds"
                )
            elif "quality_assessment_summary" in results:
                # Handle the actual data structure from the step
                for key, value in results.items():
                    if key.endswith("_count"):
                        # Extract and display count information
                        if "High quality:" in value or "Usable quality:" in value:
                            # Extract just the meaningful part
                            clean_value = value.replace("📊 ", "").replace(
                                " frames", ""
                            )
                            self.console.print(f"  • {clean_value}")

        elif step_name == "frame_selection":
            if "total_candidates" in results and "total_selected" in results:
                self.console.print(
                    f"  • Selected {results['total_selected']} best frames for output based on pose and quality"
                )
            elif "selected_summary" in results:
                # Handle the actual data structure from the step
                if "candidates_summary" in results:
                    candidates_text = (
                        results["candidates_summary"]
                        .replace("📊 Candidates: ", "")
                        .replace(" frames", "")
                    )
                    self.console.print(
                        f"  • Evaluated {candidates_text} candidate frames"
                    )

                selected_text = (
                    results["selected_summary"]
                    .replace("✅ Selected ", "")
                    .replace(" frames", "")
                )
                self.console.print(
                    f"  • Selected {selected_text} best frames for output based on pose and quality"
                )

        elif step_name == "output_generation":
            if "output_files" in results:
                self.console.print(
                    f"  • Writing {len(results['output_files'])} images to disk:"
                )
                for file_info in results["output_files"]:
                    if (
                        isinstance(file_info, dict)
                        and "filename" in file_info
                        and "frame_number" in file_info
                    ):
                        self.console.print(
                            f"    - {file_info['filename']} (from frame {file_info['frame_number']})"
                        )
                    elif isinstance(file_info, str):
                        # Fallback for simple filename strings
                        self.console.print(f"    - {file_info}")
            elif "files_generated" in results:
                # Handle the actual data structure from the step
                files_text = (
                    results["files_generated"]
                    .replace("✅ Generated ", "")
                    .replace(" files", "")
                )
                self.console.print(f"  • Generated {files_text} output files")

                if "location_info" in results:
                    location_text = results["location_info"].replace(
                        "📂 Location: ", ""
                    )
                    self.console.print(f"  • Saved to: {location_text}")

    def debug(self, message: str) -> None:
        """Print debug message if debug is enabled."""
        if self.enable_debug:
            self.console.print(f"[dim]DEBUG: {message}[/dim]")

    def print_info(self, message: str, emoji_key: str = "success") -> None:
        """Print an info message with optional emoji (compatibility method)."""
        if self.enable_debug:
            # Only show these messages in debug mode to reduce clutter
            self.console.print(f"[dim]{message}[/dim]")

    def update_progress(self, advance: int = 1, **kwargs) -> None:
        """Update current step progress (compatibility method)."""
        self.update_step_progress(advance, **kwargs)

    def create_progress_bar(self, description: str, total: Optional[int] = None):
        """Create a progress bar (compatibility method)."""
        return self.step_progress_context(description, total)

    def print_warning(self, message: str) -> None:
        """Print a warning message (compatibility method)."""
        self.console.print(f"⚠️  {message}")

    def print_error(self, message: str, step_name: Optional[str] = None) -> None:
        """Print error message (compatibility method)."""
        # If a live display is somehow still active, stop it.
        if self.live_display and self.live_display.is_started:
            self.live_display.stop()

        self.console.print(f"❌ Error: {message}")

        if step_name and step_name in self.step_timings:
            self.step_timings[step_name].end_time = time.time()

    def print_completion_summary(
        self, result: ProcessingResult, output_path: str
    ) -> None:
        """Print final processing summary (compatibility method)."""
        # Calculate total time
        total_time = (
            time.time() - self.overall_start_time if self.overall_start_time else 0
        )

        self.console.print()

        if result.success:
            self.console.print("🎉 Processing Complete")

            # Compact results summary - just the key metrics
            self.console.print(
                f"📊 Results: {result.total_frames_extracted} frames, {result.faces_found} faces, {len(result.output_files)} files"
            )

            # Top categories (single line each)
            if result.poses_found:
                body_top = sorted(
                    result.poses_found.items(), key=lambda x: x[1], reverse=True
                )[:3]
                body_str = ", ".join([f"{pose}({count})" for pose, count in body_top])
                self.console.print(f"🏃 Body poses: {body_str}")

            if result.head_angles_found:
                head_top = sorted(
                    result.head_angles_found.items(), key=lambda x: x[1], reverse=True
                )[:3]
                head_str = ", ".join([f"{angle}({count})" for angle, count in head_top])
                self.console.print(f"🎯 Head poses: {head_str}")

            # Output files are now shown during output generation step, so we don't repeat them here
            # Just show the output directory location
            if result.output_files:
                output_dir = (
                    Path(result.output_files[0]).parent
                    if result.output_files
                    else Path(output_path).parent
                )
                self.console.print(f"📂 Output: {output_dir}")
            else:
                self.console.print(f"📂 Output: {output_path}")

            self.console.print(f"⏱️  Total time: {total_time:.1f}s")

        else:
            self.console.print("❌ Processing Failed")
            if result.error_message:
                self.console.print(f"Error: {result.error_message}")

        self.console.print()

    def _format_duration(self, seconds: float) -> str:
        """Format duration consistently."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        if minutes < 60:
            return f"{minutes}m {remaining_seconds:.0f}s"
        hours = int(minutes // 60)
        remaining_minutes = minutes % 60
        return f"{hours}h {remaining_minutes}m"


def create_consolidated_formatter(
    console: Optional[Console] = None, enable_debug: bool = False
) -> ConsolidatedFormatter:
    """Create a consolidated formatter instance.

    Args:
        console: Optional Rich console instance
        enable_debug: Whether to enable debug output

    Returns:
        ConsolidatedFormatter instance
    """
    return ConsolidatedFormatter(console, enable_debug)
