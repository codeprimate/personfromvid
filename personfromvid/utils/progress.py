"""Progress display management using Rich library.

This module provides rich console progress displays for the video processing
pipeline with multi-step tracking, statistics panels, and graceful output.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
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
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns

from ..data import PipelineState, StepProgress, ProcessingResult
from ..utils.logging import get_logger
from ..data.constants import get_total_pipeline_steps


@dataclass
class ProgressStats:
    """Statistics for progress display."""

    items_per_second: float = 0.0
    eta_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0
    peak_rate: float = 0.0

    @property
    def eta_formatted(self) -> str:
        """Get formatted ETA string."""
        if self.eta_seconds is None:
            return "Unknown"
        if self.eta_seconds < 60:
            return f"{int(self.eta_seconds)}s"
        minutes = int(self.eta_seconds // 60)
        seconds = int(self.eta_seconds % 60)
        return f"{minutes}m {seconds}s"


class ProgressManager:
    """Manages rich console progress displays for pipeline processing.

    Provides multi-step progress tracking with statistics panels and
    graceful console output management that works with logging.
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize progress manager.

        Args:
            console: Rich console instance (creates new if None)
        """
        self.console = console or Console()
        self.logger = get_logger("progress")

        # Progress tracking
        self.main_progress: Optional[Progress] = None
        self.step_progress: Optional[Progress] = None
        self.current_task: Optional[TaskID] = None
        self.live_display: Optional[Live] = None

        # Statistics tracking
        self.step_stats: Dict[str, ProgressStats] = {}
        self.step_start_times: Dict[str, float] = {}
        self.overall_start_time: Optional[float] = None

        # Display state
        self.is_active = False
        self.current_step = "initialization"
        self.pipeline_state: Optional[PipelineState] = None

        # Panel content
        self.stats_table: Optional[Table] = None

    def start_pipeline_progress(self, pipeline_state: PipelineState) -> None:
        """Start the main pipeline progress display.

        Args:
            pipeline_state: Current pipeline state
        """
        self.pipeline_state = pipeline_state
        self.overall_start_time = time.time()
        self.is_active = True

        # Create main progress for overall pipeline
        self.main_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )

        # Create step-specific progress
        self.step_progress = Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("{task.fields[rate]}", style="cyan"),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="main", size=3),
            Layout(name="step", size=3),
            Layout(name="stats", size=10),
        )

        # Start live display
        self.live_display = Live(
            layout, console=self.console, refresh_per_second=4, screen=False
        )

        self.logger.info("Started pipeline progress display")

    def update_pipeline_state(self, pipeline_state: PipelineState) -> None:
        """Update the display with current pipeline state.

        Args:
            pipeline_state: Updated pipeline state
        """
        self.pipeline_state = pipeline_state

        if not self.is_active or not self.live_display:
            return

        # Update main progress
        if self.main_progress:
            # Calculate overall progress
            total_steps = get_total_pipeline_steps()
            completed_steps = len(pipeline_state.completed_steps)
            current_step_progress = 0.0

            if pipeline_state.current_step in pipeline_state.step_progress:
                current_step_progress = (
                    pipeline_state.step_progress[
                        pipeline_state.current_step
                    ].progress_percentage
                    / 100.0
                )

            overall_progress = (completed_steps + current_step_progress) / total_steps
            overall_description = f"Processing {pipeline_state.video_file.split('/')[-1]} - {pipeline_state.current_step}"

            # Update or create main task
            main_tasks = list(self.main_progress.tasks)
            if main_tasks:
                self.main_progress.update(
                    main_tasks[0].id,
                    description=overall_description,
                    completed=int(overall_progress * 100),
                    total=100,
                )
            else:
                self.main_progress.add_task(
                    overall_description,
                    total=100,
                    completed=int(overall_progress * 100),
                )

        # Update step progress
        self._update_step_progress(
            pipeline_state.current_step, pipeline_state.step_progress
        )

        # Update statistics
        self._update_statistics_panel()

        # Update layout
        if self.live_display and hasattr(self.live_display, "renderable"):
            layout = self.live_display.renderable
            layout["main"].update(
                Panel(self.main_progress, title="Overall Progress", border_style="blue")
            )
            layout["step"].update(
                Panel(self.step_progress, title="Current Step", border_style="green")
            )
            layout["stats"].update(
                Panel(
                    self.stats_table or "",
                    title="Processing Statistics",
                    border_style="yellow",
                )
            )

    def start_step_progress(
        self, step_name: str, total_items: int, description: str = ""
    ) -> None:
        """Start progress tracking for a specific step.

        Args:
            step_name: Name of the processing step
            total_items: Total number of items to process
            description: Optional description for the step
        """
        self.current_step = step_name
        self.step_start_times[step_name] = time.time()

        if step_name not in self.step_stats:
            self.step_stats[step_name] = ProgressStats()

        if not self.step_progress:
            return

        # Clear previous tasks
        for task in list(self.step_progress.tasks):
            self.step_progress.remove_task(task.id)

        # Create new task for this step
        step_description = description or f"{step_name.replace('_', ' ').title()}"

        self.current_task = self.step_progress.add_task(
            step_description, total=total_items, rate="0.0/s"
        )

        self.logger.debug(f"Started step progress: {step_name} ({total_items} items)")

    def update_step_progress(
        self,
        step_name: str,
        processed_count: int,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update progress for the current step.

        Args:
            step_name: Name of the processing step
            processed_count: Number of items processed so far
            extra_info: Additional information to display
        """
        if not self.is_active or not self.current_task or not self.step_progress:
            return

        # Calculate processing rate
        if step_name in self.step_start_times:
            elapsed = time.time() - self.step_start_times[step_name]
            if elapsed > 0:
                rate = processed_count / elapsed
                self.step_stats[step_name].items_per_second = rate
                self.step_stats[step_name].elapsed_seconds = elapsed

                # Update peak rate
                if rate > self.step_stats[step_name].peak_rate:
                    self.step_stats[step_name].peak_rate = rate

                # Calculate ETA
                task = self.step_progress.tasks[0] if self.step_progress.tasks else None
                if task and task.total and rate > 0:
                    remaining = task.total - processed_count
                    self.step_stats[step_name].eta_seconds = remaining / rate

        # Update progress display
        rate_display = f"{self.step_stats[step_name].items_per_second:.1f}/s"

        self.step_progress.update(
            self.current_task, completed=processed_count, rate=rate_display
        )

        # Add extra info to step data if provided
        if extra_info and self.pipeline_state:
            step_progress = self.pipeline_state.step_progress.get(step_name)
            if step_progress:
                for key, value in extra_info.items():
                    step_progress.set_data(key, value)

    def complete_step_progress(self, step_name: str) -> None:
        """Mark step progress as completed.

        Args:
            step_name: Name of the completed step
        """
        if step_name in self.step_start_times:
            elapsed = time.time() - self.step_start_times[step_name]
            self.step_stats[step_name].elapsed_seconds = elapsed

        if self.current_task and self.step_progress:
            # Complete the current task
            task = None
            for t in self.step_progress.tasks:
                if t.id == self.current_task:
                    task = t
                    break

            if task:
                self.step_progress.update(
                    self.current_task, completed=task.total, rate="Complete"
                )

        self.logger.debug(f"Completed step progress: {step_name}")

    def add_statistics_panel(self, stats: Dict[str, Any]) -> None:
        """Add or update the statistics panel.

        Args:
            stats: Statistics to display
        """
        self.stats_table = Table(show_header=True, header_style="bold magenta")
        self.stats_table.add_column("Metric", style="cyan", no_wrap=True)
        self.stats_table.add_column("Value", style="green")

        # Add pipeline statistics
        if self.pipeline_state:
            self.stats_table.add_row(
                "Video File", str(self.pipeline_state.video_file.split("/")[-1])
            )
            self.stats_table.add_row(
                "Current Step",
                self.pipeline_state.current_step.replace("_", " ").title(),
            )
            self.stats_table.add_row(
                "Completed Steps",
                f"{len(self.pipeline_state.completed_steps)}/{get_total_pipeline_steps()}",
            )

            # Add step-specific stats
            current_step_progress = self.pipeline_state.step_progress.get(
                self.pipeline_state.current_step
            )
            if current_step_progress:
                self.stats_table.add_row(
                    "Step Progress", f"{current_step_progress.progress_percentage:.1f}%"
                )

                if self.pipeline_state.current_step in self.step_stats:
                    step_stat = self.step_stats[self.pipeline_state.current_step]
                    self.stats_table.add_row(
                        "Processing Rate", f"{step_stat.items_per_second:.1f} items/s"
                    )
                    self.stats_table.add_row(
                        "Peak Rate", f"{step_stat.peak_rate:.1f} items/s"
                    )
                    self.stats_table.add_row("ETA", step_stat.eta_formatted)

        # Add custom stats
        for key, value in stats.items():
            self.stats_table.add_row(key, str(value))

    def display_final_summary(self, result: ProcessingResult) -> None:
        """Display final processing summary.

        Args:
            result: Final processing result
        """
        if not self.console:
            return

        # Create summary table
        summary_table = Table(
            title="Processing Complete", show_header=True, header_style="bold blue"
        )
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="green")

        # Add results
        summary_table.add_row("Success", "✅ Yes" if result.success else "❌ No")
        summary_table.add_row("Total Frames", str(result.total_frames_extracted))
        summary_table.add_row("Faces Found", str(result.faces_found))
        summary_table.add_row(
            "Processing Time", f"{result.processing_time_seconds:.1f}s"
        )

        # Add pose counts
        if result.poses_found:
            for pose, count in result.poses_found.items():
                summary_table.add_row(f"Poses ({pose})", str(count))

        # Add head angle counts
        if result.head_angles_found:
            for angle, count in result.head_angles_found.items():
                summary_table.add_row(f"Head Angles ({angle})", str(count))

        # Add output files
        if result.output_files:
            summary_table.add_row("Output Files", str(len(result.output_files)))

        # Display summary
        self.console.print()
        self.console.print(
            Panel(summary_table, border_style="green" if result.success else "red")
        )

        if result.output_files:
            self.console.print("\n[bold green]Generated Files:[/bold green]")
            for file_path in result.output_files[:10]:  # Show first 10 files
                self.console.print(f"  📁 {file_path}")
            if len(result.output_files) > 10:
                self.console.print(
                    f"  ... and {len(result.output_files) - 10} more files"
                )

    def stop_progress(self) -> None:
        """Stop the progress display."""
        if self.live_display:
            self.live_display.stop()

        self.is_active = False
        self.current_task = None
        self.main_progress = None
        self.step_progress = None
        self.live_display = None

        self.logger.info("Stopped pipeline progress display")

    def _update_step_progress(
        self, current_step: str, step_progress_dict: Dict[str, StepProgress]
    ) -> None:
        """Update step progress from pipeline state.

        Args:
            current_step: Name of current step
            step_progress_dict: Dictionary of step progress objects
        """
        if current_step not in step_progress_dict or not self.step_progress:
            return

        step_progress = step_progress_dict[current_step]

        # Auto-start step progress if not already started
        if not self.current_task and step_progress.total_items > 0:
            self.start_step_progress(
                current_step,
                step_progress.total_items,
                current_step.replace("_", " ").title(),
            )

        # Update progress
        if self.current_task and step_progress.total_items > 0:
            self.update_step_progress(current_step, step_progress.processed_count)

    def _update_statistics_panel(self) -> None:
        """Update the statistics panel with current data."""
        stats = {}

        # Add overall timing
        if self.overall_start_time:
            elapsed = time.time() - self.overall_start_time
            stats["Total Elapsed"] = f"{elapsed:.1f}s"

        # Add step-specific data
        if (
            self.pipeline_state
            and self.pipeline_state.current_step in self.pipeline_state.step_progress
        ):
            current_progress = self.pipeline_state.step_progress[
                self.pipeline_state.current_step
            ]

            # Add counts from step data
            if "extracted_count" in current_progress.step_data:
                stats["Frames Extracted"] = str(
                    current_progress.get_data("extracted_count", 0)
                )

            if "faces_found" in current_progress.step_data:
                stats["Faces Found"] = str(current_progress.get_data("faces_found", 0))

            if "poses_found" in current_progress.step_data:
                poses = current_progress.get_data("poses_found", {})
                if isinstance(poses, dict):
                    total_poses = sum(poses.values()) if poses else 0
                    stats["Poses Detected"] = str(total_poses)

        self.add_statistics_panel(stats)

    def __enter__(self):
        """Context manager entry."""
        if self.live_display and not self.live_display.is_started:
            self.live_display.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_progress()


def create_progress_manager(console: Optional[Console] = None) -> ProgressManager:
    """Create a progress manager instance.

    Args:
        console: Optional Rich console instance

    Returns:
        ProgressManager instance
    """
    return ProgressManager(console)
