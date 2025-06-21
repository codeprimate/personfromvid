from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from personfromvid.utils.exceptions import InterruptionError

if TYPE_CHECKING:
    from ..pipeline import ProcessingPipeline
    from ..state_manager import PipelineState


class PipelineStep(ABC):
    """Abstract base class for a pipeline processing step."""

    def __init__(self, pipeline: "ProcessingPipeline"):
        """Initialize the pipeline step.

        Args:
            pipeline: The main ProcessingPipeline instance.
        """
        self.pipeline = pipeline
        self.config = pipeline.config
        self.state = pipeline.state
        self.logger = pipeline.logger
        self.formatter = pipeline.formatter
        
        # Note: state may be None during initialization and will be set later

    @property
    def _state(self) -> "PipelineState":
        """Get state with type assertion - for internal use only."""
        assert self.state is not None, "Pipeline state must be initialized"
        return self.state

    @property
    @abstractmethod
    def step_name(self) -> str:
        """The name of the pipeline step."""
        pass

    def _check_interrupted(self) -> None:
        """Check if processing has been interrupted."""
        if self.pipeline._interrupted:
            raise InterruptionError(f"Processing interrupted during {self.step_name}")

    def _get_step_start_time(self) -> float:
        """Get the start time of the current step execution."""
        return self.pipeline._step_start_time or 0.0

    @abstractmethod
    def execute(self) -> None:
        """Execute the logic for this pipeline step."""
        pass
