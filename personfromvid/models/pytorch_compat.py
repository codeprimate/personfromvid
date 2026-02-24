"""Patch torch.load to weights_only=False so YOLO/other checkpoints load under PyTorch 2.6+."""

import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@contextmanager
def torch_load_allow_pickle() -> Iterator[None]:
    """Temporarily patch torch.load to use weights_only=False when not specified.

    Use when loading trusted third-party checkpoints (e.g. ultralytics YOLO)
    that are loaded via code we do not control. Restores the original
    torch.load when the context exits.
    """
    import torch

    original_load = torch.load

    def patched_load(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    try:
        torch.load = patched_load  # type: ignore[assignment]
        yield
    finally:
        torch.load = original_load  # type: ignore[assignment]
