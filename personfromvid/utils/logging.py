"""Logging configuration and utilities for Person From Vid.

This module provides structured logging with Rich console integration,
file logging support, and configurable verbosity levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from ..data.config import LoggingConfig, LogLevel
from .exceptions import PersonFromVidError, format_exception_message
from .formatting import RichFormatter, create_formatter


class PersonFromVidFormatter(logging.Formatter):
    """Custom formatter for Person From Vid log messages."""

    def __init__(self, include_timestamp: bool = True, include_module: bool = True):
        self.include_timestamp = include_timestamp
        self.include_module = include_module
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with custom styling."""
        parts = []

        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            parts.append(f"[{timestamp}]")

        # Add level with color coding
        level_colors = {
            "DEBUG": "dim",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red",
        }
        level_color = level_colors.get(record.levelname, "white")
        parts.append(f"[{level_color}]{record.levelname}[/{level_color}]")

        if self.include_module and record.name != "personfromvid":
            module_name = record.name.replace("personfromvid.", "")
            parts.append(f"[dim]{module_name}[/dim]")

        # Format message
        message = record.getMessage()

        # Handle exceptions specially
        if record.exc_info:
            exception = record.exc_info[1]
            if isinstance(exception, PersonFromVidError):
                message = format_exception_message(exception, include_traceback=False)
            else:
                message += f"\n{self.formatException(record.exc_info)}"

        parts.append(message)

        return " ".join(parts)


class ProgressAwareHandler(RichHandler):
    """Rich handler that works well with progress displays."""

    def __init__(self, console: Optional[Console] = None, **kwargs):
        if console is None:
            console = Console(stderr=True)
        super().__init__(console=console, **kwargs)
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record with progress awareness."""
        try:
            # Temporarily stop any active progress bars
            if hasattr(self.console, "_live") and self.console._live is not None:
                with self.console._live:
                    super().emit(record)
            else:
                super().emit(record)
        except Exception:
            self.handleError(record)


class PersonFromVidLogger:
    """Main logger class for Person From Vid."""

    def __init__(self, config: LoggingConfig):
        self.config = config
        self.console = Console(stderr=True) if config.enable_rich_console else None
        self._loggers: Dict[str, logging.Logger] = {}

        # Initialize structured formatter if enabled
        self.formatter: Optional[RichFormatter] = None
        if config.enable_structured_output and config.enable_rich_console:
            self.formatter = create_formatter(self.console)

        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Install rich traceback handling if enabled
        if self.config.enable_rich_console:
            install_rich_traceback(
                console=self.console, show_locals=self.config.verbose
            )

        # Configure root logger
        root_logger = logging.getLogger("personfromvid")
        root_logger.setLevel(getattr(logging, self.config.level.value))

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Add console handler
        if self.config.enable_rich_console and self.console:
            console_handler = ProgressAwareHandler(
                console=self.console,
                show_time=False,  # We handle timestamps in our formatter
                show_path=self.config.verbose,
                rich_tracebacks=True,
                tracebacks_show_locals=self.config.verbose,
            )
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(
                PersonFromVidFormatter(
                    include_timestamp=True, include_module=self.config.verbose
                )
            )

        console_handler.setLevel(getattr(logging, self.config.level.value))
        root_logger.addHandler(console_handler)

        # Add file handler if configured
        if self.config.enable_file_logging and self.config.log_file:
            self._setup_file_logging(root_logger)

        # Prevent propagation to avoid duplicate messages
        root_logger.propagate = False

    def _setup_file_logging(self, logger: logging.Logger) -> None:
        """Set up file logging."""
        if not self.config.log_file:
            return

        # Create log directory if it doesn't exist
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(self.config.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File logs everything

        # Use detailed formatter for file logs
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger for a specific module."""
        if name not in self._loggers:
            logger_name = (
                f"personfromvid.{name}"
                if not name.startswith("personfromvid")
                else name
            )
            logger = logging.getLogger(logger_name)
            self._loggers[name] = logger

        return self._loggers[name]

    def log_exception(self, exception: Exception, logger_name: str = "main") -> None:
        """Log an exception with appropriate formatting."""
        logger = self.get_logger(logger_name)

        if isinstance(exception, PersonFromVidError):
            # Use error level for our custom exceptions
            logger.error(
                format_exception_message(
                    exception, include_traceback=self.config.verbose
                )
            )
        else:
            # Use exception method for system exceptions
            logger.exception("Unexpected error occurred")

    def log_progress_update(self, message: str, logger_name: str = "progress") -> None:
        """Log progress updates in a way that works with Rich displays."""
        if self.config.verbose:
            logger = self.get_logger(logger_name)
            logger.debug(message)

    def set_level(self, level: LogLevel) -> None:
        """Change logging level at runtime."""
        self.config.level = level
        logging_level = getattr(logging, level.value)

        # Update all loggers
        root_logger = logging.getLogger("personfromvid")
        root_logger.setLevel(logging_level)

        for handler in root_logger.handlers:
            if isinstance(handler, (logging.StreamHandler, ProgressAwareHandler)):
                handler.setLevel(logging_level)

    def get_formatter(self) -> Optional[RichFormatter]:
        """Get the structured formatter if available."""
        return self.formatter


# Global logger instance
_logger_instance: Optional[PersonFromVidLogger] = None


def setup_logging(config: LoggingConfig) -> PersonFromVidLogger:
    """Set up global logging configuration."""
    global _logger_instance
    _logger_instance = PersonFromVidLogger(config)
    return _logger_instance


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    if _logger_instance is None:
        # Use default configuration if not set up
        from ..data.config import LoggingConfig

        setup_logging(LoggingConfig())

    return _logger_instance.get_logger(name)


def log_exception(exception: Exception, logger_name: str = "main") -> None:
    """Log an exception using the global logger."""
    if _logger_instance:
        _logger_instance.log_exception(exception, logger_name)
    else:
        # Fallback to basic logging
        logging.getLogger("personfromvid").exception("Error occurred")


def log_progress(message: str, logger_name: str = "progress") -> None:
    """Log progress message using the global logger."""
    if _logger_instance:
        _logger_instance.log_progress_update(message, logger_name)


def set_log_level(level: LogLevel) -> None:
    """Set logging level globally."""
    if _logger_instance:
        _logger_instance.set_level(level)


# Convenience functions for common logging operations
def debug(message: str, logger_name: str = "main") -> None:
    """Log debug message."""
    get_logger(logger_name).debug(message)


def info(message: str, logger_name: str = "main") -> None:
    """Log info message."""
    get_logger(logger_name).info(message)


def warning(message: str, logger_name: str = "main") -> None:
    """Log warning message."""
    get_logger(logger_name).warning(message)


def error(message: str, logger_name: str = "main") -> None:
    """Log error message."""
    get_logger(logger_name).error(message)


def critical(message: str, logger_name: str = "main") -> None:
    """Log critical message."""
    get_logger(logger_name).critical(message)


def get_formatter() -> Optional[RichFormatter]:
    """Get the global structured formatter if available."""
    global _logger_instance
    if _logger_instance:
        return _logger_instance.get_formatter()
    return None
