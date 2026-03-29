"""
Observability Logger Module - Structured Logging

Provides structured logging with support for JSON output format.
Used throughout the system for tracing and debugging.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # If logger already has handlers, return it
    if logger.handlers:
        return logger

    # Configure handler to stderr (don't pollute stdout for MCP)
    handler = logging.StreamHandler(sys.stderr)

    # Set default format
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def setup_file_logging(
    log_file: str,
    level: int = logging.INFO,
    format_json: bool = False,
) -> logging.Logger:
    """
    Setup file logging for persistence.

    Args:
        log_file: Path to log file
        level: Logging level
        format_json: Whether to use JSON format

    Returns:
        Configured logger
    """
    logger = logging.getLogger("modular_rag")

    # Create log directory if needed
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Set formatter
    if format_json:
        # TODO: Implement JSONFormatter in Phase F2
        formatter = logging.Formatter(
            fmt='%(message)s',
        )
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_trace_logger() -> logging.Logger:
    """
    Get a logger configured for trace output (JSON Lines format).

    Returns:
        Logger configured for trace output
    """
    logger = logging.getLogger("modular_rag.trace")
    return logger


def write_trace(trace_dict: dict) -> None:
    """
    Write a trace entry to the trace log.

    Args:
        trace_dict: Dictionary containing trace information

    Note:
        This is a placeholder. Full implementation in Phase F2.
    """
    # TODO: Implement JSON Lines writing in Phase F2
    pass
