"""
Logging setup module.

Configures a root logger that logs to both console and file (`logs/monai_segmenter.log`)
with a consistent format. Ensures logging is only initialized once per session.
"""

# Python Standard Library
import logging
from pathlib import Path

# Define logfile path and ensure parent directory exists
LOGFILE_PATH = Path("logs/monai_segmenter.log")
LOGFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

_logger_initialized = False


def setup_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Set up and return a logger instance with standardized formatting.

    Args:
        name (str): Name of the logger (usually the module name).
        level (str | None): Logging level as string (e.g., "INFO", "DEBUG"). Defaults to INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _logger_initialized

    resolved_level = getattr(logging, level.upper(), logging.INFO) if level else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)

    if not _logger_initialized:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        file_handler = logging.FileHandler(LOGFILE_PATH, mode="w")
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Attach handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(resolved_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        _logger_initialized = True

    return logger
