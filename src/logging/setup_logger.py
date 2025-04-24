# Python Standard Library
import logging
import os
from pathlib import Path


def setup_logger(name: str, log_file: str | None = None, level: int | None = None) -> logging.Logger:
    # Determine logging level from environment variable or default to INFO
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    resolved_level = level if level is not None else getattr(logging, level_name, logging.INFO)

    # Generate log file path per module if not provided
    if log_file is None:
        log_file = f"logs/{name.lower()}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler writes logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler outputs logs to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)

    # Add handlers only if logger is not already configured
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.propagate = False

    return logger
