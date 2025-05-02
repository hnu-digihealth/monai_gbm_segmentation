import logging
from pathlib import Path

LOGFILE_PATH = Path("logs/monai_segmenter.log")
LOGFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

_logger_initialized = False


def setup_logger(name: str, level: str | None = None) -> logging.Logger:
    global _logger_initialized

    resolved_level = getattr(logging, level.upper(), logging.INFO) if level else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)

    if not _logger_initialized:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(LOGFILE_PATH, mode="w")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(resolved_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        _logger_initialized = True

    return logger
