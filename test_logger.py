from src.logging.setup_logger import setup_logger

logger = setup_logger("LoggerTest")

logger.debug("This is a DEBUG message")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")
logger.error("This is an ERROR message")
logger.critical("This is a CRITICAL message")

try:
    1 / 0
except ZeroDivisionError:
    logger.exception("Exception occurred")
