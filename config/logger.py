import logging
import sys

from config import settings


def configure_logging():
    """Configures application logging using standard logging module."""

    # Create a custom logger
    logger = logging.getLogger(settings.config.APP_NAME)
    # Set level based on config if we had a debug flag, otherwise default to INFO
    # For now, let's keep it INFO as base, or DEBUG if we want verbose console
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times if function called repeatedly
    if logger.hasHandlers():
        return logger

    # Format: Time | Level | Module | Message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Logging initialized (Console only).")
    return logger


# Initialize logger globally for import
logger = configure_logging()
