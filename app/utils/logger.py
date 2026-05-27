import logging
import os
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    # one logger setup for the whole app, reuse it everywhere
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already set up, don't add duplicate handlers

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # logs to console
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # logs to file
    file_handler = logging.FileHandler(LOG_DIR / "churn_api.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
