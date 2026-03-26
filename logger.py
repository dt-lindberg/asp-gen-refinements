"""Centralized low-code logger"""

import logging
import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

from config import LOG_FORMAT, ALLOWED_LOGGERS


def _set_log_level(log_level):
    """Map string to logging level"""
    log_level = log_level.strip().lower()
    if log_level == "debug":
        return logging.DEBUG
    elif log_level == "info":
        return logging.INFO
    elif log_level == "warning":
        return logging.WARNING
    elif log_level == "error":
        return logging.ERROR
    elif log_level == "critical":
        return logging.CRITICAL
    else:
        return logging.DEBUG


def setup_logging(log_level="info", force=False):
    """Configure root logging with stdout + logging file (via LOG_DIR .env var)"""
    # Ensure logging dir is set and create file
    log_dir = os.getenv("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR env var must be set")

    # Set log directory to log_dir/day/
    log_dir = os.path.join(log_dir, f"{time.strftime('%Y%m%d')}")

    log_file_path = os.path.join(log_dir, f"log_{time.strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    log_level = _set_log_level(log_level)
    print(f"log level {log_level}")

    # Handlers for both stdout and file
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_path)]

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=handlers,
        force=force,
    )

    # Block all loggers by default, then allow only specific ones
    logging.getLogger().setLevel(logging.WARNING)
    for name in ALLOWED_LOGGERS:
        logging.getLogger(name).setLevel(log_level)

    # Route warnings to logger as well
    logging.captureWarnings(True)


def get_logger(name):
    """Get a logger with the given name"""
    return logging.getLogger(name)
