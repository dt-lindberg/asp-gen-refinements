"""Sets up a logger that writes to stdout and outputs/<name>.log."""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    os.makedirs("outputs", exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(f"outputs/{name}.log")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(lambda r: r.name == "inference")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = (
        False  # don't pass to root logger (already claimed by unsloth/transformers)
    )
    return logger
