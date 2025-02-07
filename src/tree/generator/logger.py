import logging
import os

from multiprocess import current_process  # noqa

LOG_DIR = "resources/log/"


def setup_logger():
    """
    Setups a logger for the current process.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    process_name = current_process().name
    log_filename = f"{LOG_DIR}/process_{process_name}.log"

    logger = logging.getLogger(process_name)
    logger.setLevel(logging.DEBUG)

    # File Handler
    handler = logging.FileHandler(log_filename, mode='a')
    handler.setLevel(logging.DEBUG)

    # Simple formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
