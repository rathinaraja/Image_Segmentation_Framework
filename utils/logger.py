"""
utils/logger.py
---------------
CSV + console logger. Writes one row per epoch to a CSV for easy plotting.
"""

import csv
import logging
import os
import sys
from datetime import datetime

def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger  # avoid duplicate handlers

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class CSVLogger:
    """Appends metric dicts as rows to a CSV file."""

    def __init__(self, path: str):
        self.path = path
        self._fieldnames = None

    def log(self, row: dict):
        write_header = not os.path.exists(self.path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
