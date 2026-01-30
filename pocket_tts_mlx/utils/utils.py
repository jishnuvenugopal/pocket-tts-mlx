"""Workaround for missing utils.py in pocket-tts-mlx.

Imports functions from pocket_tts and provides stubs for missing functions.
"""

import time
import logging
from contextlib import contextmanager

from pocket_tts.utils.utils import download_if_necessary, size_of_dict

logger = logging.getLogger(__name__)


class TimerResult:
    """Timer result object with elapsed_time_ms attribute."""
    def __init__(self):
        self.elapsed_time_ms = 0.0
        self.elapsed = 0.0


@contextmanager
def display_execution_time(name: str, print_output: bool = True):
    """Context manager to measure and optionally display execution time."""
    start = time.perf_counter()
    timer = TimerResult()
    try:
        yield timer
    finally:
        elapsed = time.perf_counter() - start
        timer.elapsed = elapsed
        timer.elapsed_time_ms = elapsed * 1000
        if print_output:
            logger.debug(f"{name}: {timer.elapsed_time_ms:.1f}ms")


__all__ = ["download_if_necessary", "size_of_dict", "display_execution_time"]
