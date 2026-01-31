"""Utility functions for pocket-tts-mlx.

Adapted from pocket-tts, removing PyTorch dependencies for MLX compatibility.
"""

import hashlib
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def size_of_dict(state_dict: dict) -> int:
    """Calculate the size of a state dict in bytes.

    Args:
        state_dict: Dictionary containing MLX arrays or nested dicts.

    Returns:
        Total size in bytes.
    """
    total_size = 0
    for value in state_dict.values():
        import mlx.core as mx

        if isinstance(value, mx.array):
            # MLX arrays: size = number of elements * bytes per element
            total_size += value.size * value.dtype.size
        elif isinstance(value, dict):
            total_size += size_of_dict(value)
    return total_size


class display_execution_time:
    """Context manager for timing and logging execution time.

    Args:
        task_name: Name of the task being timed.
        print_output: Whether to print the timing output. Defaults to True.
    """

    def __init__(self, task_name: str, print_output: bool = True):
        self.task_name = task_name
        self.print_output = print_output
        self.start_time = None
        self.elapsed_time_ms = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.monotonic()
        self.elapsed_time_ms = int((end_time - self.start_time) * 1000)
        if self.print_output:
            self.logger.info("%s took %d ms", self.task_name, self.elapsed_time_ms)
        return False  # Don't suppress exceptions


def make_cache_directory() -> Path:
    """Create and return the cache directory for pocket-tts-mlx.

    Returns:
        Path to the cache directory.
    """
    cache_dir = Path.home() / ".cache" / "pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_if_necessary(file_path: str) -> Path:
    """Download a file if not already cached, or return the cached path.

    Supports HTTP, HTTPS, and HuggingFace hub URLs.

    Args:
        file_path: URL or local file path. Can be:
            - HTTP/HTTPS URL
            - HuggingFace hub URL (hf://repo_id/filename@revision)
            - Local file path

    Returns:
        Path to the file (cached or local).
    """
    if file_path.startswith("http://") or file_path.startswith("https://"):
        cache_dir = make_cache_directory()
        # Create a unique cached filename based on the URL hash
        cached_file = cache_dir / (
            hashlib.sha256(file_path.encode()).hexdigest() + "." + file_path.split(".")[-1]
        )
        if not cached_file.exists():
            response = requests.get(file_path)
            response.raise_for_status()
            with open(cached_file, "wb") as f:
                f.write(response.content)
        return cached_file
    elif file_path.startswith("hf://"):
        # Parse HuggingFace URL: hf://repo_id/filename[@revision]
        from huggingface_hub import hf_hub_download

        file_path = file_path.removeprefix("hf://")
        splitted = file_path.split("/")
        repo_id = "/".join(splitted[:2])
        filename = "/".join(splitted[2:])
        if "@" in filename:
            filename, revision = filename.split("@")
        else:
            revision = None
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(cached_file)
    else:
        return Path(file_path)
