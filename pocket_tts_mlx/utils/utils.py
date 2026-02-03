"""Utility functions for pocket-tts-mlx."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def size_of_dict(state_dict: dict) -> int:
    """Estimate total byte size of nested MLX arrays in a state dict."""
    total_size = 0
    for value in state_dict.values():
        import mlx.core as mx

        if isinstance(value, mx.array):
            total_size += value.size * value.dtype.size
        elif isinstance(value, dict):
            total_size += size_of_dict(value)
    return total_size


class display_execution_time:
    """Context manager that logs elapsed time for a named task."""
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
        return False


def make_cache_directory() -> Path:
    """Create and return the cache directory for downloaded assets."""
    cache_dir = Path.home() / ".cache" / "pocket_tts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_if_necessary(file_path: str) -> Path:
    """Resolve a local path or download remote assets into cache."""
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Cache by URL hash to avoid repeated downloads.
        cache_dir = make_cache_directory()
        cached_file = cache_dir / (
            hashlib.sha256(file_path.encode()).hexdigest() + "." + file_path.split(".")[-1]
        )
        if not cached_file.exists():
            response = requests.get(file_path)
            response.raise_for_status()
            with open(cached_file, "wb") as f:
                f.write(response.content)
        return cached_file
    if file_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        # Parse hf://repo_id/path@revision into hub download arguments.
        file_path = file_path.removeprefix("hf://")
        parts = file_path.split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        if "@" in filename:
            filename, revision = filename.split("@")
        else:
            revision = None
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(cached_file)
    return Path(file_path)
