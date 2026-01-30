"""Workaround for missing config.py in pocket-tts-mlx.

Imports from pocket_tts.
"""

from pocket_tts.utils.config import (
    Config,
    FlowLMConfig,
    FlowLMTransformerConfig,
    load_config,
)

__all__ = ["Config", "FlowLMConfig", "FlowLMTransformerConfig", "load_config"]
