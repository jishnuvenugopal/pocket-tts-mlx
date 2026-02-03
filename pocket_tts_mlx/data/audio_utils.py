"""Audio conversion utilities (numpy-based)."""

import math
from typing import Any

import numpy as np
from scipy.signal import resample_poly


def _as_2d(wav: np.ndarray) -> np.ndarray:
    """Ensure audio has shape [C, T]."""
    return wav[None, :] if wav.ndim == 1 else wav


def convert_audio(
    wav: Any, from_rate: int | float, to_rate: int | float, to_channels: int
) -> np.ndarray:
    """Convert sample rate and channels using polyphase resampling."""
    wav_np = np.asarray(wav)
    wav_np = _as_2d(wav_np)

    if wav_np.shape[0] != to_channels:
        # Channel conversion via averaging or tiling.
        if to_channels == 1:
            wav_np = wav_np.mean(axis=0, keepdims=True)
        elif wav_np.shape[0] == 1:
            wav_np = np.tile(wav_np, (to_channels, 1))
        else:
            raise ValueError(
                f"Cannot convert from {wav_np.shape[0]} channels to {to_channels} channels"
            )

    from_rate_i = int(round(from_rate))
    to_rate_i = int(round(to_rate))
    if from_rate_i != to_rate_i:
        gcd = math.gcd(from_rate_i, to_rate_i)
        up = to_rate_i // gcd
        down = from_rate_i // gcd
        wav_np = resample_poly(wav_np, up, down, axis=-1)

    return wav_np.astype(np.float32)
