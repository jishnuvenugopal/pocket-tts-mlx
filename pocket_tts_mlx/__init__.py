"""pocket-tts-mlx: MLX backend for pocket-tts with Apple Silicon optimization.

This package provides an MLX implementation of pocket-tts that achieves
significantly better performance on Apple Silicon (M1/M2/M3/M4) compared
to the PyTorch CPU version.

Example usage:
    from pocket_tts_mlx import TTSModel

    model = TTSModel.load_model()
    audio = model.generate_audio(
        model.get_state_for_audio_prompt("marius"),
        "Hello, world!",
        max_tokens=500
    )
"""

__version__ = "0.1.0"

from pocket_tts_mlx.models.tts_model import TTSModel

__all__ = ["TTSModel"]
