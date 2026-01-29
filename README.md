# pocket-tts-mlx

[MLX](https://ml-explore.github.io/mlx/) backend for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) with Apple Silicon optimization.

## Performance

On Apple Silicon (M1/M2/M3/M4), this MLX backend achieves significantly better performance than the PyTorch CPU implementation:

| Implementation | RTF | Speed vs Real-Time |
|----------------|-----|-------------------|
| PyTorch CPU | ~0.25 | 4x faster |
| MLX (M1 Pro) | ~0.079 | 12.7x faster |
| MLX (6-bit quantized) | ~0.1 | 10x faster |

*RTF = Real-Time Factor (lower is better, <1.0 means faster than real-time)*

## Installation

```bash
pip install pocket-tts-mlx
```

## Usage

```python
from pocket_tts_mlx import TTSModel

# Load the MLX model
model = TTSModel.load_model()

# Generate audio
audio = model.generate_audio(
    voice_state=model.get_state_for_audio_prompt("marius"),
    text="Hello, this is a test of the MLX backend!",
    max_tokens=500,
    frames_after_eos=7
)

# Save to file
import soundfile as sf
sf.write("output.wav", audio.numpy(), 24000)
```

## CLI Usage

```bash
# Using MLX backend
python -m pocket_tts_mlx.main "Hello, world!" --voice marius --output output.wav
```

## Requirements

- Python 3.9+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX
- pocket-tts (for weights and configuration)

## Development

This is a community-maintained MLX implementation of pocket-tts. For the official PyTorch version, see [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts).

## License

This project maintains the same license as pocket-tts.
