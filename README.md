# pocket-tts-mlx

MLX backend for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) optimized for Apple Silicon.

Runtime is torch-free. Torch is only required for optional parity tests.

**Installation**

```bash
pip install pocket-tts-mlx
```

Local development:

```bash
pip install -e .
```

Model weights are downloaded from Hugging Face on first run. For voice cloning
weights, accept the model terms and authenticate:

```bash
hf auth login
```

**Quickstart**

```python
from pocket_tts_mlx import TTSModel

model = TTSModel.load_model()
state = model.get_state_for_audio_prompt("marius")
audio = model.generate_audio(state, "Hello from MLX!", max_tokens=200)
```

**CLI**

```bash
pocket-tts-mlx "Hello, world!" --voice marius --output output.wav
```

**Voices**

Predefined voices:

- alba
- marius
- javert
- jean
- fantine
- cosette
- eponine
- azelma

**Requirements**

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX
- Internet access for initial model downloads

**Notes**

- Voice cloning requires Hugging Face access to `kyutai/pocket-tts`.
- Non-voice-cloning weights are used automatically when voice cloning is unavailable.
