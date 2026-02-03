"""Audio I/O utilities with streaming WAV output (NumPy-based)."""

import logging
import os
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

FIRST_CHUNK_LENGTH_SECONDS = float(os.environ.get("FIRST_CHUNK_LENGTH_SECONDS", "0"))


def audio_read(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Read audio file. WAV uses built-in wave; others require soundfile."""
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".wav":
        with wave.open(str(filepath), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            raw_data = wav_file.readframes(-1)
            # Normalize int16 PCM to float32 in [-1, 1].
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return samples[None, :], sample_rate

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV audio files. Install with: pip install soundfile"
        ) from e

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        wav = data[None, :]
    else:
        wav = data.mean(axis=1)[None, :]
    return wav, sample_rate


class StreamingWAVWriter:
    """Streaming WAV writer with optional initial buffering."""

    def __init__(self, output_stream, sample_rate: int):
        self.output_stream = output_stream
        self.sample_rate = sample_rate
        self.wave_writer = None
        self.first_chunk_buffer = []

    def write_header(self, sample_rate: int):
        """Write a placeholder WAV header for streaming output."""
        self.wave_writer = wave.open(self.output_stream, "wb")
        self.wave_writer.setnchannels(1)
        self.wave_writer.setsampwidth(2)
        self.wave_writer.setframerate(sample_rate)
        self.wave_writer.setnframes(1_000_000_000)

    def write_pcm_data(self, audio_chunk: Any):
        """Write a chunk of float audio as int16 PCM."""
        chunk = np.asarray(audio_chunk)
        if chunk.ndim > 1:
            chunk = chunk.reshape(-1)
        chunk_int16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
        chunk_bytes = chunk_int16.tobytes()

        if self.first_chunk_buffer is not None:
            # Buffer first chunk until minimum duration to avoid truncated headers.
            self.first_chunk_buffer.append(chunk_bytes)
            total_length = sum(len(c) for c in self.first_chunk_buffer)
            target_length = int(self.sample_rate * FIRST_CHUNK_LENGTH_SECONDS) * 2
            if total_length < target_length:
                return
            self._flush()
            return

        self.wave_writer.writeframesraw(chunk_bytes)

    def _flush(self):
        """Flush buffered data into the WAV stream."""
        if self.first_chunk_buffer is not None:
            self.wave_writer.writeframesraw(b"".join(self.first_chunk_buffer))
            self.first_chunk_buffer = None

    def finalize(self):
        """Finalize WAV stream with short silence and close file."""
        self._flush()
        silence_duration_sec = 0.2
        num_silence_samples = int(self.sample_rate * silence_duration_sec)
        self.wave_writer.writeframesraw(bytes(num_silence_samples * 2))
        if self.wave_writer:
            self.wave_writer._patchheader = lambda: None
            self.wave_writer.close()


def is_file_like(obj):
    """Return True for file-like objects that support write and close."""
    return all(hasattr(obj, attr) for attr in ["write", "close"])


def stream_audio_chunks(
    path: str | Path | None | Any, audio_chunks: Iterator[Any], sample_rate: int
):
    """Write streaming audio chunks to a file path, handle, or stdout."""
    if path == "-":
        f = sys.stdout.buffer
    elif path is None:
        f = nullcontext()
    elif is_file_like(path):
        f = path
    else:
        f = open(path, "wb")

    with f:
        if path is not None:
            writer = StreamingWAVWriter(f, sample_rate)
            writer.write_header(sample_rate)

        for chunk in audio_chunks:
            if path is not None:
                writer.write_pcm_data(chunk)

        if path is not None:
            writer.finalize()
