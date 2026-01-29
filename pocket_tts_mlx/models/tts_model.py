"""TTS Model for MLX.

This module implements the complete TTS model combining FlowLM and Mimi
for text-to-speech generation using MLX.

The implementation mirrors the PyTorch version but uses MLX operations.
"""

import copy
import logging
import queue
import statistics
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import safetensors
import torch

from pocket_tts_mlx.conditioners.base import TokenizedText
from pocket_tts_mlx.conditioners.text import LUTConditioner
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts_mlx.models.flow_lm import FlowLMModel, lsd_decode
from pocket_tts_mlx.models.mimi import MimiModel
from pocket_tts_mlx.modules.dummy_quantizer import DummyQuantizer
from pocket_tts_mlx.modules.mimi_transformer import ProjectedTransformer
from pocket_tts_mlx.modules.mlp import SimpleMLPAdaLN
from pocket_tts_mlx.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts_mlx.modules.stateful_module import StatefulModule, increment_steps, init_states
from pocket_tts_mlx.utils.config import Config, load_config
from pocket_tts_mlx.utils.weight_conversion import (
    get_flow_lm_state_dict_mlx,
    get_mimi_state_dict_mlx,
    load_predefined_voice_mlx,
    load_weights_to_mlx_model,
    PREDEFINED_VOICES,
)
from pocket_tts_mlx.utils.utils import (
    display_execution_time,
    download_if_necessary,
    size_of_dict,
)

logger = logging.getLogger(__name__)


def split_into_best_sentences(tokenizer, text_to_generate: str, max_tokens: int) -> list[str]:
    """Split text into sentence chunks that fit within token limit.

    Args:
        tokenizer: Tokenizer instance.
        text_to_generate: Text to split.
        max_tokens: Maximum tokens per chunk.

    Returns:
        List of text chunks.
    """
    text_to_generate, _ = prepare_text_prompt(text_to_generate)
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens.tokens[0].tolist()

    _, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0].tolist()

    end_of_sentences_indices = [0]
    previous_was_end_of_sentence_token = False

    for token_idx, token in enumerate(list_of_tokens):
        if token in end_of_sentence_tokens:
            previous_was_end_of_sentence_token = True
        else:
            if previous_was_end_of_sentence_token:
                end_of_sentences_indices.append(token_idx)
            previous_was_end_of_sentence_token = False
    end_of_sentences_indices.append(len(list_of_tokens))

    nb_tokens_and_sentences = []
    for i in range(len(end_of_sentences_indices) - 1):
        start = end_of_sentences_indices[i]
        end = end_of_sentences_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        nb_tokens_and_sentences.append((end - start, text))

    max_nb_tokens_in_a_chunk = max_tokens
    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0

    for nb_tokens, sentence in nb_tokens_and_sentences:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_nb_tokens_in_a_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    return chunks


def prepare_text_prompt(text: str) -> tuple[str, int]:
    """Prepare text prompt for TTS generation.

    Args:
        text: Input text.

    Returns:
        Tuple of (prepared_text, frames_after_eos_guess).
    """
    text = text.strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    number_of_words = len(text.split())

    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1

    # Capitalize first letter
    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    # Add ending punctuation if needed
    if text[-1].isalnum():
        text = text + "."

    # Add padding for short texts
    if len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess


class TTSModel(nn.Module):
    """TTS Model for MLX backend.

    Args:
        flow_lm: Flow language model.
        temp: Sampling temperature.
        lsd_decode_steps: Number of LSD decode steps.
        noise_clamp: Noise clamping value (None = disabled).
        eos_threshold: EOS detection threshold.
        config: Configuration object.
    """

    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: Optional[float],
        eos_threshold: float,
        config: Config,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True

    @property
    def device(self) -> str:
        """Get device (MLX uses unified memory)."""
        return "cpu"

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls,
        config: Config,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: Optional[float],
        eos_threshold: float,
    ):
        """Create model from Pydantic config.

        Args:
            config: Configuration object.
            temp: Sampling temperature.
            lsd_decode_steps: Number of LSD decode steps.
            noise_clamp: Noise clamping value.
            eos_threshold: EOS detection threshold.

        Returns:
            TTSModel instance.
        """
        from pocket_tts_mlx.utils.config import FlowLMConfig

        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm, latent_dim=config.mimi.quantizer.dimension
        )
        tts_model = cls(flow_lm, temp, lsd_decode_steps, noise_clamp, eos_threshold, config)
        return tts_model

    @classmethod
    def _from_pydantic_config_with_weights(
        cls,
        config: Config,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: Optional[float],
        eos_threshold: float,
    ):
        """Create model with loaded weights.

        Args:
            config: Configuration object.
            temp: Sampling temperature.
            lsd_decode_steps: Number of LSD decode steps.
            noise_clamp: Noise clamping value.
            eos_threshold: EOS detection threshold.

        Returns:
            TTSModel instance with loaded weights.
        """
        tts_model = cls._from_pydantic_config(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )

        # Initialize speaker projection weight
        tts_model.flow_lm.speaker_proj_weight = mx.zeros((1024, 512), dtype=mx.float32)

        # Load FlowLM weights
        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info(f"Loading FlowLM weights from {config.flow_lm.weights_path}")
            state_dict_flowlm = get_flow_lm_state_dict_mlx(
                download_if_necessary(config.flow_lm.weights_path)
            )
            load_weights_to_mlx_model(tts_model.flow_lm, state_dict_flowlm, strict=True)

        # Build Mimi model from config
        mimi_config = config.mimi.model_dump()

        encoder = SEANetEncoder(**mimi_config["seanet"])
        decoder = SEANetDecoder(**mimi_config["seanet"])
        encoder_transformer = ProjectedTransformer(**mimi_config["transformer"])
        decoder_transformer = ProjectedTransformer(**mimi_config["transformer"])
        quantizer = DummyQuantizer(**mimi_config["quantizer"])

        tts_model.mimi = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=mimi_config["channels"],
            sample_rate=mimi_config["sample_rate"],
            frame_rate=mimi_config["frame_rate"],
            encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        )

        # Load Mimi weights
        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path"
                )
            logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
            mimi_state = get_mimi_state_dict_mlx(download_if_necessary(config.mimi.weights_path))
            load_weights_to_mlx_model(tts_model.mimi, mimi_state, strict=True)

        # Load full TTS model weights
        if config.weights_path is not None:
            logger.info(f"Loading TTSModel weights from {config.weights_path}")
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception:
                tts_model.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

            # Load and convert weights
            # Use a simpler approach: load weights one by one
            loaded_count = 0
            skipped_count = 0
            with safetensors.safe_open(str(weights_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Convert BFloat16 to Float32 if needed (MLX doesn't support BFloat16)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.float()
                    # Convert to MLX array
                    mlx_array = mx.array(tensor.cpu().numpy())

                    # Navigate to the parameter using dot notation first
                    # so we can check if it's a Conv1d weight
                    parts = key.split('.')
                    obj = tts_model
                    try:
                        for part in parts[:-1]:
                            if part.isdigit():
                                # Try to index into list/Sequential
                                obj = obj[int(part)]
                            else:
                                obj = getattr(obj, part)
                        # Set the final parameter value
                        param_name = parts[-1]
                        if param_name.isdigit():
                            # This shouldn't happen with proper model structure
                            skipped_count += 1
                            continue

                        # Check if this is a Conv1d or ConvTranspose1d weight that needs transposing
                        # PyTorch Conv1d weight: (out_channels, in_channels, kernel_size)
                        # PyTorch ConvTranspose1d weight: (in_channels, out_channels, kernel_size)
                        # MLX Conv1d/ConvTranspose1d weight: (out_channels, kernel_size, in_channels)
                        is_conv_transpose_weight = (
                            param_name == 'weight' and
                            isinstance(obj, nn.ConvTranspose1d) and
                            len(mlx_array.shape) == 3
                        )
                        is_conv_weight = (
                            param_name == 'weight' and
                            isinstance(obj, nn.Conv1d) and
                            len(mlx_array.shape) == 3
                        )
                        if is_conv_transpose_weight:
                            # Check if this is a depthwise ConvTranspose1d
                            # PyTorch depthwise format: (in_channels, 1, kernel_size)
                            # Regular format: (in_channels, out_channels, kernel_size)
                            if mlx_array.shape[1] == 1:
                                # Depthwise ConvTranspose1d: (in, 1, kernel) -> MLX: (in, kernel, 1)
                                # Example: (512, 1, 32) -> (512, 32, 1)
                                mlx_array = mx.transpose(mlx_array, (0, 2, 1))
                            else:
                                # Regular ConvTranspose1d: (in, out, kernel) -> MLX: (out, kernel, in)
                                # Example: (512, 256, 12) -> (256, 12, 512)
                                mlx_array = mx.transpose(mlx_array, (1, 2, 0))
                        elif is_conv_weight:
                            # PyTorch Conv1d: (out, in, kernel) -> MLX: (out, kernel, in)
                            # Example: (512, 32, 1) -> (512, 1, 32)
                            mlx_array = mx.transpose(mlx_array, (0, 2, 1))

                        # Direct parameter assignment
                        setattr(obj, param_name, mlx_array)
                        loaded_count += 1
                    except (KeyError, IndexError, AttributeError) as e:
                        # Parameter doesn't exist in MLX model, skip
                        skipped_count += 1
                        continue

            logger.info(f"Loaded {loaded_count} weights, skipped {skipped_count}")

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning("No weights_path specified, model is uninitialized!")

        logger.info("TTS Model loaded successfully (MLX backend)")

        return tts_model

    @classmethod
    def load_model(
        cls,
        variant: str = DEFAULT_VARIANT,
        temp: float = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: Optional[float] = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    ):
        """Load a pre-trained TTS model.

        Args:
            variant: Model variant identifier.
            temp: Sampling temperature.
            lsd_decode_steps: Number of LSD decode steps.
            noise_clamp: Noise clamping value.
            eos_threshold: EOS detection threshold.

        Returns:
            TTSModel instance.
        """
        config = load_config(Path(__file__).parents[2] / f"config/{variant}.yaml")
        tts_model = cls._from_pydantic_config_with_weights(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )
        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: Dict,
        text_tokens: Optional[mx.array] = None,
        backbone_input_latents: Optional[mx.array] = None,
        audio_conditioning: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """Run FlowLM and increment step counter.

        Args:
            model_state: Model state.
            text_tokens: Text tokens.
            backbone_input_latents: Backbone input latents.
            audio_conditioning: Audio conditioning.

        Returns:
            Tuple of (backbone_output, audio_output).
        """
        if text_tokens is None:
            text_tokens = mx.zeros((1, 0), dtype=mx.int64)
        if backbone_input_latents is None:
            backbone_input_latents = mx.zeros((1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype)
        if audio_conditioning is None:
            audio_conditioning = mx.zeros((1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype)

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
        )

        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        # Handle two cases for state keys:
        # 1. Keys have 'flow_lm.' prefix - need to strip it
        # 2. Keys don't have prefix (from init_states(self.flow_lm, ...)) - use as-is
        flow_lm_state = {}
        for key, value in model_state.items():
            if key.startswith('flow_lm.'):
                # Remove 'flow_lm.' prefix
                new_key = key[len('flow_lm.'):]
                flow_lm_state[new_key] = value
            else:
                # No prefix - use key directly (already in flow_lm state format)
                flow_lm_state[key] = value
        increment_steps(self.flow_lm, flow_lm_state, increment=increment_by)

        return output

    def _run_flow_lm(
        self,
        model_state: Dict,
        text_tokens: mx.array,
        backbone_input_latents: mx.array,
        audio_conditioning: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Run FlowLM forward pass.

        Args:
            model_state: Model state.
            text_tokens: Text tokens.
            backbone_input_latents: Backbone input latents.
            audio_conditioning: Audio conditioning.

        Returns:
            Tuple of (output_embeddings, is_eos).
        """
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        text_embeddings = mx.concatenate([text_embeddings, audio_conditioning], axis=1)

        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )

        return output_embeddings[:, None, :], is_eos

    def _encode_audio(self, audio: mx.array) -> mx.array:
        """Encode audio to latent representation.

        Args:
            audio: Audio tensor.

        Returns:
            Encoded latent representation.
        """
        encoded = self.mimi.encode_to_latent(audio)
        latents = mx.transpose(encoded, (-1, -2)).astype(mx.float32)
        conditioning = mx.matmul(latents, self.flow_lm.speaker_proj_weight.T)
        return conditioning

    def _slice_kv_cache(self, model_state: Dict, num_frames: int) -> None:
        """Slice KV cache to only keep the first num_frames elements.

        Args:
            model_state: Model state dict.
            num_frames: Number of frames to keep.
        """
        for module_name, module_state in model_state.items():
            if "cache" in module_state:
                cache = module_state["cache"]
                # Slice to keep only the first num_frames positions
                module_state["cache"] = cache[:, :, :num_frames, :, :]
            # Also update current_end to match the sliced cache length
            if "current_end" in module_state:
                module_state["current_end"] = mx.arange(num_frames, dtype=mx.int64)

    def _expand_kv_cache(self, model_state: Dict, sequence_length: int) -> None:
        """Expand KV cache back to full sequence_length.

        Args:
            model_state: Model state dict.
            sequence_length: Target sequence length.
        """
        for module_name, module_state in model_state.items():
            if "cache" in module_state:
                cache = module_state["cache"]
                current_length = cache.shape[2]
                if current_length < sequence_length:
                    # Create expanded cache filled with NaN
                    expanded_cache = mx.full(
                        (
                            cache.shape[0],
                            cache.shape[1],
                            sequence_length,
                            cache.shape[3],
                            cache.shape[4],
                        ),
                        mx.nan,
                    )
                    # Copy existing data using slice_update
                    expanded_cache = mx.slice_update(
                        expanded_cache,
                        cache,
                        mx.array([0, 0, 0, 0, 0]),  # Start position
                        axes=[0, 1, 2, 3, 4],
                    )
                    module_state["cache"] = expanded_cache

    def _decode_audio_worker(
        self, latents_queue: queue.Queue, result_queue: queue.Queue
    ):
        """Worker thread for decoding audio latents.

        Args:
            latents_queue: Queue of latents to decode.
            result_queue: Queue for decoded audio chunks.
        """
        try:
            mimi_state = init_states(self.mimi, batch_size=1, sequence_length=1000)

            while True:
                latent = latents_queue.get()
                if latent is None:
                    break

                mimi_decoding_input = latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
                # MLX Conv1d expects channels-last format (B, L, C)
                # Input is already (B, T, C) which is the correct format
                quantized = self.mimi.quantizer(mimi_decoding_input)

                t = time.monotonic()
                audio_frame = self.mimi.decode_from_latent(quantized, mimi_state)
                increment_steps(self.mimi, mimi_state, increment=16)
                audio_frame_duration = audio_frame.shape[2] / self.config.mimi.sample_rate

                logger.debug(
                    " " * 30 + "Decoded %d ms of audio with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    int((time.monotonic() - t) * 1000),
                )

                result_queue.put(("chunk", mx.array(audio_frame)))
                latents_queue.task_done()

            result_queue.put(("done", None))

        except Exception as e:
            result_queue.put(("error", e))

    def generate_audio(
        self,
        model_state: Dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: Optional[int] = None,
        copy_state: bool = True,
    ) -> mx.array:
        """Generate complete audio tensor from text input.

        Args:
            model_state: Model state dictionary.
            text_to_generate: Input text to convert to speech.
            max_tokens: Maximum tokens per chunk.
            frames_after_eos: Frames to generate after EOS.
            copy_state: Whether to copy state before generation.

        Returns:
            Generated audio tensor.
        """
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
        ):
            audio_chunks.append(chunk)
        return mx.concatenate(audio_chunks, axis=0)

    def generate_audio_stream(
        self,
        model_state: Dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: Optional[int] = None,
        copy_state: bool = True,
    ) -> Generator[mx.array, None, None]:
        """Generate audio streaming chunks from text input.

        Args:
            model_state: Model state dictionary.
            text_to_generate: Input text to convert to speech.
            max_tokens: Maximum tokens per chunk.
            frames_after_eos: Frames to generate after EOS.
            copy_state: Whether to copy state before generation.

        Yields:
            Audio chunks as they become available.
        """
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer, text_to_generate, max_tokens
        )

        for chunk in chunks:
            text_to_generate, frames_after_eos_guess = prepare_text_prompt(chunk)
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=chunk,
                frames_after_eos=effective_frames,
                copy_state=copy_state,
            )

    def _generate_audio_stream_short_text(
        self, model_state: Dict, text_to_generate: str, frames_after_eos: int, copy_state: bool
    ):
        """Generate audio for short text.

        Args:
            model_state: Model state dictionary.
            text_to_generate: Input text.
            frames_after_eos: Frames to generate after EOS.
            copy_state: Whether to copy state before generation.

        Yields:
            Audio chunks.
        """
        if copy_state:
            # Deep copy model state
            model_state = copy.deepcopy(model_state)

        # Expand KV cache
        self._expand_kv_cache(model_state, sequence_length=1000)

        # MLX backend: Use synchronous generation to avoid Metal threading conflicts
        # Initialize mimi state for decoding
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=1000)

        logger.info("starting timer now!")
        t_generating = time.monotonic()

        # Generate latents and decode synchronously
        gen_len_sec = len(text_to_generate.split()) * 1 + 2.0
        max_gen_len = int(gen_len_sec * 12.5)
        prepared = self.flow_lm.conditioner.prepare(text_to_generate)

        # Initial text prompting
        with display_execution_time("Prompting text"):
            self._run_flow_lm_and_increment_step(
                model_state=model_state, text_tokens=prepared.tokens
            )

        # Autoregressive generation with synchronous decoding
        backbone_input = mx.full(
            (1, 1, self.flow_lm.ldim),
            float("nan"),
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None
        total_generated_samples = 0

        for generation_step in range(max_gen_len):
            with display_execution_time("Generating latent", print_output=False) as timer:
                # Generate next latent
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state, backbone_input_latents=backbone_input
                )

                # Extract scalar EOS value from (1, 1) tensor
                is_eos_scalar = bool(is_eos.item())
                if is_eos_scalar and eos_step is None:
                    eos_step = generation_step

                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

                # Decode latent to audio immediately (synchronous)
                mimi_decoding_input = next_latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
                quantized = self.mimi.quantizer(mimi_decoding_input)
                audio_frame = self.mimi.decode_from_latent(quantized, mimi_state)
                increment_steps(self.mimi, mimi_state, increment=16)

                # Yield audio chunk
                audio_chunk = audio_frame[0, 0]  # Remove batch, channel
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk

                backbone_input = next_latent

            # Capture timing after context manager exits
            steps_times.append(timer.elapsed_time_ms)

        logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)) if steps_times else "N/A")

        # Log timing
        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / generation_time

        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    def _generate(
        self,
        model_state: Dict,
        text_to_generate: str,
        frames_after_eos: int,
        latents_queue: queue.Queue,
        result_queue: queue.Queue,
    ):
        """Generate latents and add to queue.

        Args:
            model_state: Model state dictionary.
            text_to_generate: Input text.
            frames_after_eos: Frames to generate after EOS.
            latents_queue: Queue for latents.
            result_queue: Queue for results (for error reporting).
        """
        gen_len_sec = len(text_to_generate.split()) * 1 + 2.0
        max_gen_len = int(gen_len_sec * 12.5)
        prepared = self.flow_lm.conditioner.prepare(text_to_generate)

        with display_execution_time("Prompting text"):
            self._run_flow_lm_and_increment_step(
                model_state=model_state, text_tokens=prepared.tokens
            )

        def run_generation():
            try:
                self._autoregressive_generation(
                    model_state, max_gen_len, frames_after_eos, latents_queue
                )
            except Exception as e:
                logger.error(f"Error in autoregressive generation: {e}")
                if latents_queue is not None:
                    latents_queue.put(None)
                if result_queue is not None:
                    result_queue.put(("error", e))

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

    def _autoregressive_generation(
        self, model_state: Dict, max_gen_len: int, frames_after_eos: int, latents_queue: queue.Queue
    ):
        """Autoregressive generation loop.

        Args:
            model_state: Model state dictionary.
            max_gen_len: Maximum generation length.
            frames_after_eos: Frames to generate after EOS.
            latents_queue: Queue for generated latents.
        """
        backbone_input = mx.full(
            (1, 1, self.flow_lm.ldim),
            float("nan"),
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None

        for generation_step in range(max_gen_len):
            with display_execution_time("Generating latent", print_output=False) as timer:
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state, backbone_input_latents=backbone_input
                )

                # Extract scalar EOS value from (1, 1) tensor
                is_eos_scalar = bool(is_eos.item())
                if is_eos_scalar and eos_step is None:
                    eos_step = generation_step

                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

                latents_queue.put(next_latent)
                backbone_input = next_latent

            steps_times.append(timer.elapsed_time_ms)
        else:
            logger.warning("Maximum generation length reached without EOS")

        latents_queue.put(None)
        logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    @lru_cache(maxsize=2)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Union[Path, str, mx.array], truncate: bool = False
    ) -> Dict:
        """Cached version of get_state_for_audio_prompt.

        Args:
            audio_conditioning: Audio prompt (path, URL, or tensor).
            truncate: Whether to truncate long audio prompts.

        Returns:
            Model state dictionary.
        """
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def get_state_for_audio_prompt(
        self, audio_conditioning: Union[Path, str, mx.array], truncate: bool = False
    ) -> Dict:
        """Create model state conditioned on audio prompt.

        Args:
            audio_conditioning: Audio prompt (path, URL, or tensor).
            truncate: Whether to truncate long audio prompts.

        Returns:
            Model state dictionary.
        """
        if isinstance(audio_conditioning, str) and audio_conditioning in PREDEFINED_VOICES:
            prompt = load_predefined_voice_mlx(audio_conditioning)
        else:
            if not self.has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
                raise ValueError("Voice cloning not available without voice cloning weights")

            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)

            if isinstance(audio_conditioning, Path):
                audio, conditioning_sample_rate = audio_read(audio_conditioning)

                if truncate:
                    max_samples = int(30 * conditioning_sample_rate)
                    if audio.shape[-1] > max_samples:
                        audio = audio[..., :max_samples]
                        logger.info(f"Audio truncated to 30 seconds")

                audio_conditioning = convert_audio(
                    audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
                )

            with display_execution_time("Encoding audio prompt"):
                prompt = self._encode_audio(mx.array(audio_conditioning)[None, ...])

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=1000)

        with display_execution_time("Prompting audio"):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        # DISABLED: KV cache slicing was causing NaN issues
        # The slicing operation creates a mismatch between cache size and current_end
        # TODO: Fix this properly by updating current_end management
        # num_audio_frames = prompt.shape[1]
        # self._slice_kv_cache(model_state, num_audio_frames)

        return model_state
