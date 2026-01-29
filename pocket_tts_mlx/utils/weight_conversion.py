"""Weight conversion utilities for loading PyTorch weights into MLX models.

This module provides functions to convert PyTorch state dictionaries (loaded from
safetensors files) into MLX-compatible format for pocket-tts models.
"""

import logging
from pathlib import Path
from typing import Dict, Union

import mlx.core as mx
import numpy as np
import safetensors
import torch

logger = logging.getLogger(__name__)

# Predefined voices configuration
_voices_names = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
PREDEFINED_VOICES = {
    x: f"hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{x}.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    for x in _voices_names
}


def convert_torch_tensor_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to an MLX array.

    Args:
        tensor: PyTorch tensor to convert.

    Returns:
        MLX array with the same data.
    """
    # Convert to numpy first (faster than going through list)
    numpy_array = tensor.cpu().numpy()
    # Convert to MLX array
    return mx.array(numpy_array)


def load_predefined_voice_mlx(voice_name: str) -> mx.array:
    """Load a predefined voice embedding as an MLX array.

    Args:
        voice_name: Name of the predefined voice (e.g., "alba", "marius").

    Returns:
        MLX array containing the audio prompt embedding.

    Raises:
        ValueError: If the voice name is not found in PREDEFINED_VOICES.
    """
    from pocket_tts_mlx.utils.utils import download_if_necessary

    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found"
            f", available voices are {list(PREDEFINED_VOICES)}."
        )
    voice_file = download_if_necessary(PREDEFINED_VOICES[voice_name])
    # Load the audio_prompt tensor using safetensors and convert to MLX
    with safetensors.safe_open(voice_file, framework="pt", device="cpu") as f:
        tensor = f.get_tensor("audio_prompt")
    return convert_torch_tensor_to_mlx(tensor)


def load_safetensors_to_mlx(path: Union[str, Path], key_filter: str = None) -> Dict[str, mx.array]:
    """Load a safetensors file and convert all tensors to MLX arrays.

    Args:
        path: Path to the safetensors file.
        key_filter: Optional prefix to filter keys (e.g., "model." to only include keys starting with that).

    Returns:
        Dictionary mapping parameter names to MLX arrays.
    """
    mlx_state_dict = {}
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key_filter is not None and not key.startswith(key_filter):
                continue
            tensor = f.get_tensor(key)
            mlx_state_dict[key] = convert_torch_tensor_to_mlx(tensor)
    return mlx_state_dict


def get_flow_lm_state_dict_mlx(path: Path) -> Dict[str, mx.array]:
    """Load and convert FlowLM weights from safetensors to MLX format.

    This mirrors the PyTorch version but returns MLX arrays.

    Args:
        path: Path to the FlowLM safetensors file.

    Returns:
        Dictionary mapping parameter names to MLX arrays, with filtered and
        remapped keys compatible with the MLX FlowLMModel.
    """
    state_dict = {}
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            # Skip lookup table weights (not used in inference)
            if (
                key.startswith("flow.w_s_t.")
                or key == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
                or key == "condition_provider.conditioners.speaker_wavs.learnt_padding"
            ):
                continue

            new_name = key
            # Remap conditioner embedding weight
            if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
                new_name = "conditioner.embed.weight"
            # Remap speaker projection weight
            if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
                new_name = "speaker_proj_weight"

            tensor = f.get_tensor(key)
            state_dict[new_name] = convert_torch_tensor_to_mlx(tensor)

    logger.info(f"Loaded FlowLM state dict with {len(state_dict)} parameters")
    return state_dict


def get_mimi_state_dict_mlx(path: Path) -> Dict[str, mx.array]:
    """Load and convert Mimi weights from safetensors to MLX format.

    This mirrors the PyTorch version but returns MLX arrays.

    Args:
        path: Path to the Mimi safetensors file.

    Returns:
        Dictionary mapping parameter names to MLX arrays, with VQ quantizer
        weights removed and "model." prefix stripped from keys.
    """
    state_dict = {}
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            # Skip VQ quantizer weights (not used in TTS)
            if key.startswith("model.quantizer.vq.") or key == "model.quantizer.logvar_proj.weight":
                continue

            # Strip "model." prefix from keys
            new_key = key.removeprefix("model.")
            tensor = f.get_tensor(key)
            state_dict[new_key] = convert_torch_tensor_to_mlx(tensor)

    logger.info(f"Loaded Mimi state dict with {len(state_dict)} parameters")
    return state_dict


def get_tts_model_state_dict_mlx(path: Path) -> Dict[str, mx.array]:
    """Load and convert full TTSModel weights from safetensors to MLX format.

    Args:
        path: Path to the TTSModel safetensors file.

    Returns:
        Dictionary mapping parameter names to MLX arrays.
    """
    state_dict = {}
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            state_dict[key] = convert_torch_tensor_to_mlx(tensor)

    logger.info(f"Loaded TTSModel state dict with {len(state_dict)} parameters")
    return state_dict


def load_weights_to_mlx_model(model: "mlx.nn.Module", state_dict: Dict[str, mx.array], strict: bool = True) -> None:
    """Load weights into an MLX model.

    Args:
        model: MLX model to load weights into.
        state_dict: Dictionary of parameter names to MLX arrays.
        strict: If True, raise error if any keys are missing or unexpected.

    Raises:
        ValueError: If strict=True and there are missing or unexpected keys.
    """
    # Get model parameters
    model_params = dict(model.parameters())

    if strict:
        # Check for missing and unexpected keys
        model_keys = set(model_params.keys())
        state_keys = set(state_dict.keys())

        missing_keys = model_keys - state_keys
        unexpected_keys = state_keys - model_keys

        if missing_keys:
            raise ValueError(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in state_dict: {unexpected_keys}")

    # Update model parameters
    model.update(state_dict)
    logger.info(f"Loaded {len(state_dict)} parameters into model")


def convert_and_save_mlx_weights(
    torch_path: Path, mlx_path: Path, weight_loader: callable = None
) -> None:
    """Convert PyTorch safetensors to MLX format and save.

    Args:
        torch_path: Path to the PyTorch safetensors file.
        mlx_path: Path to save the MLX weights file.
        weight_loader: Optional function to load and process PyTorch weights
            (e.g., get_flow_lm_state_dict_mlx). If None, loads all tensors.
    """
    if weight_loader is not None:
        mlx_state_dict = weight_loader(torch_path)
    else:
        mlx_state_dict = load_safetensors_to_mlx(torch_path)

    # Save MLX weights using mx.save_safetensors if available
    # Otherwise save as numpy .npz file
    mlx_path = Path(mlx_path)
    if mlx_path.suffix == ".safetensors":
        # MLX doesn't have native safetensors save yet, save as .npz
        mlx_path = mlx_path.with_suffix(".npz")

    # Convert MLX arrays to numpy for saving
    numpy_dict = {k: np.array(v) for k, v in mlx_state_dict.items()}
    np.savez(mlx_path, **numpy_dict)

    logger.info(f"Saved MLX weights to {mlx_path}")
