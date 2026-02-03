"""Weight conversion utilities for loading safetensors into MLX models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Union

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # handled separately
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U64": np.uint64,
    "U32": np.uint32,
    "U16": np.uint16,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

_voices_names = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
PREDEFINED_VOICES = {
    x: f"hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{x}.safetensors@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
    for x in _voices_names
}


def load_safetensors_to_numpy(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    path = Path(path)
    with open(path, "rb") as f:
        # Safetensors header stores tensor metadata and byte offsets.
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len

        tensors: Dict[str, np.ndarray] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)

            if dtype not in _DTYPE_MAP:
                raise ValueError(f"Unsupported safetensors dtype: {dtype}")

            if dtype == "BF16":
                # BF16 is stored as uint16; restore by shifting into float32.
                u16 = np.frombuffer(raw, dtype=np.uint16)
                f32 = (u16.astype(np.uint32) << 16).view(np.float32)
                tensor = f32.reshape(shape)
            else:
                tensor = np.frombuffer(raw, dtype=_DTYPE_MAP[dtype]).reshape(shape)

            tensors[name] = tensor

        return tensors


def convert_torch_tensor_to_mlx(tensor: np.ndarray) -> mx.array:
    if getattr(tensor.dtype, "name", "") == "bfloat16":
        tensor = tensor.astype("float32")
    return mx.array(tensor)


def load_predefined_voice_mlx(voice_name: str) -> mx.array:
    from pocket_tts_mlx.utils.utils import download_if_necessary

    if voice_name not in PREDEFINED_VOICES:
        raise ValueError(
            f"Predefined voice '{voice_name}' not found, available voices are {list(PREDEFINED_VOICES)}."
        )
    voice_file = download_if_necessary(PREDEFINED_VOICES[voice_name])
    tensor = load_safetensors_to_numpy(voice_file).get("audio_prompt")
    if tensor is None:
        raise KeyError("audio_prompt not found in voice embedding file")
    return convert_torch_tensor_to_mlx(tensor)


def load_safetensors_to_mlx(path: Union[str, Path], key_filter: str | None = None) -> Dict[str, mx.array]:
    mlx_state_dict: Dict[str, mx.array] = {}
    tensors = load_safetensors_to_numpy(path)
    for key, tensor in tensors.items():
        if key_filter is not None and not key.startswith(key_filter):
            continue
        mlx_state_dict[key] = convert_torch_tensor_to_mlx(tensor)
    return mlx_state_dict


def get_flow_lm_state_dict_mlx(path: Path) -> Dict[str, mx.array]:
    state_dict: Dict[str, mx.array] = {}
    tensors = load_safetensors_to_numpy(path)
    for key, tensor in tensors.items():
        if (
            key.startswith("flow.w_s_t.")
            or key == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
            or key == "condition_provider.conditioners.speaker_wavs.learnt_padding"
        ):
            continue

        # Normalize key names to match MLX module structure.
        new_name = key
        if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
            new_name = "conditioner.embed.weight"
        if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
            new_name = "speaker_proj_weight"

        state_dict[new_name] = convert_torch_tensor_to_mlx(tensor)
    logger.info("Loaded FlowLM state dict with %d parameters", len(state_dict))
    return state_dict


def get_mimi_state_dict_mlx(path: Path) -> Dict[str, mx.array]:
    state_dict: Dict[str, mx.array] = {}
    tensors = load_safetensors_to_numpy(path)
    for key, tensor in tensors.items():
        if key.startswith("model.quantizer.vq.") or key == "model.quantizer.logvar_proj.weight":
            continue
        new_key = key.removeprefix("model.")
        state_dict[new_key] = convert_torch_tensor_to_mlx(tensor)
    logger.info("Loaded Mimi state dict with %d parameters", len(state_dict))
    return state_dict


def get_tts_model_state_dict_mlx(path: Path) -> Dict[str, mx.array]:
    state_dict: Dict[str, mx.array] = {}
    tensors = load_safetensors_to_numpy(path)
    for key, tensor in tensors.items():
        state_dict[key] = convert_torch_tensor_to_mlx(tensor)
    logger.info("Loaded TTSModel state dict with %d parameters", len(state_dict))
    return state_dict


def load_weights_to_mlx_model(model: "mlx.nn.Module", state_dict: Dict[str, mx.array], strict: bool = True) -> None:
    model_params = dict(model.parameters())
    if strict:
        model_keys = set(model_params.keys())
        state_keys = set(state_dict.keys())
        missing_keys = model_keys - state_keys
        unexpected_keys = state_keys - model_keys
        if missing_keys:
            raise ValueError(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            raise ValueError(f"Unexpected keys in state_dict: {unexpected_keys}")
    model.update(state_dict)
    logger.info("Loaded %d parameters into model", len(state_dict))


def convert_and_save_mlx_weights(
    torch_path: Path, mlx_path: Path, weight_loader: callable | None = None
) -> None:
    if weight_loader is not None:
        mlx_state_dict = weight_loader(torch_path)
    else:
        mlx_state_dict = load_safetensors_to_mlx(torch_path)

    mlx_path = Path(mlx_path)
    if mlx_path.suffix == ".safetensors":
        mlx_path = mlx_path.with_suffix(".npz")
    numpy_dict = {k: np.array(v) for k, v in mlx_state_dict.items()}
    np.savez(mlx_path, **numpy_dict)
    logger.info("Saved MLX weights to %s", mlx_path)
