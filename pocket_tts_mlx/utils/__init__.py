"""MLX utilities for pocket-tts.

This package contains utilities for working with MLX models, including
weight conversion from PyTorch to MLX.
"""

from pocket_tts_mlx.utils.config import (
    Config,
    FlowLMConfig,
    FlowLMTransformerConfig,
    MimiConfig,
    MimiTransformerConfig,
    load_config,
)
from pocket_tts_mlx.utils.weight_conversion import (
    convert_and_save_mlx_weights,
    convert_torch_tensor_to_mlx,
    get_flow_lm_state_dict_mlx,
    get_mimi_state_dict_mlx,
    get_tts_model_state_dict_mlx,
    load_predefined_voice_mlx,
    load_safetensors_to_mlx,
    load_weights_to_mlx_model,
)
from pocket_tts_mlx.utils.utils import (
    display_execution_time,
    download_if_necessary,
    size_of_dict,
)

__all__ = [
    # Weight conversion
    "convert_torch_tensor_to_mlx",
    "load_safetensors_to_mlx",
    "get_flow_lm_state_dict_mlx",
    "get_mimi_state_dict_mlx",
    "get_tts_model_state_dict_mlx",
    "load_weights_to_mlx_model",
    "convert_and_save_mlx_weights",
    "load_predefined_voice_mlx",
    # Utilities
    "display_execution_time",
    "download_if_necessary",
    "size_of_dict",
    # Config
    "Config",
    "FlowLMConfig",
    "FlowLMTransformerConfig",
    "MimiConfig",
    "MimiTransformerConfig",
    "load_config",
]
