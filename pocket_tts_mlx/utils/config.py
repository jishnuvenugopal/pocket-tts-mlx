"""Configuration models for loading YAML config files.

Adapted from pocket-tts for MLX compatibility.
"""

from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model with strict validation (no extra fields allowed)."""
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    """Configuration for flow matching model."""
    dim: int
    depth: int


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    """Configuration for FlowLM transformer."""
    hidden_scale: int
    max_period: int
    d_model: int
    num_heads: int
    num_layers: int


class LookupTable(StrictModel):
    """Configuration for lookup table conditioner."""
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str


# Root configuration for FlowLM
class FlowLMConfig(StrictModel):
    """Root configuration model for FlowLM YAML config files."""

    dtype: str

    # Nested configurations
    flow: FlowConfig
    transformer: FlowLMTransformerConfig

    # Conditioning
    lookup_table: LookupTable
    weights_path: Union[str, None] = None


# SEANet configuration
class SEANetConfig(StrictModel):
    """Configuration for SEANet encoder/decoder."""
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int


# Transformer configuration for Mimi
class MimiTransformerConfig(StrictModel):
    """Configuration for Mimi transformer."""
    d_model: int
    input_dimension: int
    output_dimensions: tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    max_period: float = 10000.0
    dim_feedforward: int


# Quantizer configuration
class QuantizerConfig(StrictModel):
    """Configuration for quantizer."""
    dimension: int
    output_dimension: int


# Root configuration for Mimi
class MimiConfig(StrictModel):
    """Root configuration model for Mimi YAML config files."""

    dtype: str

    # Sample rate and channels
    sample_rate: int
    channels: int
    frame_rate: float

    # SEANet configurations
    seanet: SEANetConfig

    # Transformer
    transformer: MimiTransformerConfig

    # Quantizer
    quantizer: QuantizerConfig
    weights_path: Union[str, None] = None


# Root configuration for complete TTS model
class Config(StrictModel):
    """Root configuration model for complete TTS YAML config files."""
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: Union[str, None] = None
    weights_path_without_voice_cloning: Union[str, None] = None


def load_config(yaml_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        Config object with parsed configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
