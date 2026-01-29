"""MLX module implementations for pocket-tts.

This package contains MLX implementations of all modules from the PyTorch version,
enabling efficient inference on Apple Silicon.

Modules:
    stateful_module: Base class for stateful streaming modules
    rope: Rotary positional embeddings
    attention: Streaming multi-head attention with KV cache
    transformer: Streaming transformer layers
    conv: Streaming 1D convolutions
    seanet: SEANet encoder/decoder
    mlp: MLP layers with adaptive layer normalization
    layer_scale: Layer scale for gradient flow
"""

from pocket_tts_mlx.modules.stateful_module import StatefulModule, increment_steps, init_states

__all__ = [
    "StatefulModule",
    "init_states",
    "increment_steps",
]
