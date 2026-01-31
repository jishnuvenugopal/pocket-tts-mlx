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

from pocket_tts_mlx.modules.attention import (
    MimiStreamingMultiheadAttention,
    StreamingMultiheadAttention,
)
from pocket_tts_mlx.modules.conv import (
    DepthwiseConvTranspose1d,
    StreamingConv1d,
    StreamingConvTranspose1d,
)
from pocket_tts_mlx.modules.layer_scale import LayerScale
from pocket_tts_mlx.modules.mlp import (
    LayerNorm as MLPLayerNorm,
    RMSNorm,
    RMSNorm as MLPRMSNorm,
    SimpleMLPAdaLN,
)
from pocket_tts_mlx.modules.resample import (
    ConvDownsample1d,
    ConvTrUpsample1d,
)
from pocket_tts_mlx.modules.rope import RotaryEmbedding
from pocket_tts_mlx.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts_mlx.modules.stateful_module import StatefulModule, increment_steps, init_states
from pocket_tts_mlx.modules.transformer import (
    LayerNorm as TransformerLayerNorm,
    RMSNorm as TransformerRMSNorm,
    ProjectedTransformer,
    StreamingTransformer,
    StreamingTransformerLayer,
)

__all__ = [
    # Stateful module base
    "StatefulModule",
    "init_states",
    "increment_steps",
    # Attention
    "StreamingMultiheadAttention",
    "MimiStreamingMultiheadAttention",
    # Convolutions
    "StreamingConv1d",
    "StreamingConvTranspose1d",
    "DepthwiseConvTranspose1d",
    # Rotary embeddings
    "RotaryEmbedding",
    # Layer scale
    "LayerScale",
    # MLP modules
    "SimpleMLPAdaLN",
    "RMSNorm",
    "MLPLayerNorm",
    "MLPRMSNorm",
    # Transformer modules
    "StreamingTransformer",
    "StreamingTransformerLayer",
    "ProjectedTransformer",
    "TransformerLayerNorm",
    "TransformerRMSNorm",
    # SEANet
    "SEANetEncoder",
    "SEANetDecoder",
    # Resample
    "ConvDownsample1d",
    "ConvTrUpsample1d",
]
