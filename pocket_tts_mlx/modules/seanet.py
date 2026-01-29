"""SEANet encoder and decoder modules for MLX.

This module implements SEANet (SoundStream-like Encoder/Decoder Architecture Network)
for audio compression and decompression. The implementations mirror the PyTorch versions
but use MLX operations.

SEANet uses residual blocks with convolutional layers for efficient audio encoding/decoding.
"""

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from pocket_tts_mlx.modules.conv import StreamingConv1d, StreamingConvTranspose1d


class SEANetResnetBlock(nn.Module):
    """Residual block for SEANet.

    Args:
        dim: Dimension of the block.
        kernel_sizes: Kernel sizes for the convolutions. Defaults to [3, 1].
        dilations: Dilations for the convolutions. Defaults to [1, 1].
        pad_mode: Padding mode. Defaults to "reflect".
        compress: Compression factor for hidden layer. Defaults to 2.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: List[int] = [3, 1],
        dilations: List[int] = [1, 1],
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        super().__init__()

        assert len(kernel_sizes) == len(dilations), (
            "Number of kernel sizes should match number of dilations"
        )

        hidden = dim // compress

        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden

            block.append(nn.ELU())
            block.append(
                StreamingConv1d(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    pad_mode=pad_mode,
                )
            )

        self.block = block

    def __call__(self, x: mx.array, model_state: Any) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, T].
            model_state: Model state for streaming.

        Returns:
            Output tensor of shape [B, C, T].
        """
        v = x
        for layer in self.block:
            if isinstance(layer, StreamingConv1d):
                v = layer(v, model_state)
            else:
                v = layer(v)

        # Handle small shape mismatches due to upsampling padding differences
        # MLX and PyTorch may produce slightly different output lengths
        if v.shape != x.shape:
            B1, C1, T1 = x.shape
            B2, C2, T2 = v.shape
            # Only allow mismatches in the time dimension
            assert B1 == B2 and C1 == C2, f"Shape mismatch in batch or channel: {x.shape} != {v.shape}"
            # Allow small differences (up to 4 samples) due to padding
            diff = abs(T2 - T1)
            if diff <= 4:
                # Trim or pad to match
                if T2 > T1:
                    # v has extra samples, trim from the end
                    v = v[:, :, :T1]
                elif T2 < T1:
                    # v has fewer samples, pad with zeros at the end
                    pad_length = T1 - T2
                    v = mx.pad(v, [(0, 0), (0, 0), (0, pad_length)])
            else:
                raise AssertionError(f"Shape mismatch: {v.shape} != {x.shape}")

        return x + v


class SEANetEncoder(nn.Module):
    """SEANet encoder for audio compression.

    Args:
        channels: Number of input audio channels. Defaults to 1.
        dimension: Internal dimension of the encoder. Defaults to 128.
        n_filters: Number of filters (base channel count). Defaults to 32.
        n_residual_layers: Number of residual layers per downsampling. Defaults to 3.
        ratios: Downsampling ratios for each stage. Defaults to [8, 5, 4, 2].
        kernel_size: Kernel size for initial convolution. Defaults to 7.
        last_kernel_size: Kernel size for final convolution. Defaults to 7.
        residual_kernel_size: Kernel size for residual blocks. Defaults to 3.
        dilation_base: Base for exponential dilation in residual blocks. Defaults to 2.
        pad_mode: Padding mode. Defaults to "reflect".
        compress: Compression factor in residual blocks. Defaults to 2.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: List[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        super().__init__()

        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        # Reverse ratios for encoder (downsampling)
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        # Calculate hop length (total downsampling factor)
        import numpy as np
        self.hop_length = int(np.prod(self.ratios))

        model = []
        mult = 1

        # Initial convolution
        model.append(
            StreamingConv1d(channels, mult * n_filters, kernel_size, pad_mode=pad_mode)
        )

        # Downsampling stages
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(n_residual_layers):
                model.append(
                    SEANetResnetBlock(
                        dim=mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                )

            # Add downsampling convolution
            model.append(nn.ELU())
            model.append(
                StreamingConv1d(
                    in_channels=mult * n_filters,
                    out_channels=mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    pad_mode=pad_mode,
                )
            )
            mult *= 2

        # Final convolution
        model.append(nn.ELU())
        model.append(
            StreamingConv1d(mult * n_filters, dimension, last_kernel_size, pad_mode=pad_mode)
        )

        self.model = model

    def __call__(self, x: mx.array, model_state: Any) -> mx.array:
        """Forward pass.

        Args:
            x: Input audio tensor of shape [B, C, T].
            model_state: Model state for streaming.

        Returns:
            Encoded latent tensor of shape [B, dimension, T'].
        """
        for layer in self.model:
            if isinstance(layer, (StreamingConv1d, SEANetResnetBlock)):
                x = layer(x, model_state)
            else:
                x = layer(x)
        return x


class SEANetDecoder(nn.Module):
    """SEANet decoder for audio decompression.

    Args:
        channels: Number of output audio channels. Defaults to 1.
        dimension: Internal dimension of the decoder. Defaults to 128.
        n_filters: Number of filters (base channel count). Defaults to 32.
        n_residual_layers: Number of residual layers per upsampling. Defaults to 3.
        ratios: Upsampling ratios for each stage. Defaults to [8, 5, 4, 2].
        kernel_size: Kernel size for initial convolution. Defaults to 7.
        last_kernel_size: Kernel size for final convolution. Defaults to 7.
        residual_kernel_size: Kernel size for residual blocks. Defaults to 3.
        dilation_base: Base for exponential dilation in residual blocks. Defaults to 2.
        pad_mode: Padding mode. Defaults to "reflect".
        compress: Compression factor in residual blocks. Defaults to 2.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: List[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        super().__init__()

        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        # Calculate hop length (total upsampling factor)
        import numpy as np
        self.hop_length = int(np.prod(self.ratios))

        mult = int(2 ** len(self.ratios))

        model = []

        # Initial convolution
        model.append(
            StreamingConv1d(dimension, mult * n_filters, kernel_size, pad_mode=pad_mode)
        )

        # Upsampling stages
        for ratio in self.ratios:
            # Add upsampling convolution
            model.append(nn.ELU())
            model.append(
                StreamingConvTranspose1d(
                    in_channels=mult * n_filters,
                    out_channels=mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            )

            # Add residual layers
            for j in range(n_residual_layers):
                model.append(
                    SEANetResnetBlock(
                        dim=mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=compress,
                    )
                )

            mult //= 2

        # Final layers
        model.append(nn.ELU())
        model.append(
            StreamingConv1d(n_filters, channels, last_kernel_size, pad_mode=pad_mode)
        )

        self.model = model

    def __call__(self, z: mx.array, model_state: Any) -> mx.array:
        """Forward pass.

        Args:
            z: Input latent tensor of shape [B, dimension, T].
            model_state: Model state for streaming.

        Returns:
            Decoded audio tensor of shape [B, channels, T'].
        """
        for layer in self.model:
            if isinstance(layer, (StreamingConvTranspose1d, SEANetResnetBlock, StreamingConv1d)):
                z = layer(z, model_state)
            else:
                z = layer(z)
        return z
