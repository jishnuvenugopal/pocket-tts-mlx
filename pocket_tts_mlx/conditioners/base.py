"""Base conditioner for MLX.

This module provides the base class for all conditioner modules.
"""

import logging
from typing import Generic, NamedTuple, TypeVar

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

Prepared = TypeVar("Prepared")  # represents the prepared condition input type


class TokenizedText(NamedTuple):
    """Tokenized text representation.

    Args:
        tokens: Token IDs tensor (int64).
    """
    tokens: mx.array


class BaseConditioner(nn.Module, Generic[Prepared]):
    """Base model for all conditioner modules.

    Args:
        dim: Internal dim of the model.
        output_dim: Output dim of the conditioner.
        output_bias: If True, the output projection will have a bias.
        force_linear: Force linear projection even when dim == output_dim.
    """

    def __init__(
        self,
        dim: int,
        output_dim: int,
        output_bias: bool = False,
        force_linear: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        assert force_linear or dim != output_dim
        assert not output_bias

    def __call__(self, inputs: TokenizedText) -> mx.array:
        """Forward pass.

        Args:
            inputs: TokenizedText input.

        Returns:
            Conditioning tensor.
        """
        return self._get_condition(inputs)
