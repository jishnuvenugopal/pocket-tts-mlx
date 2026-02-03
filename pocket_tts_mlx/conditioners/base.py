"""Base classes for conditioners."""

import logging
from typing import Generic, NamedTuple, TypeVar

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class TokenizedText(NamedTuple):
    """Token container for text conditioning."""
    tokens: mx.array


Input = TypeVar("Input")


class BaseConditioner(nn.Module, Generic[Input]):
    """Base class for conditioners that map inputs to embeddings."""
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim

    def prepare(self, x: Input) -> Input:
        """Normalize or tokenize inputs before embedding."""
        return x

    def _get_condition(self, inputs: Input) -> mx.array:
        """Return raw embeddings for prepared inputs."""
        raise NotImplementedError

    def __call__(self, inputs: Input) -> mx.array:
        """Compute conditioning embeddings with shape validation."""
        out = self._get_condition(inputs)
        assert out.shape[-1] == self.dim
        return out
