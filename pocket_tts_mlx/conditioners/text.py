"""Text conditioner for MLX.

This module implements text conditioning using SentencePiece tokenization
and lookup table embeddings. The implementation mirrors the PyTorch version
but uses MLX operations.
"""

import logging
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import sentencepiece

from pocket_tts_mlx.conditioners.base import BaseConditioner, TokenizedText
from pocket_tts_mlx.utils.utils import download_if_necessary

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """SentencePiece tokenizer for natural language.

    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77, PAD, PAD, PAD, PAD, PAD, PAD]]

    Args:
        n_bins: Number of bins (should equal vocab size).
        tokenizer_path: Path to SentencePiece tokenizer model.
    """

    def __init__(self, n_bins: int, tokenizer_path: str):
        logger.info("Loading SentencePiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        assert n_bins == self.sp.vocab_size(), (
            f"sentencepiece tokenizer has vocab_size={self.sp.vocab_size()} "
            f"but n_bins={n_bins} was specified"
        )

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text.

        Args:
            text: Input text string.

        Returns:
            TokenizedText with token IDs (with batch dimension).
        """
        tokens = self.sp.encode(text, out_type=int)
        # SentencePiece returns numpy array, convert to mx.array with batch dimension
        return TokenizedText(mx.array(tokens)[None, :])


class LUTConditioner(BaseConditioner):
    """Lookup table text conditioner.

    Args:
        n_bins: Number of bins (vocab size).
        tokenizer_path: Path to tokenizer model.
        dim: Hidden dim of the conditioner.
        output_dim: Output dim of the conditioner.
    """

    def __init__(self, n_bins: int, tokenizer_path: str, dim: int, output_dim: int):
        super().__init__(dim=dim, output_dim=output_dim)
        self.tokenizer = SentencePieceTokenizer(n_bins, tokenizer_path)
        # n_bins + 1 for padding token
        self.embed = nn.Embedding(n_bins + 1, self.dim)

    def prepare(self, x: str) -> TokenizedText:
        """Prepare text input.

        Args:
            x: Input text string.

        Returns:
            TokenizedText with tokens.
        """
        tokens = self.tokenizer(x)
        return tokens

    def _get_condition(self, inputs: TokenizedText) -> mx.array:
        """Get conditioning from tokens.

        Args:
            inputs: TokenizedText containing token IDs.

        Returns:
            Embedding tensor of shape [B, T, D].
        """
        embeds = self.embed(inputs.tokens)
        return embeds
