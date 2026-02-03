"""Text conditioner for MLX using SentencePiece and LUT embeddings."""

import logging

import mlx.core as mx
import mlx.nn as nn
import sentencepiece

from pocket_tts_mlx.conditioners.base import BaseConditioner, TokenizedText
from pocket_tts_mlx.utils.utils import download_if_necessary

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper with cached model download."""
    def __init__(self, n_bins: int, tokenizer_path: str):
        logger.info("Loading SentencePiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        assert n_bins == self.sp.vocab_size(), (
            f"sentencepiece tokenizer has vocab_size={self.sp.vocab_size()} "
            f"but n_bins={n_bins} was specified"
        )

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text into integer ids."""
        tokens = self.sp.encode(text, out_type=int)
        return TokenizedText(mx.array(tokens)[None, :])


class LUTConditioner(BaseConditioner):
    """Lookup-table conditioner that maps tokens to embeddings."""
    def __init__(self, n_bins: int, tokenizer_path: str, dim: int, output_dim: int):
        super().__init__(dim=dim, output_dim=output_dim)
        self.tokenizer = SentencePieceTokenizer(n_bins, tokenizer_path)
        self.embed = nn.Embedding(n_bins + 1, self.dim)

    def prepare(self, x: str) -> TokenizedText:
        """Tokenize raw text into TokenizedText."""
        return self.tokenizer(x)

    def _get_condition(self, inputs: TokenizedText) -> mx.array:
        """Embed token ids into conditioning vectors."""
        return self.embed(inputs.tokens)
