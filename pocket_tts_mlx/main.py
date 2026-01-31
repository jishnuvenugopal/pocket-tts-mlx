"""Command-line interface for pocket-tts-mlx."""

import argparse
import logging
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from tqdm import tqdm

from pocket_tts_mlx import TTSModel

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate speech from text using pocket-tts with MLX backend"
    )
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument(
        "--voice", "-v", default="marius", help="Voice name (default: marius)"
    )
    parser.add_argument(
        "--output", "-o", default="output.wav", help="Output audio file (default: output.wav)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate (default: 500)",
    )
    parser.add_argument(
        "--frames-after-eos",
        type=int,
        default=7,
        help="Frames to generate after EOS (default: 7)",
    )
    parser.add_argument(
        "--verbose", "-V", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    try:
        # Load model
        logger.info("Loading MLX model...")
        model = TTSModel.load_model()

        # Get model state with voice
        logger.info(f"Loading voice: {args.voice}")
        model_state = model.get_state_for_audio_prompt(args.voice)

        # Generate audio
        logger.info(f"Generating audio for: \"{args.text}\"")
        audio = model.generate_audio(
            model_state=model_state,
            text_to_generate=args.text,
            max_tokens=args.max_tokens,
            frames_after_eos=args.frames_after_eos,
        )

        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to: {output_path}")
        sf.write(str(output_path), np.array(audio), model.config.mimi.sample_rate)

        duration = audio.shape[-1] / model.config.mimi.sample_rate
        logger.info(f"Generated {duration:.2f}s of audio")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
