"""Command-line interface for pocket-tts-mlx."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pocket_tts_mlx import TTSModel

logger = logging.getLogger(__name__)


def main() -> int:
    """Parse CLI arguments, run TTS, and write WAV output."""
    parser = argparse.ArgumentParser(
        description="Generate speech from text using pocket-tts with MLX backend"
    )
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("--voice", "-v", default="marius", help="Voice name (default: marius)")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens per chunk")
    parser.add_argument("--frames-after-eos", type=int, default=7, help="Frames after EOS")
    parser.add_argument(
        "--trim-start-ms",
        type=int,
        default=0,
        help="Trim this many milliseconds from start of generated audio",
    )
    parser.add_argument(
        "--fade-in-ms",
        type=int,
        default=0,
        help="Apply linear fade-in over this many milliseconds",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=1,
        help="Number of initial Mimi frames to decode and discard for cleaner onset",
    )
    parser.add_argument("--verbose", "-V", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    try:
        logger.info("Loading MLX model...")
        model = TTSModel.load_model()

        logger.info("Loading voice: %s", args.voice)
        model_state = model.get_state_for_audio_prompt(args.voice)

        logger.info("Generating audio...")
        audio = model.generate_audio(
            model_state=model_state,
            text_to_generate=args.text,
            max_tokens=args.max_tokens,
            frames_after_eos=args.frames_after_eos,
            trim_start_ms=args.trim_start_ms,
            fade_in_ms=args.fade_in_ms,
            warmup_frames=args.warmup_frames,
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write float audio as WAV at model sample rate.
        sf.write(str(out_path), np.array(audio), model.config.mimi.sample_rate)
        duration = audio.shape[-1] / model.config.mimi.sample_rate
        logger.info("Wrote %s (%.2fs)", out_path, duration)
        return 0
    except Exception as exc:
        logger.error("Error: %s", exc)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
