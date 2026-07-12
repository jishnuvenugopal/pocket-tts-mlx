# Changelog

## Unreleased

### Added

- Ported upstream text normalization for decimals and currency so structural punctuation is converted to spoken text before tokenization.
- Added `UserDictionary` pronunciation overrides with JSON/YAML loading, language-specific and common sections, literal or regex matching, and dictionary composition.
- Added CLI `--dictionary` support and automatic loading from `~/.config/pocket-tts/dictionary.{yaml,yml,json}`.

### Tests

- Added regression coverage for text normalization, dictionary loading and composition, and generation-path pronunciation overrides.

## v0.2.1 - 2026-02-11

### Added

- Startup artifact cleanup controls in CLI and Python API:
  - `warmup_frames` / `--warmup-frames`
  - `trim_start_ms` / `--trim-start-ms`
  - `fade_in_ms` / `--fade-in-ms`

### Improved

- Ported upstream dynamic KV cache sizing behavior to reduce unnecessary cache preallocation.
- Materialized generated audio in `generate_audio()` to align reported timing with end-to-end usage.
- README updated with recommended clean-onset command and option explanations.

## v0.2.0 - 2026-02-03

- Initial public MLX release on PyPI.
