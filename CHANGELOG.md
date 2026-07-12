# Changelog

## Unreleased

### Fixed

- Split oversized single sentences at commas, semicolons, and colons before generation, keeping each natural clause group within `max_tokens` when possible. Previously, a sentence without terminal punctuation before the end could bypass the chunk limit and produce skipped words, garbled speech, or an excessively long audio tail.
- Warn when a chunk still exceeds `max_tokens` because it contains no usable sentence or clause boundary.
- Aligned the CLI `--max-tokens` default with the safer 50-token library default instead of 500.

### Tests

- Added regression coverage for oversized clause splitting and for preserving short sentences containing commas.

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
