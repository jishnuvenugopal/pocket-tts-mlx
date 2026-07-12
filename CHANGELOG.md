# Changelog

## Unreleased

### Added

- Added per-generation `temperature` and deterministic `seed` controls to the Python API and CLI.
- Use explicit per-generation MLX random keys so seeded requests are reproducible without mutating global RNG state.

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
