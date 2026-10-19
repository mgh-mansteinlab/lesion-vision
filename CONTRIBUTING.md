# Contributing

1. Use a feature branch; keep commits focused.
2. Run `pytest tests/ -v` and `python -m compileall -q src scripts` before opening a PR.
3. Do not commit large binaries (`*.pth`, NDPI, `predictions/`, `wandb/`, `manuscript/`). Checkpoints and imaging data are on request only; the repository contains code only.
4. For experiment changes, update `experiments/EXPERIMENT_LOG.md` and `docs/reproducibility/BASELINE.md` when baseline metrics shift.

## Code style

- Prefer `ruff check` (see `pyproject.toml`).
- Keep training and inference paths configurable via CLI or environment variables, not hard-coded machine paths.
