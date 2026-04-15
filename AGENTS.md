# Repository Guidelines

## Project Structure & Module Organization
MLIA uses a Python `src/` layout. Main package code lives in `src/mlia/`, split across `backend/`, `cli/`, `core/`, `plugins/`, `target/`, `testing/`, and `utils/`. Static data such as target and optimization profiles lives under `src/mlia/resources/`. Tests live in `tests/` with names like `test_core_*.py` and `test_utils_*.py`. Documentation sources are in `docs/source/`, local pre-commit hooks live in `pre_commit_hooks/`, and OpenSpec artifacts live in `openspec/`.

## Build, Test, and Development Commands
Use `uv` for local workflows.

- `uv sync --group dev`: install development dependencies.
- `uv run pytest --no-success-flaky-report -m "not slow" tests/`: run the quick test suite used in CI.
- `uv run pytest --no-success-flaky-report tests/`: run the full test suite.
- `uv run pre-commit run --all-files --hook-stage=push`: run lint, formatting, SPDX, and commit checks.
- `uv build --wheel`: build the distributable wheel.
- `make -C docs html`: build Sphinx docs into `docs/build/`.

## Coding Style & Naming Conventions
Target Python 3.10+ and keep imports and packaging compatible with `pyproject.toml`. Use 4-space indentation, type annotations on new code, snake_case for modules, functions, and tests, and CapWords for classes. Ruff enforces imports and common lint rules; mypy runs with strict options such as `disallow_untyped_defs = true`. Pre-commit also runs `pyupgrade`, `pydocstyle`, `reuse`, `blocklint`, and a local copyright-header check.

## Testing Guidelines
Pytest is the test framework, with coverage required at `--cov-fail-under=65`. Name tests `test_*.py` and keep helper-only modules under `tests/utils/`. Use markers already declared in `pyproject.toml` such as `e2e`, `install`, `command`, `model_gen`, and `slow`. For end-to-end coverage, follow the shared patterns documented in `src/mlia/testing/README.md`.

## OpenSpec Workflow
Use the OpenSpec workflow skills when a task involves requirement discovery, proposing a change, implementing an approved change, or archiving a completed change. Trigger `openspec-explore` to clarify requirements or investigate scope, `openspec-propose` to draft a new change, `openspec-apply-change` to implement an approved change, and `openspec-archive-change` to close out completed work.

## Commit & Pull Request Guidelines
Commits follow a constrained Conventional Commits style, for example `feat: Add Python API for MLIA analysis` or `refactor: Remove in-process e2e CLI validation`. Valid types include `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `style`, and `test`, and the subject should be capitalized. Pull requests should include a clear summary, linked issue or rationale, updated tests for behavior changes, and notes on CLI or docs impact.

## Repository-Specific Notes
Preserve SPDX headers in new source files and run pre-commit before pushing. Avoid committing generated caches, local virtualenv changes, or large resource files unless they are intentional project inputs.
