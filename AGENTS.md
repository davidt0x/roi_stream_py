# ROI Stream Agents Guide

## Purpose
- Python toolkit and GUI for ROI-based video processing and live visualization.
- Command-line entrypoints are exposed via `roi_stream` and related utilities defined in `pyproject.toml`.

## Environment
- Primary development happens on Windows via WSL; host webcams are not typically accessible inside WSL, so use recorded video sources when testing camera flows.
- Python tooling is managed with `uv`. Create or refresh the project environment with `uv sync` (this populates the `.venv` directory tracked locally).
- The canonical virtual environment for editors and tooling is `.venv/` at the repository root; point IDEs and language servers there.

## Daily Workflow
- Activate the environment for ad-hoc commands with `source .venv/bin/activate`, or run one-offs via `uv run …`.
- Install editable dependencies (including GUI extras) inside the managed environment using `uv pip install -e '.[gui]'` when GUI features are required.
- Run the unit test suite with `uv run pytest`; add markers (for example `-m offline`) when targeting specific subsets.
- Common CLI usage examples live in `README.md`; refer there for invocation patterns and device probing hints.

## Files of Note
- `pyproject.toml` defines dependencies, optional GUI extras, and console scripts.
- `src/roi_stream/gui_app.py` hosts the primary GUI logic.
- `tests/` contains pytest-based coverage, including smoke tests for the GUI layer.
- `examples/` provides sample ROI definitions for quick manual runs.
