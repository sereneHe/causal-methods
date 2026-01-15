# bestdagsolverintheworld

## Setup

If you need the Hydra launcher:

```bash
python -m pip install --upgrade hydra-joblib-launcher
```

## Reproducible tests (uv)

### Option A (recommended): run only `test_shd_utils.py` with minimal deps

`requirements.txt` includes optional solver dependencies (e.g. `pygobnilp`) that currently pull `numba/llvmlite`, which
is not compatible with Python 3.13.

Use this minimal file to run the SHD tests without installing the full stack:

```bash
cd "/Users/xiaoyuhe/Causal methods/bestdagsolverintheworld-main"
uv venv --clear

# Important: target the repo's venv explicitly (avoids installing into an already-activated venv)
uv pip install --python .venv/bin/python -r requirements-test-shd.txt

uv run --python .venv/bin/python pytest -q tests/test_shd_utils.py
```

### Option B: install full `requirements.txt`

If you need *all* dependencies in `requirements.txt`, you will likely need an older Python (often 3.9) due to
`llvmlite` version constraints.

Example:

```bash
cd "/Users/xiaoyuhe/Causal methods/bestdagsolverintheworld-main"
uv python install 3.9
uv venv --clear --python 3.9
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
```

## ExMAG experiments

Work-in-progress.

## Updated Project Structure

The `src/exdbn/` directory contains the following components:

```
src/exdbn/
├── __init__.py
├── cli.py              # Typer CLI (entry point)
├── config.py           # Explicit configuration (replaces env)
├── generate.py         # Data generation
├── run.py              # Parallel dispatcher
├── core.py             # Core algorithm logic (slightly modified)
```

Additional files:
- `tasks.py`: Invoke automation
- `pyproject.toml`: Project configuration

Refer to the respective files for detailed documentation.
