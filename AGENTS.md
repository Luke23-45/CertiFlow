# opencode Agent Configuration

## Lint command
ruff check src/ tests/

## Type-check command
python -m mypy src/ --ignore-missing-imports

## Test command
pytest tests/ -v

## Build command
pip install -e .
