# Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/melvinrdn/fourierslm
cd fourierslm
uv sync --group dev
uv run pytest
```

Lint and format with [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`.

```bash
uvx ruff check .
uvx ruff format .
```

Build the documentation, with warnings as errors.

```bash
uv run --group docs sphinx-build -W docs docs/_build/html
```

The benchmark against the analytic Gaussian beam is described on [its page](benchmark.md).

The figure of the README comes from the same code as its example.

```bash
uv run --with matplotlib python docs/readme_figure.py
```
