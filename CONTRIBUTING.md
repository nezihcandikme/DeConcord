# Contributing to DEConcord

This is a small, actively-developed project (see [`DEVLOG.md`](DEVLOG.md)
for the day-by-day reasoning behind it). Contributions, bug reports, and
questions are welcome — this guide is short on purpose so you'll actually
read it.

## Setup

```bash
git clone https://github.com/nezihcandikme/DeConcord.git
cd DeConcord
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

`.[dev]` installs `pytest`, `pytest-cov`, `ruff`, and `mypy` alongside the
package itself. The optional AI-summary feature (`explain_results=True`)
needs an extra: `pip install -e ".[ai]"`.

## Running tests

```bash
pytest
```

All tests must pass before a pull request is merged. If you're fixing a
bug, add a test that fails without your fix and passes with it — a
regression test, not just the fix itself.

## Code style

- `ruff check src/ tests/ benchmarks/ examples/` should report no errors (`ruff check --fix` for
  the auto-fixable ones, mostly import ordering).
- `mypy src/deconcord --ignore-missing-imports` should report no errors -- CI-gated, alongside ruff.
- Type hints on public function signatures.
- Docstrings on public functions: what it does, parameters, returns,
  raises — see any existing function in `src/deconcord/` for the expected
  level of detail. Explain *why* a design choice was made where it isn't
  obvious, not just *what* the code does.
- Error messages should name the actual problem (which sample, which
  column, which value) rather than a bare exception type. See
  `src/deconcord/io/counts.py` for examples.
- Prefer raising early and loudly over silently producing a plausible-looking
  wrong answer. This project's whole premise is being trustworthy about
  differential expression conclusions — a function that fails quietly on
  bad input undermines that more than one that crashes with a clear message.

## Scope

DEConcord's mission is deliberately narrow: robustness and concordance
analysis for RNA-seq differential expression (see the README's "What
DEConcord does not build" section). A pull request adding single-cell
support, a new omics data type, or a general-purpose ML feature is
unlikely to be merged, however well-implemented — not a reflection on the
code, just outside what this project is trying to be. If you're unsure
whether something fits, open an issue first and ask.

## Submitting issues

Use the bug report or feature request template. For bugs, a minimal
reproducible example (a small count matrix or DE result table that
triggers the problem) makes a fix much faster than a description alone.

## Submitting pull requests

1. Fork the repo and create a branch from `main`.
2. Make your change, with tests.
3. Run `pytest`, `ruff check src/ tests/ benchmarks/ examples/`, and
   `mypy src/deconcord --ignore-missing-imports` locally.
4. Open a PR using the template — what changed, why, what tests you ran.
5. Update `CHANGELOG.md` under `[Unreleased]` if the change is user-facing.

Keep pull requests focused on one change. A PR that fixes a bug and
reorganizes an unrelated module is harder to review and harder to revert
if something goes wrong.
