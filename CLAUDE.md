# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

DEConcord: robustness and concordance analysis for RNA-seq differential expression. Given DE result tables from DESeq2, edgeR, or anything with a gene ID/log fold change/p-value, it checks whether their conclusions hold up under reasonable changes — a different DE tool, a different significance threshold, a different resampling of the samples. It is not trying to replace DESeq2/edgeR, and it is not headed toward single-cell, spatial, proteomics, or general ML tooling — see the README's "What DEConcord does not build" section and `CONTRIBUTING.md`'s "Scope" section.

The core question (concordance/threshold sensitivity/pathway stability/resampling stability) is the identity of the project. The RNA-seq pipeline (validation, QC, normalization, from-scratch DE, annotation, enrichment, plotting) is kept infrastructure — it's what generates the DE tables the concordance functions compare, not a competing feature set.

## Commands

```bash
pip install -e ".[dev]"                              # dev setup: pytest + pytest-cov + ruff + mypy
pytest -q                                             # run all tests
pytest tests/test_concordance.py -q                   # run one test file
pytest tests/test_concordance.py::test_name -q        # run a single test
pytest --cov=deconcord --cov-report=term-missing -q  # run tests with a coverage report
ruff check src/ tests/ benchmarks/ examples/           # lint (required, CI-gated)
ruff check src/ tests/ benchmarks/ examples/ --fix     # auto-fix (mostly import ordering)
mypy src/deconcord --ignore-missing-imports           # type check (required, CI-gated)
python examples/quickstart.py                          # full workflow demo, no network/R required
```

Ruff is scoped to `E9, F, FA` (syntax errors, pyflakes-equivalent, and type-hint py-version compatibility) rather than opinionated style — see `pyproject.toml`'s `[tool.ruff.lint]` comment. CI (`.github/workflows/tests.yml`) runs lint, a `mypy` type check, `pytest -v` across Python 3.9-3.12, then a CLI smoke test against a real fixture. The package ships a `py.typed` marker (PEP 561), so its type hints are meant to be relied on by downstream callers' own type checkers, not just treated as documentation.

## Architecture

`src/deconcord/` is organized by pipeline stage, one subpackage per stage, each with a `methods.py` (or similarly named module) holding the actual functions:

- `io/` — `counts.py` (load/validate count matrices), `geo.py` (GEO dataset acquisition)
- `qc/metrics.py` — sample QC (library size, genes detected, outlier flags); uses `core.py`'s bare `library_size`/`genes_detected` helpers
- `filtering/methods.py` — low-count gene filtering
- `normalization/methods.py` — CPM + log2 transform
- `differential_expression/methods.py` — DEConcord's own from-scratch DE: Welch's t-test, empirical-Bayes moderated t-test, covariate-adjusted linear model
- `concordance/` — the actual core of the project: `methods.py` (`compute_de_concordance`, pairwise comparison between two DE tables), `threshold_sensitivity.py`, `pathway_stability.py`, `resampling_stability.py`
- `annotation/methods.py` — gene ID → symbol mapping
- `pathway_analysis/` — `methods.py` (local GMT-based enrichment), `gmt.py` (GMT file parsing), `live_enrichment.py` (live g:Profiler API call)
- `visualization/plots.py` — volcano, PCA, pathway enrichment dot plots
- `ai_explanation/methods.py` — optional, off-by-default plain-language result narration (needs the `ai` extra)

`pipeline.py` (`run_analysis`) wires these into one end-to-end call. `cli.py` is the `deconcord` entry point: a bare `deconcord counts.csv --group1 ... --group2 ...` invocation runs the pipeline; `deconcord concordance a.csv b.csv ...` (checked as a literal first-argument string, not a real argparse subparser — see the Aug 17 DEVLOG entry for why) wraps `compute_de_concordance` directly on two existing result tables. Both write a `run_metadata.json` (version, params, timestamp) alongside their output for reproducibility.

`core.py` holds small dependency-free helpers used by multiple stages (currently just `library_size`/`genes_detected`).

`src/deconcord/__init__.py` re-exports the public API flat (`import deconcord as dc; dc.compute_de_concordance(...)`) — every name is also reachable from its actual submodule. When adding a new public function, add it to both the imports and `__all__` there.

Lazy imports are used deliberately for two things that shouldn't be required dependencies unless actually invoked: `ai_explanation` (needs `anthropic`/`python-dotenv`, the `ai` extra) and `live_enrichment` (makes a live network call). Follow that pattern for anything with the same shape — an optional extra or a network dependency shouldn't be paid for by callers who don't use the feature.

`benchmarks/` validates DEConcord's methods against real DESeq2/edgeR output (`airway`, `pasilla` datasets) and is where the README's validation numbers come from — see `benchmarks/README.md`.

## Versioning and changelog discipline

Every real (user-facing) feature or fix gets, together in the same change:

1. A version bump, applied in lockstep to **both** `pyproject.toml`'s `version` and `CITATION.cff`'s `version`/`date-released`.
2. A `CHANGELOG.md` entry under the new version heading — user-facing, in Keep-a-Changelog-ish style (`### Added`/`### Changed`/etc.), and *not* the format the DEVLOG entry uses; it's a summary, not the reasoning.
3. A `DEVLOG.md` entry — the dated, narrative account of what changed, why, what got rejected along the way, and the test count before/after. Write it in the existing voice: first-person reasoning, explicit "Decisions rejected along the way" subsection, an explicit "Version bump: X to Y" line at the end. Read a couple of recent entries before writing one — the voice and structure matter as much as the content.

Pre-1.0, SemVer allows breaking changes in a minor bump, but "allowed" isn't a reason to prefer breaking an existing interface — see the Aug 17 DEVLOG entry's reasoning about not restructuring the CLI's argument shape just because SemVer would permit it.

## Restraint

- Don't refactor working code for aesthetics alone. If a change isn't required to implement the requested feature or fix the requested bug, leave it.
- Don't expand scope beyond what the README's Roadmap section already lists as current/next. If an idea seems useful but isn't on the roadmap (a new omics type, a general ML feature, a new CLI surface not already flagged as under consideration), flag it to the user instead of building it — see `CONTRIBUTING.md`'s "Scope" section for the kind of thing that gets rejected regardless of implementation quality.
- Prefer raising early and loudly over a plausible-looking wrong answer; error messages should name the specific problem (which sample, which column, which value), not a bare exception type — see `src/deconcord/io/counts.py` for the expected style.
- Docstrings on public functions (what it does, parameters, returns, raises) explain *why* a non-obvious design choice was made, not just *what* the code does.
