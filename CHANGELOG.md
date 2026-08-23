# Changelog

All notable changes to DEConcord are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). Versioning follows
[Semantic Versioning](https://semver.org/). Pre-1.0, breaking changes can
happen in a minor version bump.

This file is user-facing: what changed and what it means for you. For the
reasoning behind each decision, including rejected alternatives, see
[`DEVLOG.md`](DEVLOG.md).

## [0.20.0] — 2026-08-23

### Fixed
- `check_nonnegative_counts` no longer flags a missing value (`NaN`) as
  also being a negative count. A single `NaN` used to trip both checks at
  once (`nan >= 0` evaluates to `False`), producing a confusing combined
  "missing values" + "counts must be non-negative" error for what was
  really one problem. `NaN` cells are now excluded from the non-negativity
  comparison, matching how `check_all_finite` already excludes them from
  the infinite-value check.
- `compute_resampling_stability`'s summary no longer emits a spurious
  `RuntimeWarning: Mean of empty slice` when every iteration (baseline and
  every resample) finds zero significant genes — a real outcome with a
  strict `alpha` or a genuinely null dataset. `mean_jaccard_to_baseline`
  still correctly comes back as `NaN` in that case, just without the
  warning noise.

### Changed
- Core dependencies (`pandas`, `numpy`, `scipy`, `statsmodels`,
  `matplotlib`, `scikit-learn`) now declare minimum versions instead of
  being unpinned. An unpinned dependency could previously resolve to a
  years-old, silently incompatible release in an old or frozen
  environment instead of failing at install time.
- The package now ships a `py.typed` marker (PEP 561): its type hints are
  meant to be checked by a downstream caller's own type checker (mypy,
  pyright), not just read as documentation. Every module now passes
  `mypy` cleanly, and CI enforces that going forward.

### Added
- `.[dev]` now also installs `pytest-cov` and `mypy`.

This release is a testing, typing, and packaging hardening pass with no
new user-facing features — see [`DEVLOG.md`](DEVLOG.md) for the full
reasoning and what was deliberately left out.

## [0.19.0] — 2026-08-23

### Changed
- `validate_counts`/`load_count_matrix` no longer require whole-number
  counts. Salmon/kallisto/RSEM-style "estimated counts" (legitimately
  fractional, since multi-mapping reads get split across the transcripts
  they map to) are now accepted alongside featureCounts/htseq-count's
  integer output. Non-negative and finite is still required; a
  non-numeric column and an infinite value are both still rejected, now
  with their own specific error types (`NonNumericCountsError`,
  `NonFiniteCountsError`). `check_all_integer` is kept as a standalone,
  stricter check for a caller who wants to require whole numbers.
- `load_count_matrix` auto-detects comma vs. tab delimiter by default
  (new `sniff_delimiter`, also a `sep` parameter to override it). A
  tab-separated file — the shape most raw bulk RNA-seq quantification
  tools (featureCounts, htseq-count) actually export — used to silently
  collapse into one unparsed column instead of failing or loading
  correctly.

### Added
- `validate_counts` now rejects an empty (zero-row or zero-column) count
  matrix and a duplicated sample column name, both with a specific error
  naming the problem. Previously both passed validation silently (every
  check trivially passes on an empty axis; only the gene-ID side of the
  uniqueness check existed, not the sample side).
- `compute_de_concordance`/`compute_pathway_stability` now emit a
  `UserWarning` when the matched gene/pathway count is well below either
  input table's own size — usually a sign of an ID-format mismatch
  between the two tables (an Ensembl ID version suffix present in one
  table and not the other, an Ensembl ID vs. a gene symbol, a naming
  convention difference between two pathway enrichment sources) rather
  than a real biological difference in what was tested. A total mismatch
  (zero overlap) already raised a clear error; this covers the partial
  case, which previously just looked like a slightly lower Jaccard index
  with no indication anything was off.
- `plot_volcano` now validates its input (required columns, non-empty)
  the same way `plot_pca`/`plot_pathway_enrichment` already did, instead
  of raising a bare `KeyError` on a missing column.

## [0.18.0] — 2026-08-19

### Added
- `deconcord.concordance.resampling_stability.compute_resampling_stability`
  now reports two more per-gene robustness metrics, computed from the
  same reruns it already performs (no additional DE reruns):
  - `direction_stability`: fraction of reruns whose log fold change has
    the same sign as the baseline's, restricted to reruns where that
    gene produced a finite log fold change. Pure sign comparison, no
    significance criterion, so it's distinct from `frac_significant`.
  - `rank_stability`: a per-gene score (not one dataset-wide correlation)
    measuring how consistently a gene holds the same relative position,
    by signed log fold change, among every other gene across reruns
    versus the baseline. `1.0` = same relative position every time.
  - Both added to the `gene_stability` DataFrame, alongside two new
    summary keys, `mean_direction_stability` and `mean_rank_stability`,
    averaged over baseline-significant genes only, matching the existing
    `baseline_replication_rate` convention.
  - `is_stable` and its inputs are unchanged.

### Documentation
- `METHODOLOGY.md`'s resampling stability section now documents the
  exact formulas, denominators, and edge-case (NaN/insufficient-gene)
  handling for both new metrics.

## [0.17.0] — 2026-08-19

### Added
- `benchmarks/run_limma_voom.R`: runs limma-voom (standard `filterByExpr`
  / TMM / `voom` / `lmFit` / `eBayes` workflow) on the same `airway` and
  `pasilla` datasets and group comparisons the existing DESeq2/edgeR
  benchmark scripts use, writing `limma_voom_results.csv` /
  `pasilla_limma_voom_results.csv` in the same per-gene CSV shape
  (`gene_id`, tool-native `logFC`, a normalized `pvalue` column, and the
  tool's own adjusted-p-value column name), so the output is a drop-in
  third input to `compute_de_concordance`.
- `benchmarks/method_concordance.py` now runs all three pairwise
  comparisons among DESeq2, edgeR, and limma-voom (previously DESeq2 vs
  edgeR only), on both datasets. The DESeq2-vs-edgeR numbers are
  unchanged (verified byte-identical to the previous version).

### Changed
- README's validation table and `docs/index.html`'s concordance section
  now report all three pairwise comparisons instead of DESeq2 vs edgeR
  alone.

## [0.16.0] — 2026-08-17

### Added
- `deconcord concordance results_a.csv results_b.csv` CLI subcommand.
  Wraps `compute_de_concordance` directly: point it at two existing DE
  result table CSVs and it writes `summary.json`, `merged.csv`, and
  separate CSVs for the concordant, discordant, and method-specific
  gene lists, without running the pipeline first. Configurable gene ID,
  log fold change, and p-value column names, same as the Python
  function.
- Writes a `run_metadata.json` for the concordance subcommand, matching
  the existing pipeline CLI's reproducibility record.

### Changed
- The original `deconcord counts.csv --group1 ... --group2 ...`
  invocation is unchanged. `concordance` is checked for as the first
  argument; anything else falls through to the original pipeline
  behavior, so no existing script or command breaks.

## [0.15.0] — 2026-08-16

### Added
- `deconcord.concordance.resampling_stability.compute_resampling_stability`.
  Reruns differential expression on perturbed sample sets (random
  subsamples or exhaustive leave-one-out) and reports, per gene, what
  fraction of reruns still call it significant, compared against the
  full-data baseline call. Returns a summary (baseline replication rate,
  mean Jaccard to the baseline significant-gene set) plus a per-gene
  stability table and explicit stable/sensitive gene lists.
- Exported from the top-level `deconcord` package
  (`dc.compute_resampling_stability`).
- `METHODOLOGY.md`: full documentation for resampling stability,
  replacing the previous "not implemented yet" placeholder.

### Changed
- Bootstrap resampling (sampling with replacement) is deliberately not
  offered, since it would need reworking how the DE functions index
  samples. Documented as a scope decision, not an oversight.

## [0.14.0] — 2026-08-15

### Added
- `deconcord.concordance.threshold_sensitivity.compute_threshold_sensitivity`.
  Sweeps `compute_de_concordance` across a range of significance
  thresholds (`alpha`) between two DE result tables. Returns an
  alpha-by-alpha summary table plus, per gene, what fraction of the
  swept thresholds call it significant in each table, so a threshold-
  sensitive finding (significant only at some cutoffs) is distinguished
  from a threshold-stable one.
- `deconcord.concordance.pathway_stability.compute_pathway_stability`.
  Compares two pathway enrichment result tables the same way method
  concordance compares two DE tables: significant-pathway overlap and
  Jaccard index, plus explicit stable/method-specific pathway lists.
- Both functions are exported from the top-level `deconcord` package
  (`dc.compute_threshold_sensitivity`, `dc.compute_pathway_stability`).
- `METHODOLOGY.md`: full documentation for both new functions (what they
  compute, how to interpret the output, limitations), replacing the
  previous "not implemented yet" placeholder for these two.

## [0.13.2] — 2026-08-14

### Changed
- Rewrote the prose in `README.md`, `docs/index.html`, `CHANGELOG.md`,
  and `DEVLOG.md` into shorter, plainer sentences and removed em-dashes
  throughout. No factual content changed: same numbers, same decisions,
  same test-count checkpoints.

## [0.13.1] — 2026-08-14

### Fixed
- The GitHub repository was renamed `BioInsight` to `DeConcord`, but every
  URL in `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`,
  `pyproject.toml`, and `docs/index.html` still pointed at the old
  `BioInsight` path, including the README's own `git clone` command.
  Updated all of them. GitHub's redirect from the old name masked this
  for anyone clicking through, but not for anyone reading the raw text.
  `DEVLOG.md`'s historical entries are unaffected; they describe what
  was true when written. `pyproject.toml`'s `Homepage` now also points
  at the live GitHub Pages site instead of the repository itself.

## [0.13.0] — 2026-08-14

Professionalization pass. No new analytical capability. The focus was
making the repository's structure, docs, tests, and packaging match a
project that expects outside users. See `DEVLOG.md` for the full
reasoning.

### Changed
- **Breaking: CLI output layout.** `deconcord_output/` now organizes
  results into `tables/` (CSVs) and `figures/` (PNGs) instead of a flat
  directory, plus a new `run_metadata.json`. Existing scripts that read
  `deconcord_output/differential_expression.csv` directly need to read
  `deconcord_output/tables/differential_expression.csv` instead. Same
  rename for `qc.csv`, `pathway_enrichment.csv`,
  `live_pathway_enrichment.csv`, and the `.png` figures.
- **Package layout**: source moved to `src/deconcord/` (was `deconcord/`
  at the repo root). A `pip install -e .` from a clean checkout is
  unaffected. This only matters if you had tooling pointed at the old
  path.
- `anthropic`/`python-dotenv` are now an optional `[ai]` extra
  (`pip install deconcord[ai]`) instead of core dependencies. Only
  needed for the opt-in `explain_results=True` AI summary. Calling it
  without the extra now raises a clear `ImportError` naming the fix,
  instead of a bare `ModuleNotFoundError`.
- `tests/test_differential_expression.py` (554 lines) split into
  `test_differential_expression_welch.py`, `_moderated.py`, and
  `_covariates.py`. No test content changed, only organization.

### Added
- `deconcord/__init__.py` now exposes a curated public API
  (`import deconcord as dc`; `dc.run_analysis`, `dc.compute_de_concordance`,
  etc.) and `deconcord.__version__`.
- `deconcord --version` CLI flag.
- CLI writes `run_metadata.json` alongside its normal output: DEConcord
  version, Python version, core dependency versions, the exact command,
  an input summary, and the parameters used, for reproducing a run.
- `LICENSE` (MIT), `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, and
  GitHub issue/PR templates.
- `METHODOLOGY.md`: what each concordance metric computes, its source
  if it's a standard statistic, and its limitations.
- `examples/quickstart.py`: a single runnable script covering the full
  current workflow. Synthetic data, load and validate, two DE methods,
  concordance, plot. No network or R required.
- Test coverage for previously untested edge cases: duplicate gene IDs
  and infinite values in `validate_counts`.

## [0.12.0] — 2026-08-14

### Changed
- **Project renamed from OmicForge to DEConcord**, and scope narrowed
  from "general bulk RNA-seq pipeline" to "robustness/concordance
  analysis for RNA-seq differential expression." Package import path is
  now `deconcord` (was `omicforge`); CLI command is now `deconcord`
  (was `omicforge`). See the README's "Why" section and the Aug 14
  DEVLOG entry for the full reasoning.

### Added
- `deconcord.concordance.methods.compute_de_concordance`: the project's
  new core function. Given two DE result tables (any tool, configurable
  column names), it computes significant-gene overlap, Jaccard index,
  Pearson/Spearman log fold change correlation, directional agreement,
  and explicit concordant, discordant, and method-specific gene lists.
- `benchmarks/method_concordance.py`: runs the new function against
  real, committed DESeq2/edgeR result tables (airway, pasilla).

## [0.11.0] — 2026-08-13

### Added
- Covariate-adjusted differential expression:
  `run_differential_expression_with_covariates`, a per-gene linear model
  that tests group_1 vs group_2 while holding one or more covariates
  (batch, sex, etc.) fixed. Wired into `run_analysis`
  (`metadata`/`covariate_cols`/`moderated_covariates`) and the CLI
  (`--metadata`/`--covariates`/`--no-moderated-covariates`).

## [0.10.0] — 2026-08-13

### Added
- `plot_pathway_enrichment`: a dot plot (term vs. -log10 p-value, dot
  size and color tied to gene overlap) for both local and live pathway
  enrichment results. Generated automatically by `run_analysis` and the
  CLI when enrichment was run.

## [0.9.1] — 2026-08-13

### Changed
- Finished the OmicForge rename at the code level: package import path,
  CLI command, output directory name. The previous release only renamed
  user-facing text.
- README rewritten. Real, verified quick-start examples moved near the
  top, a benchmark results table added, an explicit Limitations section
  added, narrative tone reduced (about 30% shorter).

### Removed
- `requirements.txt`, unmaintained since v0.2 and duplicated by
  `pyproject.toml`.

## [0.9.0] — 2026-08-13

### Added
- Second independent benchmark dataset (`pasilla`, *Drosophila*)
  alongside the original `airway` (human) benchmark, to check whether
  the precision/recall pattern found on one dataset held on another. It
  did.
- Live pathway enrichment against the g:Profiler API
  (`run_gprofiler_enrichment`): GO, KEGG, Reactome, WikiPathways.
  Opt-in, requires outbound internet access.
- GEO series data acquisition (`deconcord.io.geo`): list and download a
  GEO series' supplementary files, and parse its sample metadata from a
  `GSE` accession.

## [0.8.0] — 2026-08-12

### Added
- Gene ID to symbol annotation (`load_gene_annotation`,
  `annotate_de_results`) from a local two-column CSV. Wired into
  `run_analysis` (`gene_annotation`) and the CLI (`--annotation`).

## [0.7.0] — 2026-08-12

### Added
- Empirical-Bayes moderated t-test (`compute_moderated_pvalues`),
  selectable via `method="moderated"`. Shrinks each gene's variance
  estimate toward a value fit from every other gene's variance, the
  same idea behind limma's `eBayes`. More statistical power than the
  default per-gene Welch's t-test at the same precision.

## [0.6.0] — 2026-08-12

### Added
- Command-line interface (`deconcord counts.csv --group1 ... --group2
  ...`) as a real console script, installed via `pip install -e .`.

## [0.5.0] — 2026-08-11

### Added
- `load_gmt`: a parser for the standard tab-separated GMT gene-set file
  format, so pathway enrichment can be run against a real MSigDB or
  Enrichr download instead of a hand-built Python dict.

## [0.4.0] — 2026-08-11

### Added
- Pre-DE low-count gene filtering (`filter_low_count_genes`, opt-in via
  `min_count`). Drops genes that never had a realistic chance of
  reaching significance, improving multiple-testing correction for
  every other gene.

### Changed
- `alpha` and `lfc_threshold` (previously hardcoded) are now real,
  validated parameters on `run_differential_expression` and
  `run_analysis`.

## [0.3.x] — 2026-08-09 to 2026-08-11

### Fixed
- A correctness pass across the whole pipeline (v0.3.1 to v0.3.3):
  several bugs found by re-reading the code closely while rewriting
  docstrings, including NaN handling in constant-expression genes and
  multiple-testing correction.

## [0.3.0] — 2026-08-09

### Added
- Pathway enrichment: hypergeometric test against an explicit
  background gene universe (`run_pathway_enrichment_analysis`).

## [0.2.0] — 2026-08-08

### Added
- Visualization: volcano plot and PCA plot.

## [0.1.0] — 2026-08-07

### Added
- Differential expression: log fold change, per-gene Welch's t-test,
  Benjamini-Hochberg correction (`run_differential_expression`).

## [0.0.1] to [0.0.3] — 2026-08-06 to 2026-08-07

### Added
- Count matrix validation (non-negative integers, unique gene IDs, no
  missing values).
- Sample QC (library size, genes detected, MAD-based outlier
  detection).
- Normalization (CPM, log2 transform).
