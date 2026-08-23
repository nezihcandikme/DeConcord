# DEConcord

[![Tests](https://github.com/nezihcandikme/DeConcord/actions/workflows/tests.yml/badge.svg)](https://github.com/nezihcandikme/DeConcord/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Robustness and concordance analysis for RNA-seq differential expression.

Most RNA-seq workflows report the significant genes or pathways from one analysis configuration. DEConcord checks whether those conclusions stay stable when reasonable choices change: a different DE tool, a different significance threshold, a different resampling of the same data.

DEConcord isn't a DESeq2/edgeR replacement, and running its own differential expression is a small, secondary part of it. It takes DE result tables (from DESeq2, edgeR, or anything with a gene ID, a log fold change, and a p-value) and reports how much their conclusions actually agree, and where they don't.

**Status**: early, pre-1.0 (v0.19.0), under active development. Method concordance, threshold sensitivity, pathway stability, and resampling stability (including per-gene direction and rank stability) are implemented; method concordance is checked against real data (see [Validation](#validation)).

## Quick start

```bash
git clone https://github.com/nezihcandikme/DeConcord.git
cd DeConcord
pip install -e .
```

Compare two DE result tables. Here's real DESeq2 and edgeR output on the same count matrix:

```python
import pandas as pd
from deconcord.concordance.methods import compute_de_concordance

deseq2 = pd.read_csv("deseq2_results.csv", index_col="gene_id")
edger = pd.read_csv("edger_results.csv", index_col="gene_id")

result = compute_de_concordance(
    deseq2, edger, name_a="DESeq2", name_b="edgeR",
    lfc_col_a="log2FoldChange", pvalue_col_a="padj",
    lfc_col_b="logFC", pvalue_col_b="FDR",
)

result["summary"]["jaccard_index"]          # significant-gene set overlap
result["summary"]["directional_agreement"]  # among genes sig. in both, % same direction
result["discordant_genes"]                  # significant in both, but disagree on direction
```

Or do the same comparison from the command line, no Python required:

```bash
deconcord concordance deseq2_results.csv edger_results.csv \
    --name-a DESeq2 --name-b edgeR \
    --lfc-col-a log2FoldChange --pvalue-col-a padj \
    --lfc-col-b logFC --pvalue-col-b FDR \
    --out concordance_output
```

Writes `summary.json`, `merged.csv` (every compared gene, both tables' values side by side), and separate CSVs for the concordant, discordant, and method-specific gene lists.

The underlying pipeline is still there and still real: validation, QC, normalization, DE, enrichment. Run it with `deconcord counts.csv --group1 ... --group2 ...` on the CLI, or call `deconcord.pipeline.run_analysis` as a library function.

Supported input is a CSV count matrix: genes as rows, samples as columns, non-negative integer values. The CLI and `run_analysis` both validate this up front. If something's wrong (wrong orientation, duplicate gene IDs, missing values), they name the specific problem instead of failing deep inside a later step.

A CLI run writes:

```
deconcord_output/
├── tables/differential_expression.csv, qc.csv, ...
├── figures/volcano.png, pca.png, ...
└── run_metadata.json   # version, parameters, timestamp, for reproducing the run
```

<img src="docs/assets/volcano-demo.png" alt="Volcano plot from a real DEConcord CLI run: log fold change on the x-axis, -log10 adjusted p-value on the y-axis, significant genes highlighted." width="500" />

*Volcano plot from an actual `deconcord` CLI run on a synthetic dataset. Every point is a gene that was really tested, not a mockup.*

For the full workflow in one runnable script, see [`examples/quickstart.py`](examples/quickstart.py). It generates data, loads it, validates it, runs two DE methods, checks their concordance, and plots the result. Run it with `python examples/quickstart.py`. No network or R required.

## Why

I'm a high-school student learning statistics and computational biology. This project started as a general bulk RNA-seq pipeline: validation, QC, normalization, a from-scratch differential expression implementation. That was useful for learning the mechanics, but a narrower question turned out to be more interesting, and more realistic about what a small project can actually contribute.

Mature platforms like [OmicVerse](https://github.com/Starlitnightly/omicverse) already build general DE tooling well, with a published paper and years of development behind them, so trying to out-build that isn't the point here. What seemed more worth checking was how much the conclusions from existing tools actually hold up once you stress-test them, which is more a matter of being careful than building something new.

Getting the code to run was step one. Making sure the numbers meant what I thought they meant took a lot longer, and is really what this project is about. The dated account of what changed, why, and what didn't work, including the reasoning behind this rename, is in [`DEVLOG.md`](DEVLOG.md).

## What it does

The intended workflow, once every stage exists:

```
DE result tables (DESeq2, edgeR, ...)
        ↓
method concordance          ← built (compute_de_concordance)
        ↓
threshold sensitivity       ← built (compute_threshold_sensitivity)
        ↓
pathway enrichment + pathway stability   ← built (compute_pathway_stability)
        ↓
resampling stability                     ← built (compute_resampling_stability)
        ↓
robustness assessment                    ← not built yet
        ↓
interpretable report                     ← not built yet
```

**Built today**:

- `deconcord.concordance.methods.compute_de_concordance`. Given two DE result tables for the same comparison, it computes the overlap and Jaccard index of their significant-gene calls, the Pearson and Spearman correlation of log fold change across every gene both tables tested, directional agreement (among genes significant in both, the fraction that agree on the sign of the effect), and explicit concordant, discordant, and method-specific gene lists. See `benchmarks/method_concordance.py` for a real run against real DESeq2/edgeR output.
- `deconcord.concordance.threshold_sensitivity.compute_threshold_sensitivity`. Sweeps `compute_de_concordance` across a range of significance thresholds (`alpha` values) between the same two tables, so you can see whether a concordance conclusion holds up across a reasonable range of cutoffs or is an artifact of the particular threshold checked. Reports both an alpha-by-alpha summary and, per gene, what fraction of the swept thresholds call it significant in each table.
- `deconcord.concordance.pathway_stability.compute_pathway_stability`. The same overlap-and-Jaccard idea, applied to two pathway enrichment result tables instead of two DE tables. Answers "does this pathway stay enriched regardless of which DE method or threshold produced the significant-gene list," rather than "does this gene stay significant."
- `deconcord.concordance.resampling_stability.compute_resampling_stability`. Reruns DE on perturbed sample sets (repeated random subsamples, or exhaustive leave-one-out) and reports, per gene, what fraction of reruns still call it significant. Answers "does this gene stay significant regardless of which samples happened to be in the dataset," the sample-level analog of threshold sensitivity. Also reports `direction_stability` (does the gene keep the same effect direction across reruns) and `rank_stability` (does it keep roughly the same position relative to every other gene), computed from the same reruns at no extra cost.

Everything else below is infrastructure rather than the core contribution — what's left of the original bulk RNA-seq pipeline. It's kept because it's useful (it's how the `benchmarks/` DE result tables get generated, and it's a source of a third method to check for concordance), not because DEConcord is trying to be a general-purpose RNA-seq toolkit:

- **Validation, QC, normalization**: count matrix checks, library size and outlier detection, CPM plus log2 scaling.
- **Differential expression**: Welch's t-test, an empirical-Bayes moderated t-test, or a covariate-adjusted linear model. This is DEConcord's own from-scratch DE implementation. It's useful as one more method to check for concordance against DESeq2/edgeR, not as a competing product.
- **Gene annotation, pathway enrichment (local and live g:Profiler), GEO acquisition, plots** (volcano, PCA, enrichment dot plot). Unchanged from before, all still real and tested. See `deconcord.pipeline.run_analysis` for the full pipeline these support.
- **Optional AI summary**: narrates a result table, doesn't verify it, off by default.

## What DEConcord does not build

This is deliberate: depth on one question rather than breadth across many. DEConcord isn't headed toward single-cell analysis, spatial transcriptomics, proteomics, metabolomics, molecular docking or dynamics, protein structure prediction, general-purpose ML tooling, or integrations added just because other omics packages have them.

If a proposed feature doesn't help answer how robust a differential expression conclusion is, it probably doesn't belong here.

## Validation

**Method concordance** is the actual new core question. Here are the two real datasets checked so far: [`airway`](https://bioconductor.org/packages/release/data/experiment/html/airway.html) (human) and [`pasilla`](https://bioconductor.org/packages/release/data/experiment/html/pasilla.html) (*Drosophila*), each pair of tools run independently with its own default settings. Three tools checked pairwise: DESeq2, edgeR, and limma-voom.

| dataset | pair | genes compared | Jaccard (sig. overlap) | log2FC Pearson r | directional agreement | discordant genes |
|---|---|---:|---:|---:|---:|---:|
| airway | DESeq2 vs edgeR | 15,896 | 0.758 | 0.9995 | 1.000 | 0 |
| airway | DESeq2 vs limma-voom | 15,896 | 0.718 | 0.9901 | 1.000 | 0 |
| airway | edgeR vs limma-voom | 15,926 | 0.913 | 0.9897 | 1.000 | 0 |
| pasilla | DESeq2 vs edgeR | 7,919 | 0.785 | 0.9988 | 1.000 | 0 |
| pasilla | DESeq2 vs limma-voom | 7,919 | 0.766 | 0.9876 | 1.000 | 0 |
| pasilla | edgeR vs limma-voom | 7,919 | 0.951 | 0.9891 | 1.000 | 0 |

Three established tools, run independently, agree strongly with each other in every pairwise combination. Correlation is high across all three pairs, and zero genes get called significant by both members of any pair while disagreeing on direction. The disagreement that exists is almost entirely about which marginal genes clear each tool's own significance threshold, not about direction or magnitude for genes multiple tools agree are real. This is a real finding on two datasets, not a general law about these three tools. See `benchmarks/method_concordance.py` and `benchmarks/README.md` for the full methodology.

**Stress test, all three regimes (unbalanced groups, batch effects, low counts)**: the same three-tool concordance check, pushed outside the two clean, balanced datasets above, all on real Bottomly subsets. Regime 1 (group sizes shrinking from a balanced 10-vs-11 to a severe 2-vs-11) produced the largest effect: Jaccard 0.930 to 0.704 for DESeq2 vs edgeR, 0.880 to 0.602 for DESeq2 vs limma-voom. Regime 2 (a real but only mildly confounded technical covariate, per-sample library size, modeled or not) moved Jaccard by only about 0.005-0.015 per pair, in both directions. Regime 3 (sequencing depth thinned to 25% and 5% of real reads) degraded Jaccard mildly (0.930 to 0.890 for DESeq2 vs edgeR) and left log fold change correlation essentially flat to slightly improved. Across all three regimes, 24 tool-pair comparisons total, one number never moved: directional agreement among jointly-significant genes stayed at exactly 1.000, zero discordant genes, every time. See `benchmarks/README.md`'s stress-test section for the full tables and interpretation.

**DEConcord's own DE methods vs DESeq2/edgeR** is a check on the infrastructure, not the project's main claim anymore.

| dataset | method | precision (DESeq2 / edgeR) | recall (DESeq2 / edgeR) | log2FC Pearson r (DESeq2 / edgeR) |
|---|---|---:|---:|---:|
| airway | welch | 1.0 / 1.0 | 0.075 / 0.098 | 0.938 / 0.947 |
| airway | moderated | 1.0 / 1.0 | 0.165 / 0.217 | 0.938 / 0.947 |
| pasilla | welch | 1.0 / 1.0 | 0.085 / 0.104 | 0.974 / 0.975 |
| pasilla | moderated | 1.0 / 1.0 | 0.174 / 0.215 | 0.974 / 0.975 |

Precision is the fraction of genes DEConcord's own method calls significant that DESeq2/edgeR also call significant. Recall is the reverse. No false-positive calls relative to the reference DESeq2/edgeR calls were observed in these benchmarks. That's not the same claim as a universal zero false-positive rate.

## Limitations

- `compute_de_concordance` itself is still pairwise, one fixed threshold at a time (default `alpha=0.05`). `compute_threshold_sensitivity` covers checking a range of thresholds, but it's a separate function you call in addition, not something the base comparison does automatically.
- `compute_resampling_stability` offers subsampling and leave-one-out, not bootstrap. Supporting a true bootstrap (sampling with replacement) would need reworking how the DE functions index samples, since they currently refuse a duplicate sample name in a group. See `METHODOLOGY.md`.
- The robustness and concordance metrics so far are standard, well-understood ones: Jaccard, Pearson/Spearman correlation, directional agreement. No new statistical method has been invented, deliberately. See the roadmap.
- Concordance findings so far come from two datasets (`airway`, `pasilla`), both standard two-group designs. Whether DESeq2/edgeR concordance holds up on unbalanced groups, batch effects, or more complex designs hasn't been checked.
- The underlying DE infrastructure (validation, QC, DE, enrichment) still only supports two-group condition comparisons, with optional covariate adjustment. No paired samples, no more than two condition levels.
- Pathway stability only compares presence/absence of significance between two enrichment tables. It has no way to detect that the two tables were built on different gene-set collections or background universes, which would make the comparison meaningless even though the function runs without error. See `METHODOLOGY.md`.

## Research (experimental, not part of the package)

`research/ml/` is a separate, exploratory track testing whether DEConcord's own robustness metrics carry predictive information beyond ordinary DE statistics. It's not merged into `src/deconcord/`, not installed by `pip install -e .`, and not marketed as an "AI" feature, see `CONTRIBUTING.md`'s Scope section for why that boundary is deliberate.

This phase is now closed, tested against four datasets under one unmodified, frozen experiment design (borderline window, conventional DE features versus the same plus `rank_stability`, L1-penalized logistic regression with regularization chosen by cross-validation on the training dataset alone, then locked and scored once on held-out data). The result is dataset-dependent, not universal: on `airway`/`pasilla`, adding `rank_stability` moved PR-AUC from 0.50 to 0.65 in a `pasilla -> airway` locked evaluation, a gain that held up under residualization against ordinary DE strength, L1 regularization checks, a window-sensitivity sweep, and bootstrap uncertainty intervals. On a third dataset, Bottomly (a valid, evaluable external test, 1,997 borderline genes), the same frozen design gave the opposite result: `rank_stability` moved `pasilla -> bottomly` PR-AUC from 0.9491 down to 0.9230. A fourth dataset, zebrafish, was non-evaluable under this endpoint (edgeR found zero significant genes on it at all) and counts as neither positive nor negative evidence.

DeConcord rank stability showed incremental predictive value for borderline cross-method significance agreement in airway/pasilla transfer experiments, but the benefit did not generalize universally: it decreased performance on Bottomly, while zebrafish was non-evaluable under the frozen endpoint. The current evidence therefore supports dataset-dependent, rather than universal, predictive value.

Read plainly, not overstated: this predicts whether two DE *methods* agree on a borderline gene, not biological replication. See [`research/ml/FINDINGS.md`](research/ml/FINDINGS.md) for the full writeup, every dataset's numbers, and every caveat, and [`research/ml/INVENTORY.md`](research/ml/INVENTORY.md) for exactly what data and features this used. No further experiments are planned against this frozen design; a resumed track would need new data or a different question.

## Reproducibility

Every CLI run writes a `run_metadata.json`: DEConcord version, Python version, core dependency versions, the exact command, an input summary (gene and sample counts, group membership), and every parameter used. It's meant to make a generated analysis reproducible without having to remember what was installed or passed in at the time. Check it into version control alongside your results for a real audit trail.

## Roadmap

**Current** (built): method concordance between any two DE result tables — validated pairwise across DESeq2, edgeR, and limma-voom (overlap, Jaccard, directional and effect-size agreement, concordant and discordant genes). Threshold sensitivity, checking which findings survive small, reasonable changes to the significance cutoff. Pathway stability, checking whether an enriched pathway stays enriched across methods and settings. Resampling stability, checking whether a gene's significance call survives a different sample set through subsampling or leave-one-out. A `deconcord concordance` CLI subcommand, comparing two existing DE result tables without running the pipeline first. The underlying RNA-seq pipeline (QC, DE, enrichment) that generates the tables to compare. The stress-test benchmark suite: a third DE tool (limma-voom) and all three harder data regimes (unbalanced groups, batch effects, low counts) checked against real Bottomly subsets, real results above and in `benchmarks/README.md`.

**Next**: not yet decided. The stress-test suite's original scope (third tool, three harder regimes) is complete; what comes after it hasn't been picked yet.

**Later**: a scientifically defensible robustness or concordance summary statistic, if one turns out to be justified once the above exists, rather than invented ahead of it. CLI subcommands for threshold sensitivity and pathway stability too, once the single-function `concordance` subcommand has been used enough to know if the same shape actually fits them. Richer interpretation and reporting.

No dates here, since this project moves at whatever pace it moves at, and a roadmap with fake deadlines would be more misleading than one without.

## Documentation, citation, and contributing

- **Methodology**: what "concordance," "directional agreement," and the other terms above actually mean, how they're computed, and their limitations. See [`METHODOLOGY.md`](METHODOLOGY.md).
- **Development history**: `DEVLOG.md` has the day-by-day account of what changed, what broke, what got rejected, and why. Git commit messages say *what* changed; that file is for *why*. `CHANGELOG.md` has the user-facing summary per version.
- **Citation**: see [`CITATION.cff`](CITATION.cff), also usable via GitHub's "Cite this repository" button. It's a software citation only. There's no associated peer-reviewed publication.
- **Contributing**: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for environment setup, running tests, and code style. Bug reports and feature requests use the templates under `.github/ISSUE_TEMPLATE/`.
