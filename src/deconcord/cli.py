"""
Command-line entry point for DEConcord.

Two things live here:

- The default invocation, ``deconcord counts.csv --group1 ... --group2
  ...``, wraps ``deconcord.pipeline.run_analysis``: point it at a count
  matrix CSV, name your two groups, and it writes the differential
  expression table (and, if asked, plots and a pathway enrichment table)
  to an output directory. This is the original CLI shape and it is
  unchanged.
- ``deconcord concordance results_a.csv results_b.csv`` wraps
  ``deconcord.concordance.methods.compute_de_concordance``: point it at
  two existing DE result tables (from DESeq2, edgeR, DEConcord's own
  output, anything with a gene ID, a log fold change, and a p-value) and
  it writes the comparison to an output directory, without running any
  pipeline stage first.

There's no subcommand named "run", on purpose. Adding one would mean every
existing invocation of the form ``deconcord counts.csv ...`` breaks and
needs a new "run" prefix, which is exactly the kind of gratuitous breaking
change this project tries to avoid even though pre-1.0 SemVer would allow
it. Instead, ``concordance`` is the one explicit subcommand name, checked
for as the first argument; anything else falls through to the original,
untouched pipeline behavior. The one real cost of this: a count matrix
literally named ``concordance`` (or ``concordance.csv`` passed without an
extension check) would be misread as the subcommand instead of a file
path. Rare enough in practice not to be worth a more complex parser for.

Both paths are deliberately thin wrappers. All the actual logic --
validation, QC, filtering, normalization, DE, enrichment, concordance --
lives in the library modules; this module's only job is argument parsing,
wiring, and turning exceptions that are expected to happen on bad input (a
missing file, an unknown sample name, a malformed GMT file, a missing
column in a results table) into a clean error message instead of a
traceback.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from deconcord.concordance.methods import compute_de_concordance
from deconcord.io.counts import CountMatrixError, sniff_delimiter
from deconcord.pathway_analysis.gmt import load_gmt
from deconcord.pipeline import run_analysis

# Packages a run_metadata.json should record the installed version of --
# every core (non-optional) runtime dependency, so a generated analysis
# can be reproduced without guessing what was installed when it ran.
_TRACKED_PACKAGES = ["deconcord", "pandas", "numpy", "scipy", "statsmodels", "matplotlib", "scikit-learn"]


def _installed_versions() -> dict[str, str]:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _parse_sample_list(raw: str) -> list[str]:
    samples = [s.strip() for s in raw.split(",") if s.strip()]
    if not samples:
        raise ValueError("Expected a comma-separated list of sample names, got an empty one.")
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deconcord",
        description="Run the DEConcord RNA-seq pipeline on a count matrix CSV.",
    )
    parser.add_argument("counts", help="Path to a count matrix CSV (genes as rows, samples as columns).")
    parser.add_argument("--group1", required=True, help="Comma-separated sample names for the first comparison group.")
    parser.add_argument("--group2", required=True, help="Comma-separated sample names for the second comparison group.")
    parser.add_argument("--gmt", help="Path to a .gmt gene-set file to run pathway enrichment against.")
    parser.add_argument(
        "--background",
        help="Comma-separated background gene list for enrichment. Defaults to every gene in the "
             "(filtered, if --min-count is set) count matrix if --gmt is given without this.",
    )
    parser.add_argument("--min-count", type=int, default=None, help="Minimum raw read count for a gene to pass pre-DE filtering. Off by default.")
    parser.add_argument("--min-samples", type=int, default=2, help="Number of samples that must clear --min-count. Default 2. Ignored without --min-count.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Adjusted p-value cutoff for significance. Default 0.05.")
    parser.add_argument("--lfc-threshold", type=float, default=1.0, help="Absolute log fold change cutoff for significance. Default 1.0.")
    parser.add_argument(
        "--method", choices=["welch", "moderated"], default="welch",
        help="'welch' (default): per-gene t-test, no cross-gene assumptions. "
             "'moderated': empirical-Bayes variance shrinkage across genes -- more statistical "
             "power, assumes each gene's two groups share one variance. See benchmarks/ for the tradeoff.",
    )
    parser.add_argument(
        "--annotation",
        help="Path to a two-column gene_id,gene_symbol CSV. If given, adds a gene_symbol "
             "column to differential_expression.csv (NaN for genes not in the file).",
    )
    parser.add_argument(
        "--metadata",
        help="Path to a sample metadata CSV (a 'sample' column, or the sample name as the "
             "first column, plus one column per covariate). Required together with "
             "--covariates; ignored otherwise.",
    )
    parser.add_argument(
        "--covariates",
        help="Comma-separated column names in --metadata to adjust for (e.g. a batch or sex "
             "column). Switches differential expression to a linear-model test that holds "
             "these fixed while testing group1 vs group2 -- see --method's help for the "
             "simpler default. Requires --metadata.",
    )
    parser.add_argument(
        "--no-moderated-covariates", action="store_true",
        help="With --covariates, use a plain per-gene OLS t-test instead of the default "
             "empirical-Bayes variance shrinkage across genes.",
    )
    parser.add_argument(
        "--live-enrichment-organism",
        help="g:Profiler organism code (e.g. 'hsapiens'). If given, also runs pathway "
             "enrichment against the live g:Profiler API and writes live_pathway_enrichment.csv. "
             "Independent of --gmt/--background. Requires outbound internet access.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating the volcano and PCA plots.")
    parser.add_argument("--explain", action="store_true", help="Generate a plain-language explanation of the results (requires ANTHROPIC_API_KEY).")
    parser.add_argument("--out", default="deconcord_output", help="Output directory. Default 'deconcord_output'.")
    parser.add_argument("--version", action="version", version=f"deconcord {_installed_versions()['deconcord']}")
    return parser


def _run(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    group_1 = _parse_sample_list(args.group1)
    group_2 = _parse_sample_list(args.group2)

    covariate_cols = None
    metadata = None
    if args.covariates is not None:
        if args.metadata is None:
            raise ValueError("--covariates requires --metadata to also be given.")
        covariate_cols = _parse_sample_list(args.covariates)
        metadata = pd.read_csv(args.metadata)
        if "sample" in metadata.columns:
            metadata = metadata.set_index("sample")
        else:
            metadata = metadata.set_index(metadata.columns[0])
    elif args.metadata is not None:
        raise ValueError("--metadata is only used together with --covariates.")

    gene_sets = None
    background_genes = None
    if args.gmt is not None:
        gene_sets = load_gmt(args.gmt)
        if args.background is not None:
            background_genes = set(_parse_sample_list(args.background))
        else:
            # No explicit background given: fall back to every gene in the
            # count matrix, which is the honest "everything that could have
            # been tested" universe for this run. Sniffed the same way
            # load_count_matrix (called inside run_analysis, below) reads
            # args.counts itself -- this needs to agree with that, or a
            # tab-separated counts file would parse fine in the actual
            # pipeline run but silently produce a single-column, useless
            # background gene set here.
            background_genes = set(pd.read_csv(args.counts, index_col=0, sep=sniff_delimiter(args.counts)).index)

    results = run_analysis(
        args.counts,
        group_1=group_1,
        group_2=group_2,
        gene_sets=gene_sets,
        background_genes=background_genes,
        min_count=args.min_count,
        min_samples=args.min_samples,
        alpha=args.alpha,
        lfc_threshold=args.lfc_threshold,
        method=args.method,
        metadata=metadata,
        covariate_cols=covariate_cols,
        moderated_covariates=not args.no_moderated_covariates,
        gene_annotation=args.annotation,
        generate_plots=not args.no_plots,
        explain_results=args.explain,
        live_enrichment_organism=args.live_enrichment_organism,
    )

    out_dir = Path(args.out)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    de = results["differential_expression"]
    de.to_csv(tables_dir / "differential_expression.csv")
    results["qc"].to_csv(tables_dir / "qc.csv")

    if "pathway_enrichment" in results:
        results["pathway_enrichment"].to_csv(tables_dir / "pathway_enrichment.csv", index=False)

    if "live_pathway_enrichment" in results:
        results["live_pathway_enrichment"].to_csv(tables_dir / "live_pathway_enrichment.csv", index=False)

    if "volcano_fig" in results:
        results["volcano_fig"].savefig(figures_dir / "volcano.png", dpi=150, bbox_inches="tight")
    if "pca_fig" in results:
        results["pca_fig"].savefig(figures_dir / "pca.png", dpi=150, bbox_inches="tight")
    if "pathway_enrichment_fig" in results:
        results["pathway_enrichment_fig"].savefig(figures_dir / "pathway_enrichment.png", dpi=150, bbox_inches="tight")
    if "live_pathway_enrichment_fig" in results:
        results["live_pathway_enrichment_fig"].savefig(figures_dir / "live_pathway_enrichment.png", dpi=150, bbox_inches="tight")

    if "explanation" in results:
        (out_dir / "explanation.txt").write_text(results["explanation"])

    run_metadata = {
        "deconcord_version": _installed_versions()["deconcord"],
        "python_version": platform.python_version(),
        "package_versions": _installed_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "deconcord " + " ".join(argv),
        "input_summary": {
            "counts_file": args.counts,
            "n_genes": int(results["raw_counts"].shape[0]),
            "n_samples": int(results["raw_counts"].shape[1]),
            "group_1": group_1,
            "group_2": group_2,
        },
        "parameters": {
            "method": args.method,
            "alpha": args.alpha,
            "lfc_threshold": args.lfc_threshold,
            "min_count": args.min_count,
            "min_samples": args.min_samples,
            "covariates": covariate_cols,
            "moderated_covariates": (not args.no_moderated_covariates) if covariate_cols else None,
            "gmt_file": args.gmt,
            "live_enrichment_organism": args.live_enrichment_organism,
            "annotation_file": args.annotation,
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    n_sig = int(de["significant"].sum())
    print(f"{len(de)} genes tested, {n_sig} significant at alpha={args.alpha}, |log2FC|>{args.lfc_threshold}.")
    print(f"Results written to {out_dir}/ (tables/, figures/, run_metadata.json)")

    return 0


def build_concordance_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deconcord concordance",
        description="Compare two existing DE result tables with compute_de_concordance, "
                     "without running the pipeline first.",
    )
    parser.add_argument("results_a", help="Path to the first DE result table CSV.")
    parser.add_argument("results_b", help="Path to the second DE result table CSV.")
    parser.add_argument("--name-a", default="method_a", help="Label for the first table. Default 'method_a'.")
    parser.add_argument("--name-b", default="method_b", help="Label for the second table. Default 'method_b'.")
    parser.add_argument(
        "--gene-id-col-a",
        help="Column in results_a to use as the gene ID. Defaults to the first column in the file.",
    )
    parser.add_argument(
        "--gene-id-col-b",
        help="Column in results_b to use as the gene ID. Defaults to the first column in the file.",
    )
    parser.add_argument("--lfc-col-a", default="log_fold_change", help="Log fold change column in results_a. Default 'log_fold_change'.")
    parser.add_argument("--pvalue-col-a", default="adjusted_p_value", help="Adjusted p-value column in results_a. Default 'adjusted_p_value'.")
    parser.add_argument("--lfc-col-b", default="log_fold_change", help="Log fold change column in results_b. Default 'log_fold_change'.")
    parser.add_argument("--pvalue-col-b", default="adjusted_p_value", help="Adjusted p-value column in results_b. Default 'adjusted_p_value'.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Adjusted p-value cutoff for significance. Default 0.05.")
    parser.add_argument("--out", default="deconcord_concordance_output", help="Output directory. Default 'deconcord_concordance_output'.")
    return parser


def _load_results_table(path: str, gene_id_col: str | None) -> pd.DataFrame:
    # A caller-given column name is used as-is; with no column named,
    # falling back to the first column (index_col=0) matches how
    # DEConcord's own DE output and a plain df.to_csv() from DESeq2/edgeR
    # both write their gene ID column, without forcing a specific name.
    if gene_id_col is not None:
        return pd.read_csv(path, index_col=gene_id_col)
    return pd.read_csv(path, index_col=0)


def _run_concordance(argv: list[str]) -> int:
    parser = build_concordance_parser()
    args = parser.parse_args(argv)

    results_a = _load_results_table(args.results_a, args.gene_id_col_a)
    results_b = _load_results_table(args.results_b, args.gene_id_col_b)

    result = compute_de_concordance(
        results_a, results_b,
        name_a=args.name_a, name_b=args.name_b,
        lfc_col_a=args.lfc_col_a, pvalue_col_a=args.pvalue_col_a,
        lfc_col_b=args.lfc_col_b, pvalue_col_b=args.pvalue_col_b,
        alpha=args.alpha,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2))

    merged = result["merged"].copy()
    merged.index.name = "gene_id"
    merged.to_csv(out_dir / "merged.csv")

    for key in ("concordant_genes", "discordant_genes", f"only_in_{args.name_a}", f"only_in_{args.name_b}"):
        genes = result[key]
        filename = key + ".csv"
        pd.Series(genes, name="gene_id").to_csv(out_dir / filename, index=False)

    run_metadata = {
        "deconcord_version": _installed_versions()["deconcord"],
        "python_version": platform.python_version(),
        "package_versions": _installed_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "deconcord concordance " + " ".join(argv),
        "input_summary": {
            "results_a_file": args.results_a,
            "results_b_file": args.results_b,
            "name_a": args.name_a,
            "name_b": args.name_b,
        },
        "parameters": {
            "alpha": args.alpha,
            "lfc_col_a": args.lfc_col_a,
            "pvalue_col_a": args.pvalue_col_a,
            "lfc_col_b": args.lfc_col_b,
            "pvalue_col_b": args.pvalue_col_b,
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    summary = result["summary"]
    print(
        f"{summary['genes_compared']} genes compared, "
        f"jaccard={summary['jaccard_index']:.3f}, "
        f"{summary['significant_in_both']} significant in both at alpha={args.alpha}."
    )
    print(f"Results written to {out_dir}/")

    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    try:
        if argv and argv[0] == "concordance":
            return _run_concordance(argv[1:])
        return _run(argv)
    except (CountMatrixError, ValueError, FileNotFoundError, ConnectionError, RuntimeError) as exc:
        print(f"deconcord: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
