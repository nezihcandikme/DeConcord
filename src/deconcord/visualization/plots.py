from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Adjusted p-values of exactly 0 are possible (e.g. a constant-expression
# gene with a clear between-group difference — see compute_pvalues). -log10(0)
# is -inf, which would otherwise stretch the y-axis to infinity and print a
# RuntimeWarning. Flooring at a tiny epsilon keeps the plot readable without
# hiding how significant the point actually is.
_MIN_PLOTTABLE_PVALUE = 1e-300


def plot_volcano(results_df: pd.DataFrame) -> plt.Figure:
    """
    Volcano plot: log fold change (x) vs. -log10 adjusted p-value (y).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``run_differential_expression`` — needs ``log_fold_change``,
        ``adjusted_p_value``, and ``significant`` columns.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If ``results_df`` is empty, or missing any of the three required
        columns — matches the validation ``plot_pca``/``plot_pathway_enrichment``
        already do, so a mistyped column name or an accidentally-empty
        results table fails with a specific message here instead of a bare
        ``KeyError`` from inside matplotlib's plotting call.
    """
    required_cols = ["log_fold_change", "adjusted_p_value", "significant"]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        raise ValueError(
            f"results_df is missing column(s) {missing}. "
            f"Found columns: {list(results_df.columns)}."
        )
    if results_df.empty:
        raise ValueError("results_df is empty — nothing to plot.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(y=-np.log10(0.05), color="black", linestyle="--", label="Significance threshold (0.05)")
    ax.set_title("Volcano Plot")

    significant = results_df["significant"]
    plottable_pvalues = results_df["adjusted_p_value"].clip(lower=_MIN_PLOTTABLE_PVALUE)

    ax.scatter(
        results_df.loc[~significant, "log_fold_change"],
        -np.log10(plottable_pvalues.loc[~significant]),
        alpha=0.5, color="gray", label="Not significant",
    )
    ax.scatter(
        results_df.loc[significant, "log_fold_change"],
        -np.log10(plottable_pvalues.loc[significant]),
        alpha=0.5, color="red", label="Significant",
    )

    ax.axvline(1, color="black", linestyle="--")
    ax.axvline(-1, color="black", linestyle="--")
    ax.legend()
    ax.set_xlabel("Log Fold Change")
    ax.set_ylabel("-Log10 Adjusted P-value")
    return fig


def plot_pathway_enrichment(
    enrichment_df: pd.DataFrame,
    name_col: str = "pathway",
    pvalue_col: str = "adjusted_p_value",
    size_col: str | None = "overlap_count",
    top_n: int = 20,
) -> plt.Figure:
    """
    Dot plot of pathway enrichment results: one row per pathway, x-axis is
    -log10 p-value, dot size is the gene overlap/intersection count. Same
    layout the field generally uses for this (e.g. clusterProfiler's
    ``dotplot``) — significance and effect size in one view instead of
    scanning a table.

    Column names are parameters, not hardcoded, because the two enrichment
    sources in this package use different schemas: the local hypergeometric
    test (``run_pathway_enrichment_analysis``) returns ``pathway``/
    ``adjusted_p_value``/``overlap_count`` (the defaults here), while the
    live g:Profiler client (``run_gprofiler_enrichment``) returns ``name``/
    ``p_value``/``intersection_size``. Same function, either source.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Enrichment results, e.g. from ``run_pathway_enrichment_analysis``
        or ``run_gprofiler_enrichment``.
    name_col : str, optional
        Column with the pathway/term name. Default ``"pathway"``.
    pvalue_col : str, optional
        Column with the p-value to plot (adjusted or raw — whichever the
        source provides). Default ``"adjusted_p_value"``.
    size_col : str, optional
        Column with the gene overlap count, used for dot size. Default
        ``"overlap_count"``. Pass ``None`` for uniform dot size if the
        source doesn't provide one.
    top_n : int, optional
        Plot at most this many pathways, the most significant first.
        Default 20 — enough to see the shape of the result without a plot
        that's mostly whitespace and tiny text.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If ``enrichment_df`` is empty, or missing any of the requested
        columns — an empty enrichment result is a legitimate outcome
        (call code should check for it and skip plotting), but this
        function's job is to plot rows, not to decide whether zero rows
        is worth a blank chart.
    """
    if enrichment_df.empty:
        raise ValueError("enrichment_df is empty — nothing to plot.")

    required_cols = [name_col, pvalue_col] + ([size_col] if size_col else [])
    missing = [c for c in required_cols if c not in enrichment_df.columns]
    if missing:
        raise ValueError(
            f"enrichment_df is missing column(s) {missing}. "
            f"Found columns: {list(enrichment_df.columns)}."
        )

    plot_df = enrichment_df.sort_values(pvalue_col).head(top_n).iloc[::-1]
    plottable_pvalues = plot_df[pvalue_col].clip(lower=_MIN_PLOTTABLE_PVALUE)
    neg_log10_p = -np.log10(plottable_pvalues)

    if size_col:
        counts = plot_df[size_col]
        # Scaled so the smallest real overlap is still visible and the
        # largest doesn't swallow its neighbors -- purely a display
        # transform, doesn't touch the underlying counts.
        sizes = 40 + 260 * (counts - counts.min()) / (counts.max() - counts.min() + 1e-9)
    else:
        counts = None
        sizes = 120

    fig_height = max(3, 0.4 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    scatter = ax.scatter(neg_log10_p, plot_df[name_col], s=sizes, c=neg_log10_p, cmap="viridis", alpha=0.85)
    ax.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="p = 0.05")

    ax.set_xlabel(f"-Log10({pvalue_col})")
    ax.set_title("Pathway Enrichment")
    ax.legend(loc="lower right")
    fig.colorbar(scatter, ax=ax, label=f"-Log10({pvalue_col})")

    if counts is not None:
        # A small size-legend, not tied to the color scale -- two encodings,
        # two legends, so dot size doesn't get silently read as color.
        for count_value in sorted(set([counts.min(), counts.median(), counts.max()])):
            size = 40 + 260 * (count_value - counts.min()) / (counts.max() - counts.min() + 1e-9)
            ax.scatter([], [], s=size, color="gray", alpha=0.6, label=f"overlap = {int(count_value)}")
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


def plot_pca(df: pd.DataFrame, labels: pd.Series) -> plt.Figure:
    """
    2-component PCA scatter plot over samples, colored by ``labels``.

    Parameters
    ----------
    df : pd.DataFrame
        Expression matrix, genes as rows, samples as columns. Typically
        the normalized, log-transformed matrix rather than raw counts,
        so a handful of very highly expressed genes don't dominate the
        variance PCA is built to summarize.
    labels : pd.Series
        Group label per sample, indexed to match ``df.columns``.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If ``df`` has fewer than 2 samples or fewer than 2 genes — there's
        no such thing as 2 principal components of less than 2 dimensions,
        and scikit-learn's own error for this is not exactly self-explanatory.
    """
    n_genes, n_samples = df.shape
    if n_samples < 2 or n_genes < 2:
        raise ValueError(
            f"plot_pca needs at least 2 samples and 2 genes to compute 2 "
            f"principal components; got {n_samples} sample(s) and "
            f"{n_genes} gene(s)."
        )

    pca = PCA(n_components=2)
    coords = pca.fit_transform(df.T)
    variance_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in labels.unique():
        mask = (labels == label).values
        ax.scatter(coords[mask, 0], coords[mask, 1], label=str(label), alpha=0.7)
    ax.legend()

    for i, sample_name in enumerate(df.columns):
        ax.annotate(sample_name, (coords[i, 0], coords[i, 1]))

    ax.set_title("PCA Plot")
    ax.set_xlabel(f"Principal Component 1 ({variance_ratio[0]*100:.1f}% variance)")
    ax.set_ylabel(f"Principal Component 2 ({variance_ratio[1]*100:.1f}% variance)")

    return fig
