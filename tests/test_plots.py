import numpy as np
import pandas as pd
import pytest

from deconcord.differential_expression.methods import run_differential_expression
from deconcord.visualization.plots import plot_volcano, plot_pca, plot_pathway_enrichment


def test_plot_volcano_returns_figure():
    df = pd.DataFrame({
        "sample1": [10, 5], "sample2": [20, 15],
        "sample3": [30, 25], "sample4": [28, 24],
    }, index=["gene1", "gene2"])
    results_df = run_differential_expression(df, ["sample1", "sample2"], ["sample3", "sample4"])
    fig = plot_volcano(results_df)
    assert fig is not None


def test_plot_volcano_handles_zero_adjusted_pvalue():
    # An adjusted p-value of exactly 0 is a real possibility now (see the
    # constant-expression special case in compute_pvalues) and used to
    # send -log10 to -inf. This should plot cleanly instead of warning or
    # producing an unbounded axis.
    results_df = pd.DataFrame({
        "log_fold_change": [3.0, 0.2],
        "adjusted_p_value": [0.0, 0.6],
        "significant": [True, False],
    }, index=["gene1", "gene2"])

    with np.errstate(divide="raise"):
        fig = plot_volcano(results_df)

    assert fig is not None
    assert np.isfinite(fig.axes[0].collections[1].get_offsets()[:, 1]).all()


def test_plot_volcano_missing_column_raises():
    # plot_pca/plot_pathway_enrichment both validate required columns and
    # raise a specific ValueError; plot_volcano didn't, so a mistyped
    # column name used to surface as a bare KeyError from inside the
    # scatter call instead of a message naming the actual problem.
    results_df = pd.DataFrame({
        "log_fold_change": [3.0, 0.2],
        "adjusted_p_value": [0.01, 0.6],
        # "significant" column missing on purpose
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="significant"):
        plot_volcano(results_df)


def test_plot_volcano_empty_raises():
    results_df = pd.DataFrame(columns=["log_fold_change", "adjusted_p_value", "significant"])

    with pytest.raises(ValueError, match="empty"):
        plot_volcano(results_df)


def test_plot_pca_returns_figure():
    df = pd.DataFrame({
        "sample1": [10, 5, 2, 8], "sample2": [12, 6, 3, 9],
        "sample3": [30, 25, 20, 28], "sample4": [28, 24, 22, 26],
    }, index=["gene1", "gene2", "gene3", "gene4"])
    labels = pd.Series(["control", "control", "treatment", "treatment"], index=df.columns)

    fig = plot_pca(df, labels)
    assert fig is not None


def test_plot_pca_too_few_samples_raises():
    df = pd.DataFrame({"sample1": [10, 5, 2]}, index=["gene1", "gene2", "gene3"])
    labels = pd.Series(["control"], index=df.columns)

    with pytest.raises(ValueError, match=r"got 1 sample"):
        plot_pca(df, labels)


def test_plot_pca_too_few_genes_raises():
    df = pd.DataFrame({"sample1": [10], "sample2": [20]}, index=["gene1"])
    labels = pd.Series(["control", "treatment"], index=df.columns)

    with pytest.raises(ValueError, match=r"got 2 sample\(s\) and 1 gene"):
        plot_pca(df, labels)


def _local_enrichment_df():
    return pd.DataFrame({
        "pathway": ["PATHWAY_A", "PATHWAY_B", "PATHWAY_C"],
        "p_value": [0.001, 0.02, 0.5],
        "adjusted_p_value": [0.003, 0.03, 0.5],
        "overlap_count": [10, 4, 1],
    })


def test_plot_pathway_enrichment_local_schema_returns_figure():
    fig = plot_pathway_enrichment(_local_enrichment_df())
    assert fig is not None
    # 3 rows in, 3 dots plotted (the scatter's y-tick labels are the
    # pathway names since they're plotted against a categorical axis).
    assert len(fig.axes[0].collections[0].get_offsets()) == 3


def test_plot_pathway_enrichment_live_schema_returns_figure():
    live_df = pd.DataFrame({
        "source": ["GO:BP", "KEGG"],
        "name": ["apoptotic process", "Apoptosis"],
        "p_value": [0.02, 0.001],
        "intersection_size": [5, 8],
    })
    fig = plot_pathway_enrichment(
        live_df, name_col="name", pvalue_col="p_value", size_col="intersection_size",
    )
    assert fig is not None


def test_plot_pathway_enrichment_respects_top_n():
    df = pd.DataFrame({
        "pathway": [f"PATHWAY_{i}" for i in range(30)],
        "adjusted_p_value": np.linspace(0.001, 0.5, 30),
        "overlap_count": range(30),
    })
    fig = plot_pathway_enrichment(df, top_n=5)
    assert len(fig.axes[0].collections[0].get_offsets()) == 5


def test_plot_pathway_enrichment_handles_zero_pvalue():
    df = pd.DataFrame({
        "pathway": ["PATHWAY_A", "PATHWAY_B"],
        "adjusted_p_value": [0.0, 0.4],
        "overlap_count": [10, 2],
    })

    with np.errstate(divide="raise"):
        fig = plot_pathway_enrichment(df)

    assert np.isfinite(fig.axes[0].collections[0].get_offsets()[:, 0]).all()


def test_plot_pathway_enrichment_empty_raises():
    empty = pd.DataFrame(columns=["pathway", "adjusted_p_value", "overlap_count"])
    with pytest.raises(ValueError, match="empty"):
        plot_pathway_enrichment(empty)


def test_plot_pathway_enrichment_missing_column_raises():
    df = pd.DataFrame({"pathway": ["PATHWAY_A"], "adjusted_p_value": [0.01]})
    with pytest.raises(ValueError, match="overlap_count"):
        plot_pathway_enrichment(df)


def test_plot_pathway_enrichment_without_size_col():
    df = pd.DataFrame({"pathway": ["PATHWAY_A", "PATHWAY_B"], "adjusted_p_value": [0.01, 0.2]})
    fig = plot_pathway_enrichment(df, size_col=None)
    assert fig is not None
