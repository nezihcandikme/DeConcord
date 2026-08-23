"""Welch's t-test differential expression: log fold change, p-values,
BH adjustment, and run_differential_expression's default (method="welch")
path. See test_differential_expression_moderated.py for the empirical-Bayes
method and test_differential_expression_covariates.py for the covariate-
adjusted linear model.
"""
import numpy as np
import pandas as pd
import pytest

from deconcord.differential_expression.methods import (
    compute_log_fold_change,
    compute_pvalues,
    compute_adjusted_pvalues,
    run_differential_expression,
)


def test_compute_log_fold_change():
    df = pd.DataFrame({
        "sample1": [10, 5, 2],
        "sample2": [20, 15, 4],
        "sample3": [30, 25, 6],
    }, index=["gene1", "gene2", "gene3"])

    group_1 = ['sample1', 'sample2']
    group_2 = ['sample3']

    log_fc = compute_log_fold_change(df, group_1, group_2)

    expected_log_fc = pd.Series({
        'gene1': (10 + 20) / 2 - 30,
        'gene2': (5 + 15) / 2 - 25,
        'gene3': (2 + 4) / 2 - 6
    })

    pd.testing.assert_series_equal(log_fc, expected_log_fc)


def test_compute_pvalues_clear_difference():
    df = pd.DataFrame({
        "sample1": [1, 1, 1],
        "sample2": [2, 2, 2],
        "sample3": [1000, 1000, 1000],
        "sample4": [999, 999, 999],
    }, index=["gene1", "gene2", "gene3"])

    pvalues = compute_pvalues(df, ["sample1", "sample2"], ["sample3", "sample4"])

    assert all(pvalues < 0.05)


def test_compute_pvalues_no_difference():
    df = pd.DataFrame({
        "sample1": [9, 19, 4],
        "sample2": [11, 21, 6],
        "sample3": [8, 18, 3],
        "sample4": [12, 22, 7],
    }, index=["gene1", "gene2", "gene3"])

    pvalues = compute_pvalues(df, ["sample1", "sample2"], ["sample3", "sample4"])

    assert all(pvalues.apply(lambda p: p == pytest.approx(1.0, abs=0.01)))


def test_compute_adjusted_pvalues_single():
    pvalues = pd.Series({"gene1": 0.03})
    adjusted = compute_adjusted_pvalues(pvalues)
    assert adjusted["gene1"] == pytest.approx(0.03)


def test_compute_adjusted_pvalues_never_smaller():
    pvalues = pd.Series({"gene1": 0.01, "gene2": 0.2, "gene3": 0.04, "gene4": 0.5})
    adjusted = compute_adjusted_pvalues(pvalues)
    assert all(adjusted >= pvalues)


def test_compute_adjusted_pvalues_equal_spacing():
    pvalues = pd.Series({
        "gene1": 0.01, "gene2": 0.02, "gene3": 0.03, "gene4": 0.04, "gene5": 0.05
    })
    adjusted = compute_adjusted_pvalues(pvalues)
    assert all(adjusted.apply(lambda p: p == pytest.approx(0.05, abs=1e-6)))


def test_compute_adjusted_pvalues_nan_input_raises():
    # A single NaN silently poisons every adjusted p-value via statsmodels
    # — that should never happen quietly, so this must raise instead.
    pvalues = pd.Series({"gene1": 0.01, "gene2": np.nan, "gene3": 0.03})

    with pytest.raises(ValueError, match="gene2"):
        compute_adjusted_pvalues(pvalues)


def test_run_differential_expression():
    df = pd.DataFrame({
        "sample1": [10, 5, 2],
        "sample2": [20, 15, 4],
        "sample3": [30, 25, 6],
        "sample4": [28, 24, 5],
    }, index=["gene1", "gene2", "gene3"])

    results_df = run_differential_expression(df, ["sample1", "sample2"], ["sample3", "sample4"])

    assert "log_fold_change" in results_df.columns
    assert "p_value" in results_df.columns
    assert "adjusted_p_value" in results_df.columns
    assert "significant" in results_df.columns


def test_run_differential_expression_custom_thresholds_change_significance():
    df = pd.DataFrame({
        "sample1": [1, 1, 1],
        "sample2": [2, 2, 2],
        "sample3": [3, 3, 3],
        "sample4": [4, 4, 4],
    }, index=["gene1", "gene2", "gene3"])

    default_results = run_differential_expression(df, ["sample1", "sample2"], ["sample3", "sample4"])
    lenient_results = run_differential_expression(
        df, ["sample1", "sample2"], ["sample3", "sample4"], alpha=1.0, lfc_threshold=0.0
    )

    # Same underlying numbers either way — only which rows count as
    # "significant" should move when the thresholds move.
    pd.testing.assert_series_equal(default_results["p_value"], lenient_results["p_value"])
    assert lenient_results["significant"].sum() >= default_results["significant"].sum()


def test_run_differential_expression_invalid_alpha_raises():
    df = pd.DataFrame({
        "sample1": [1, 2], "sample2": [1, 2], "sample3": [3, 4], "sample4": [3, 4],
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="alpha"):
        run_differential_expression(df, ["sample1", "sample2"], ["sample3", "sample4"], alpha=0)

    with pytest.raises(ValueError, match="alpha"):
        run_differential_expression(df, ["sample1", "sample2"], ["sample3", "sample4"], alpha=1.5)


def test_run_differential_expression_negative_lfc_threshold_raises():
    df = pd.DataFrame({
        "sample1": [1, 2], "sample2": [1, 2], "sample3": [3, 4], "sample4": [3, 4],
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="lfc_threshold"):
        run_differential_expression(df, ["sample1", "sample2"], ["sample3", "sample4"], lfc_threshold=-1)


def test_compute_pvalues_constant_expression_equal_means():
    df = pd.DataFrame({
        "sample1": [5, 5],
        "sample2": [5, 5],
        "sample3": [5, 5],
        "sample4": [5, 5],
    }, index=["gene1", "gene2"])

    pvalues = compute_pvalues(df, ["sample1", "sample2"], ["sample3", "sample4"])

    assert not pvalues.isna().any()
    assert all(pvalues.apply(lambda p: p == pytest.approx(1.0)))


def test_compute_pvalues_constant_expression_different_means():
    df = pd.DataFrame({
        "sample1": [1, 1],
        "sample2": [1, 1],
        "sample3": [9, 9],
        "sample4": [9, 9],
    }, index=["gene1", "gene2"])

    pvalues = compute_pvalues(df, ["sample1", "sample2"], ["sample3", "sample4"])

    assert not pvalues.isna().any()
    assert all(pvalues.apply(lambda p: p == pytest.approx(0.0)))


def test_run_differential_expression_no_nan_leaks_through_for_constant_genes():
    # Regression test tying the constant-expression fix in compute_pvalues
    # to the NaN guard in compute_adjusted_pvalues: a dataset mixing a
    # constant gene with normal ones should produce a fully usable table.
    df = pd.DataFrame({
        "sample1": [5, 1, 3],
        "sample2": [5, 2, 4],
        "sample3": [5, 1000, 15],
        "sample4": [5, 999, 13],
    }, index=["constant_gene", "clear_gene", "ordinary_gene"])

    results_df = run_differential_expression(df, ["sample1", "sample2"], ["sample3", "sample4"])

    assert not results_df["p_value"].isna().any()
    assert not results_df["adjusted_p_value"].isna().any()


def test_compute_pvalues_insufficient_samples():
    df = pd.DataFrame({
        "sample1": [10, 5, 2],
        "sample2": [20, 15, 4],
        "sample3": [30, 25, 6],
    }, index=["gene1", "gene2", "gene3"])

    with pytest.raises(ValueError):
        compute_pvalues(df, ["sample1", "sample2"], ["sample3"])


def test_compute_pvalues_missing_sample_names_the_sample():
    df = pd.DataFrame({
        "sample1": [10, 5], "sample2": [20, 15],
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="control_1"):
        compute_pvalues(df, ["sample1", "control_1"], ["sample2"])


def test_compute_log_fold_change_duplicate_sample_in_same_group_raises():
    df = pd.DataFrame({
        "sample1": [10, 5], "sample2": [20, 15],
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="sample1"):
        compute_log_fold_change(df, ["sample1", "sample1"], ["sample2"])


def test_compute_log_fold_change_sample_in_both_groups_raises():
    df = pd.DataFrame({
        "sample1": [10, 5], "sample2": [20, 15],
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="sample1"):
        compute_log_fold_change(df, ["sample1"], ["sample1", "sample2"])


def test_compute_log_fold_change_empty_group_raises():
    df = pd.DataFrame({
        "sample1": [10, 5], "sample2": [20, 15],
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="group_2"):
        compute_log_fold_change(df, ["sample1"], [])


def test_compute_log_fold_change_empty_group_1_raises():
    # The group_2 case above only exercises one of the two symmetric empty
    # checks in _validate_groups -- pinned down separately so group_1's own
    # check (raised first, before group_2 is even looked at) has direct
    # coverage rather than being assumed to work because group_2's does.
    df = pd.DataFrame({
        "sample1": [10, 5], "sample2": [20, 15],
    }, index=["gene1", "gene2"])

    with pytest.raises(ValueError, match="group_1"):
        compute_log_fold_change(df, [], ["sample2"])

