"""Covariate-adjusted linear model differential expression:
run_differential_expression_with_covariates. See
test_differential_expression_welch.py and
test_differential_expression_moderated.py for the two-group-only methods
this builds on.
"""
import numpy as np
import pandas as pd
import pytest

from deconcord.differential_expression.methods import (
    compute_log_fold_change,
    run_differential_expression_with_covariates,
)


def _confounded_batch_dataset():
    # group_1 = s1,s2,s3 / group_2 = s4,s5,s6, but batch membership is
    # unevenly split across the two condition groups (2 group_1 samples
    # and 1 group_2 sample are batch_B) -- a real, if imperfect,
    # confound. True condition effect is +2.0; true batch_B effect is
    # +5.0 on top of a baseline of 10. Small deterministic perturbations
    # are added so the model doesn't fit exactly (residual variance would
    # otherwise be zero).
    group_1 = ["s1", "s2", "s3"]
    group_2 = ["s4", "s5", "s6"]
    batch = {"s1": "batch_A", "s2": "batch_A", "s3": "batch_B",
             "s4": "batch_B", "s5": "batch_B", "s6": "batch_A"}
    noiseless = {"s1": 12.0, "s2": 12.0, "s3": 17.0, "s4": 15.0, "s5": 15.0, "s6": 10.0}
    perturbation = {"s1": 0.05, "s2": -0.05, "s3": 0.02, "s4": -0.02, "s5": 0.03, "s6": -0.03}
    values = {s: noiseless[s] + perturbation[s] for s in noiseless}

    df = pd.DataFrame({"gene1": values}).T
    metadata = pd.DataFrame({"batch": batch})
    return df, group_1, group_2, metadata


def test_run_differential_expression_with_covariates_matches_manual_ols():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    samples = group_1 + group_2

    results = run_differential_expression_with_covariates(
        df, group_1, group_2, metadata, ["batch"], moderated=False,
    )

    # Rebuild the same design matrix independently (intercept, condition,
    # batch_B dummy -- batch_A is the alphabetically-first, dropped level)
    # and solve it directly, to check the function's result against a
    # from-scratch computation rather than another layer of the same code.
    condition = np.array([1.0 if s in group_1 else 0.0 for s in samples])
    batch_b = np.array([1.0 if metadata.loc[s, "batch"] == "batch_B" else 0.0 for s in samples])
    X = np.column_stack([np.ones(6), condition, batch_b])
    y = df.loc["gene1", samples].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    expected_lfc = beta[1]

    assert results.loc["gene1", "log_fold_change"] == pytest.approx(expected_lfc)
    assert results.loc["gene1", "log_fold_change"] == pytest.approx(2.0, abs=0.2)


def test_run_differential_expression_with_covariates_recovers_effect_naive_test_confounds():
    # The point of the feature: the plain two-group comparison conflates
    # the batch effect into the condition estimate (batch is unevenly
    # split across the groups), while the covariate-adjusted model
    # isolates the true ~2.0 condition effect.
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    naive_lfc = compute_log_fold_change(df, group_1, group_2)["gene1"]
    adjusted = run_differential_expression_with_covariates(
        df, group_1, group_2, metadata, ["batch"], moderated=False,
    )
    adjusted_lfc = adjusted.loc["gene1", "log_fold_change"]

    assert abs(adjusted_lfc - 2.0) < abs(naive_lfc - 2.0)


def test_run_differential_expression_with_covariates_numeric_covariate():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    metadata["numeric_batch"] = metadata["batch"].map({"batch_A": 0.0, "batch_B": 1.0})

    results = run_differential_expression_with_covariates(
        df, group_1, group_2, metadata, ["numeric_batch"], moderated=False,
    )

    assert results.loc["gene1", "log_fold_change"] == pytest.approx(2.0, abs=0.2)


def test_run_differential_expression_with_covariates_returns_expected_columns():
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    results = run_differential_expression_with_covariates(
        df, group_1, group_2, metadata, ["batch"], moderated=False,
    )

    assert list(results.columns) == ["log_fold_change", "p_value", "adjusted_p_value", "significant"]
    assert list(results.index) == list(df.index)


def test_run_differential_expression_with_covariates_empty_covariate_cols_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    with pytest.raises(ValueError, match="covariate_cols"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, [])


def test_run_differential_expression_with_covariates_missing_sample_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    metadata = metadata.drop("s3")

    with pytest.raises(ValueError, match="s3"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, ["batch"])


def test_run_differential_expression_with_covariates_missing_column_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    with pytest.raises(ValueError, match="not_a_real_column"):
        run_differential_expression_with_covariates(
            df, group_1, group_2, metadata, ["not_a_real_column"],
        )


def test_run_differential_expression_with_covariates_nan_covariate_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    metadata.loc["s2", "batch"] = None

    with pytest.raises(ValueError, match="s2"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, ["batch"])


def test_run_differential_expression_with_covariates_constant_covariate_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    metadata["batch"] = "same_batch_for_everyone"

    with pytest.raises(ValueError, match="only one distinct value"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, ["batch"], moderated=False)


def test_run_differential_expression_with_covariates_constant_numeric_covariate_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    metadata["numeric_batch"] = 3.0

    with pytest.raises(ValueError, match="rank-deficient"):
        run_differential_expression_with_covariates(
            df, group_1, group_2, metadata, ["numeric_batch"], moderated=False,
        )


def test_run_differential_expression_with_covariates_collinear_with_condition_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    # A covariate that's just a relabeling of group membership is perfectly
    # collinear with the condition term -- their individual effects can't
    # be separated.
    metadata["batch"] = ["batch_A" if s in group_1 else "batch_B" for s in metadata.index]

    with pytest.raises(ValueError, match="rank-deficient"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, ["batch"])


def test_run_differential_expression_with_covariates_no_residual_df_raises():
    # A design matrix that's full column rank (so it passes the
    # rank-deficient check) but square -- as many parameters as samples --
    # leaves zero residual degrees of freedom to estimate variance from.
    # intercept + condition + 2 batch dummies (a 3-level covariate) = 4
    # parameters, exactly matching 4 samples here, by construction.
    group_1 = ["s1", "s2"]
    group_2 = ["s3", "s4"]
    batch = {"s1": "batch_A", "s2": "batch_B", "s3": "batch_C", "s4": "batch_A"}
    df = pd.DataFrame({"gene1": {"s1": 12.0, "s2": 13.0, "s3": 15.0, "s4": 11.0}}).T
    metadata = pd.DataFrame({"batch": batch})

    with pytest.raises(ValueError, match="residual degree of freedom"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, ["batch"], moderated=False)


def test_run_differential_expression_with_covariates_moderated_needs_two_genes():
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    with pytest.raises(ValueError, match="at least 2 genes"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, ["batch"], moderated=True)


def test_run_differential_expression_with_covariates_moderated_multi_gene():
    df, group_1, group_2, metadata = _confounded_batch_dataset()
    # Add background genes with tight, consistent variance so the
    # empirical Bayes prior has something to fit.
    for i in range(6):
        base = 10 + i * 0.4
        df.loc[f"background_{i}"] = [
            base + 0.03, base - 0.02, base + 0.04, base - 0.01, base + 0.02, base - 0.03,
        ]

    results = run_differential_expression_with_covariates(
        df, group_1, group_2, metadata, ["batch"], moderated=True,
    )

    assert not results["p_value"].isna().any()
    assert ((results["p_value"] >= 0) & (results["p_value"] <= 1)).all()


def test_run_differential_expression_with_covariates_invalid_alpha_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    with pytest.raises(ValueError, match="alpha"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, ["batch"], alpha=0)


def test_run_differential_expression_with_covariates_negative_lfc_threshold_raises():
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    with pytest.raises(ValueError, match="lfc_threshold"):
        run_differential_expression_with_covariates(
            df, group_1, group_2, metadata, ["batch"], lfc_threshold=-1,
        )


def test_run_differential_expression_with_covariates_no_covariates_message():
    df, group_1, group_2, metadata = _confounded_batch_dataset()

    with pytest.raises(ValueError, match="run_differential_expression instead"):
        run_differential_expression_with_covariates(df, group_1, group_2, metadata, [])
