"""Resampling stability: rerunning DE on perturbed sample sets and checking
how often each gene's significance call reappears. See
src/deconcord/concordance/resampling_stability.py's module docstring for
why bootstrap resampling isn't offered (only "subsample" and
"leave_one_out" are).
"""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from deconcord.concordance.resampling_stability import (
    _direction_stability,
    _rank_stability,
    compute_resampling_stability,
)
from deconcord.differential_expression.methods import run_differential_expression

GROUP_1 = ["s1", "s2", "s3"]
GROUP_2 = ["s4", "s5", "s6"]


def _df():
    # "strong": huge, low-noise gap between groups -- stays significant no
    # matter which sample is dropped or subsampled out.
    # "borderline": significant with every sample present, but losing
    # either s1 or s2 (both slightly pull group_1's mean down) flips it to
    # non-significant. Losing anything else leaves it significant. Values
    # and the resulting flip pattern were checked against a real run of
    # run_differential_expression before being hardcoded here.
    # "flat": near-identical means in both groups -- never significant.
    return pd.DataFrame({
        "s1": [1.0, 5.0, 5.0],
        "s2": [1.2, 5.05, 5.0],
        "s3": [0.8, 4.5, 5.0],
        "s4": [9.0, 3.0, 5.02],
        "s5": [9.3, 3.05, 4.98],
        "s6": [8.7, 3.1, 5.03],
    }, index=["strong", "borderline", "flat"])


def test_leave_one_out_baseline_matches_direct_call():
    df = _df()
    result = compute_resampling_stability(df, GROUP_1, GROUP_2, resample_method="leave_one_out")
    direct = run_differential_expression(df, GROUP_1, GROUP_2, alpha=0.05, lfc_threshold=1.0, method="welch")
    pd.testing.assert_frame_equal(result["baseline_results"], direct)


def test_leave_one_out_gene_stability_fractions():
    result = compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]

    assert stability.loc["strong", "baseline_significant"]
    assert stability.loc["strong", "frac_significant"] == pytest.approx(1.0)
    assert stability.loc["borderline", "baseline_significant"]
    assert stability.loc["borderline", "frac_significant"] == pytest.approx(float(Fraction(4, 6)))
    assert not stability.loc["flat", "baseline_significant"]
    assert stability.loc["flat", "frac_significant"] == pytest.approx(0.0)


def test_leave_one_out_stable_vs_sensitive_genes():
    result = compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")

    # "strong" replicates every time (stable, baseline-significant),
    # "flat" is never significant either way (stable, baseline-non-
    # significant), "borderline" only replicates 4/6 of the time, below
    # the default stability_threshold of 0.9, so it's sensitive despite
    # being significant in the baseline run.
    assert list(result["stable_genes"]) == ["flat", "strong"]
    assert list(result["sensitive_genes"]) == ["borderline"]


def test_leave_one_out_summary():
    result = compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    summary = result["summary"]

    assert summary["resample_method"] == "leave_one_out"
    assert summary["n_iterations"] == 6  # 3 + 3 samples, each dropped once
    assert summary["n_baseline_significant"] == 2  # strong, borderline
    assert summary["mean_jaccard_to_baseline"] == pytest.approx(float(Fraction(5, 6)))
    assert summary["baseline_replication_rate"] == pytest.approx(float(Fraction(5, 6)))


def test_leave_one_out_summary_all_nan_when_no_baseline_significant_genes():
    # An alpha strict enough that nothing clears it in the baseline run --
    # the three summary stats that are averages *over* baseline-significant
    # genes (baseline_replication_rate, mean_direction_stability,
    # mean_rank_stability) have nothing to average, so they must come back
    # as NaN rather than crashing on an empty .mean() or silently returning
    # 0 (which would misleadingly read as "unstable" instead of "undefined").
    result = compute_resampling_stability(
        _df(), GROUP_1, GROUP_2, resample_method="leave_one_out", alpha=1e-12,
    )
    summary = result["summary"]

    assert summary["n_baseline_significant"] == 0
    assert np.isnan(summary["baseline_replication_rate"])
    assert np.isnan(summary["mean_direction_stability"])
    assert np.isnan(summary["mean_rank_stability"])


def test_leave_one_out_needs_at_least_three_samples_per_group():
    df = _df()
    with pytest.raises(ValueError, match="leave_one_out needs at least 3 samples"):
        compute_resampling_stability(df, ["s1", "s2"], GROUP_2, resample_method="leave_one_out")


def test_subsample_reproducible_with_same_random_state():
    df = _df()
    a = compute_resampling_stability(
        df, GROUP_1, GROUP_2, resample_method="subsample", n_iterations=50, random_state=7,
    )
    b = compute_resampling_stability(
        df, GROUP_1, GROUP_2, resample_method="subsample", n_iterations=50, random_state=7,
    )
    pd.testing.assert_frame_equal(a["gene_stability"], b["gene_stability"])
    assert a["summary"] == b["summary"]


def test_subsample_strong_and_flat_genes_are_stable():
    result = compute_resampling_stability(
        _df(), GROUP_1, GROUP_2, resample_method="subsample", n_iterations=100, random_state=1,
    )
    stability = result["gene_stability"]

    # subsample_fraction defaults to 0.8; with 3 samples per group that
    # rounds to keeping 2, the same "drop one" gap "strong"/"flat" were
    # designed to survive regardless of which one is dropped.
    assert stability.loc["strong", "frac_significant"] == pytest.approx(1.0)
    assert stability.loc["flat", "frac_significant"] == pytest.approx(0.0)
    assert "strong" in result["stable_genes"]
    assert "flat" in result["stable_genes"]


def test_subsample_honors_n_iterations():
    result = compute_resampling_stability(
        _df(), GROUP_1, GROUP_2, resample_method="subsample", n_iterations=37, random_state=1,
    )
    assert result["summary"]["n_iterations"] == 37
    assert result["summary"]["resample_method"] == "subsample"


def test_invalid_resample_method_raises():
    with pytest.raises(ValueError, match="resample_method must be one of"):
        compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="bootstrap")


def test_invalid_stability_threshold_raises():
    with pytest.raises(ValueError, match="stability_threshold must be in"):
        compute_resampling_stability(_df(), GROUP_1, GROUP_2, stability_threshold=0.5)
    with pytest.raises(ValueError, match="stability_threshold must be in"):
        compute_resampling_stability(_df(), GROUP_1, GROUP_2, stability_threshold=1.1)


def test_invalid_n_iterations_raises():
    with pytest.raises(ValueError, match="n_iterations must be a positive integer"):
        compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="subsample", n_iterations=0)
    with pytest.raises(ValueError, match="n_iterations must be a positive integer"):
        compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="subsample", n_iterations=-5)


def test_invalid_subsample_fraction_raises():
    with pytest.raises(ValueError, match="subsample_fraction must be in"):
        compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="subsample", subsample_fraction=0)
    with pytest.raises(ValueError, match="subsample_fraction must be in"):
        compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="subsample", subsample_fraction=1)


def test_subsample_fraction_leaving_too_few_samples_raises():
    df = _df()
    with pytest.raises(ValueError, match="fewer than 2 samples"):
        compute_resampling_stability(
            df, ["s1", "s2"], ["s4", "s5"], resample_method="subsample", subsample_fraction=0.4,
        )


def test_propagates_run_differential_expression_errors():
    df = _df()
    with pytest.raises(ValueError, match="not present in the count matrix"):
        compute_resampling_stability(df, ["s1", "nonexistent"], GROUP_2)


def test_new_metrics_do_not_change_existing_stability_semantics():
    # gene_stability gains two columns; is_stable/stable_genes/sensitive_genes
    # must come out byte-identical to before, since is_stable is still
    # computed only from frac_significant/stability_threshold.
    result = compute_resampling_stability(_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]
    assert list(stability.columns) == [
        "baseline_significant", "frac_significant", "direction_stability", "rank_stability", "is_stable",
    ]
    assert list(result["stable_genes"]) == ["flat", "strong"]
    assert list(result["sensitive_genes"]) == ["borderline"]


def _direction_df():
    # "hi"/"lo": uniform within each group -> log fold change is identical
    # in the baseline and every leave-one-out rerun (dropping one of three
    # identical values doesn't change the mean), so direction never moves.
    # "flip": baseline lfc is small and positive (0.1); dropping s3
    # specifically flips it negative (-0.9, since s3=6.0 was the only
    # group_1 value pulling the mean above group_2's 3.9); every other
    # drop keeps it positive. Hand-checked, not just run-and-observed.
    # "zero_baseline": both groups uniformly 5.0 -> log fold change is
    # exactly 0 in the baseline and every rerun -- no baseline direction
    # to compare against.
    return pd.DataFrame({
        "s1": [10.0, 0.0, 3.0, 5.0],
        "s2": [10.0, 0.0, 3.0, 5.0],
        "s3": [10.0, 0.0, 6.0, 5.0],
        "s4": [0.0, 10.0, 3.9, 5.0],
        "s5": [0.0, 10.0, 3.9, 5.0],
        "s6": [0.0, 10.0, 3.9, 5.0],
    }, index=["hi", "lo", "flip", "zero_baseline"])


def test_direction_stability_is_one_when_consistent():
    result = compute_resampling_stability(_direction_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]
    assert stability.loc["hi", "direction_stability"] == pytest.approx(1.0)
    assert stability.loc["lo", "direction_stability"] == pytest.approx(1.0)


def test_direction_stability_decreases_when_sign_flips():
    result = compute_resampling_stability(_direction_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]
    # 5 of 6 leave-one-out reruns keep the baseline's positive sign; only
    # dropping s3 flips it -- hand-derived, see _direction_df.
    assert stability.loc["flip", "direction_stability"] == pytest.approx(float(Fraction(5, 6)))
    assert stability.loc["flip", "direction_stability"] < stability.loc["hi", "direction_stability"]


def test_direction_stability_nan_when_baseline_lfc_is_zero():
    result = compute_resampling_stability(_direction_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]
    assert np.isnan(stability.loc["zero_baseline", "direction_stability"])


def test_direction_stability_denominator_excludes_invalid_reruns():
    # Direct unit test of the per-gene helper: real DE reruns essentially
    # never produce a non-finite log fold change from valid input, so this
    # exercises the documented "denominator = reruns with a finite value
    # for that gene" rule deterministically instead of trying to coax a
    # real DE call into producing NaN.
    baseline = pd.Series({"g1": 2.0, "g2": -1.0, "g3": 0.0})
    iteration_lfc = pd.DataFrame({
        0: {"g1": 1.0, "g2": -1.0, "g3": 5.0},
        1: {"g1": -1.0, "g2": float("nan"), "g3": 5.0},
        2: {"g1": float("nan"), "g2": 1.0, "g3": float("nan")},
    })
    result = _direction_stability(baseline, iteration_lfc)

    # g1: baseline +, valid reruns are cols 0 (+, match) and 1 (-, no
    # match); col 2 is NaN and excluded from the denominator -> 1/2.
    assert result["g1"] == pytest.approx(0.5)
    # g2: baseline -, valid reruns are cols 0 (-, match) and 2 (+, no
    # match); col 1 is NaN and excluded -> 1/2, not 1/3.
    assert result["g2"] == pytest.approx(0.5)
    # g3: baseline lfc is 0 -> no direction to compare against, NaN
    # regardless of what the reruns say.
    assert np.isnan(result["g3"])


def test_direction_stability_bounded_in_unit_interval():
    result = compute_resampling_stability(_direction_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    values = result["gene_stability"]["direction_stability"].dropna()
    assert not values.empty
    assert (values >= 0).all() and (values <= 1).all()


def _rank_df():
    # Six genes, all uniform within each group except "swap" -- dropping
    # any single sample from a uniform group leaves the mean unchanged, so
    # "anchor"/"bottom"/"stable_mid"/"top"/"ceiling" have the exact same
    # log fold change in the baseline and every leave-one-out rerun.
    # "anchor" (lfc -1000) and "ceiling" (lfc +1000) sit far outside
    # "swap"'s possible range (-20 to 19), so they stay rank 1 / rank 6 no
    # matter what "swap" does -- a clean "always perfectly stable" case.
    # "swap" (group_1 = [58, -20, -20], group_2 uniform 0.0) has baseline
    # lfc (58-20-20)/3 = 6.0, ranked between "stable_mid" (3.0) and "top"
    # (10.0); dropping s1 sends it to -20.0 (below "bottom"), dropping
    # s2 or s3 sends it to 19.0 (above "top"). Dropping s4/s5/s6 (group_2)
    # doesn't touch "swap"'s group_1 values, so those three reruns match
    # baseline exactly. All values hand-derived, not run-and-observed --
    # see the DEVLOG entry for the full per-iteration percentile table.
    return pd.DataFrame({
        "s1": [0.0, 0.0, 3.0, 58.0, 10.0, 1000.0],
        "s2": [0.0, 0.0, 3.0, -20.0, 10.0, 1000.0],
        "s3": [0.0, 0.0, 3.0, -20.0, 10.0, 1000.0],
        "s4": [1000.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        "s5": [1000.0, 10.0, 0.0, 0.0, 0.0, 0.0],
        "s6": [1000.0, 10.0, 0.0, 0.0, 0.0, 0.0],
    }, index=["anchor", "bottom", "stable_mid", "swap", "top", "ceiling"])


def test_rank_stability_is_one_when_rank_preserved():
    result = compute_resampling_stability(_rank_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]
    assert stability.loc["anchor", "rank_stability"] == pytest.approx(1.0)
    assert stability.loc["ceiling", "rank_stability"] == pytest.approx(1.0)


def test_rank_stability_decreases_when_rank_moves_substantially():
    result = compute_resampling_stability(_rank_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]
    # Hand-derived exact values (six six-gene percentile tables, one per
    # leave-one-out iteration): "swap" moves in 3 of 6 iterations and ends
    # up the least rank-stable gene; "bottom"/"stable_mid"/"top" each move
    # in at most 2 of 6 and are all more stable than "swap".
    assert stability.loc["swap", "rank_stability"] == pytest.approx(float(Fraction(13, 15)))
    assert stability.loc["bottom", "rank_stability"] == pytest.approx(float(Fraction(29, 30)))
    assert stability.loc["stable_mid", "rank_stability"] == pytest.approx(float(Fraction(29, 30)))
    assert stability.loc["top", "rank_stability"] == pytest.approx(float(Fraction(14, 15)))
    assert stability.loc["swap", "rank_stability"] < stability.loc["anchor", "rank_stability"]
    assert stability.loc["swap", "rank_stability"] < stability.loc["top", "rank_stability"]


def test_rank_stability_denominator_excludes_invalid_reruns():
    # Direct unit test of the per-gene helper, same rationale as the
    # direction_stability denominator test above.
    baseline = pd.Series({"g1": 1.0, "g2": 2.0, "g3": 3.0, "g4": 4.0})
    iteration_lfc = pd.DataFrame({
        0: {"g1": 1.0, "g2": 2.0, "g3": 3.0, "g4": 4.0},
        1: {"g1": 1.0, "g2": 2.0, "g3": 3.0, "g4": 4.0},
        2: {"g1": 1.0, "g2": 2.0, "g3": float("nan"), "g4": 4.0},
    })
    result = _rank_stability(baseline, iteration_lfc)

    # g3 is invalid in column 2 -- excluded from its own denominator
    # (averaged over 2 valid reruns, both perfectly rank-preserving), not
    # treated as a rank-preserving 3rd rerun and not corrupting to NaN.
    assert result["g3"] == pytest.approx(1.0)
    # g1 and g4 are valid and rank-preserved in every rerun, including
    # column 2 (still rank 1 of 3 valid genes there, same relative
    # position as rank 1 of 4 in the baseline).
    assert result["g1"] == pytest.approx(1.0)
    assert result["g4"] == pytest.approx(1.0)
    # g2 is valid in every rerun, but column 2's ranking universe shrank
    # to 3 genes (g3 dropped out), which moves g2's normalized rank
    # relative to the 4-gene baseline even though g2's own value didn't
    # change -- a real consequence of "rank_stability" measuring relative
    # position, not an invalid-rerun artifact.
    assert result["g2"] == pytest.approx(float(Fraction(17, 18)))


def test_rank_stability_nan_when_baseline_lfc_missing():
    baseline = pd.Series({"g1": 1.0, "g2": 2.0, "g3": float("nan")})
    iteration_lfc = pd.DataFrame({
        0: {"g1": 1.0, "g2": 2.0, "g3": 5.0},
        1: {"g1": 1.5, "g2": 2.5, "g3": 6.0},
    })
    result = _rank_stability(baseline, iteration_lfc)
    assert np.isnan(result["g3"])


def test_rank_stability_nan_when_fewer_than_two_rankable_genes():
    # rank_stability is a *relative* position among every other gene --
    # with fewer than 2 genes actually having a baseline value, there's
    # nothing to rank against, so the whole result is undefined regardless
    # of what the reruns look like.
    baseline = pd.Series({"g1": 1.0, "g2": float("nan"), "g3": float("nan")})
    iteration_lfc = pd.DataFrame({
        0: {"g1": 1.0, "g2": float("nan"), "g3": float("nan")},
        1: {"g1": 1.5, "g2": float("nan"), "g3": float("nan")},
    })

    result = _rank_stability(baseline, iteration_lfc)

    assert result.isna().all()


def test_rank_stability_bounded_in_unit_interval():
    result = compute_resampling_stability(_rank_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    values = result["gene_stability"]["rank_stability"].dropna()
    assert not values.empty
    assert (values >= 0).all() and (values <= 1).all()


def test_summary_mean_direction_and_rank_stability_use_baseline_significant_genes_only():
    result = compute_resampling_stability(_direction_df(), GROUP_1, GROUP_2, resample_method="leave_one_out")
    stability = result["gene_stability"]
    significant_genes = stability.index[stability["baseline_significant"]]

    # "hi"/"lo" have log fold change 10 and lfc_threshold defaults to 1.0,
    # so both clear the default significance threshold; "flip" (max |lfc|
    # 0.9) and "zero_baseline" (lfc 0) don't -- confirmed via the actual
    # baseline_significant column rather than assumed.
    assert set(significant_genes) == {"hi", "lo"}

    expected_mean_direction = stability.loc[significant_genes, "direction_stability"].mean()
    expected_mean_rank = stability.loc[significant_genes, "rank_stability"].mean()
    assert result["summary"]["mean_direction_stability"] == pytest.approx(expected_mean_direction)
    assert result["summary"]["mean_rank_stability"] == pytest.approx(expected_mean_rank)

    # And that mean is not silently the all-gene mean: "flip"'s
    # direction_stability (5/6) would pull the average down if it were
    # wrongly included.
    assert result["summary"]["mean_direction_stability"] == pytest.approx(1.0)
    all_gene_mean_direction = stability["direction_stability"].mean()
    assert result["summary"]["mean_direction_stability"] != pytest.approx(all_gene_mean_direction)
