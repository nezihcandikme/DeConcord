"""
Resampling stability: does a gene's significance call survive perturbing
*which samples* went into the analysis, holding the DE method, threshold,
and everything else fixed?

Method concordance, threshold sensitivity, and pathway stability all ask
some version of "does this conclusion survive a different reasonable
choice about how to analyze the data." This module asks a different
question: does it survive a different reasonable choice about *which
samples happened to be in the dataset*? A gene that's significant only
because one particular sample pulled the mean in one direction is a much
weaker finding than one that's significant no matter which subset of
samples you look at.

The approach: run the DE workflow once on the full sample set (the
baseline), then run it again many times on perturbed sample sets, and
track what fraction of those reruns still call each gene significant. A
gene significant in the baseline and in nearly every rerun is stable. A
gene significant in the baseline but only sometimes in the reruns is
exactly the kind of fragile finding this function exists to surface.

The same reruns also produce a log fold change per gene, whether or not
that gene was called significant, which is enough to answer two related
but distinct robustness questions without any extra DE calls:
``direction_stability`` (does the gene keep pointing the same way?) and
``rank_stability`` (does the gene keep roughly the same position relative
to every other gene?). See ``_direction_stability``/``_rank_stability``
below for the exact definitions.

Two resampling schemes are offered:

- ``"subsample"``: repeatedly draw a random subset of each group (without
  replacement) and rerun DE. Randomized, so results depend on
  ``random_state``; run enough iterations for the per-gene fractions to
  be meaningful.
- ``"leave_one_out"``: drop each sample exactly once and rerun DE.
  Deterministic and exhaustive, no ``random_state`` needed, but the
  number of iterations is fixed at the number of samples, and dropping a
  single sample from an already-small group is a much gentler
  perturbation than subsampling.

Bootstrap resampling (drawing samples *with* replacement) is deliberately
not offered. ``run_differential_expression`` refuses a group with a
repeated sample name -- a real, useful guardrail against a caller
accidentally reusing a sample -- and a bootstrap draw routinely produces
repeats. Supporting it would mean reworking how the DE functions index
into the expression matrix (by name today) to tolerate duplicates, which
is a bigger change than this module's scope justifies. Subsampling and
leave-one-out give the same "how much does the exact sample set matter"
signal without that rework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deconcord.differential_expression.methods import run_differential_expression

_RESAMPLE_METHODS = {"subsample", "leave_one_out"}


def _direction_stability(baseline_lfc: pd.Series, iteration_lfc: pd.DataFrame) -> pd.Series:
    """
    Per gene: fraction of reruns whose log fold change has the same sign
    as the baseline log fold change, restricted to reruns where that
    gene's rerun log fold change is a finite number (the denominator).
    Pure sign comparison -- no p-value or significance threshold involved,
    so this stays distinct from frac_significant/is_stable.

    NaN if the baseline log fold change is exactly 0 (no baseline
    direction to compare against), if it's missing, or if the gene has no
    rerun with a finite log fold change.
    """
    baseline_sign = np.sign(baseline_lfc)
    valid = iteration_lfc.notna()
    matches = np.sign(iteration_lfc).eq(baseline_sign, axis=0) & valid

    n_valid = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        result = matches.sum(axis=1) / n_valid
    result = result.where(n_valid > 0)
    result = result.where(baseline_lfc.notna() & (baseline_sign != 0))
    return result


def _rank_stability(baseline_lfc: pd.Series, iteration_lfc: pd.DataFrame) -> pd.Series:
    """
    Per gene: 1 minus the mean absolute difference between the gene's
    normalized rank (percentile, by signed log fold change, ties broken
    by average rank) in the baseline and its normalized rank in each
    rerun, restricted to reruns where that gene's rerun log fold change
    is a finite number (the denominator). Ranking on the signed log fold
    change -- not p-value or significance -- measures relative position
    among all genes, a different property from direction_stability's
    per-gene sign check or frac_significant's per-gene call rate.

    Percentile = (rank - 1) / (n_valid_genes - 1) within a given run, so
    0 = lowest log fold change in that run, 1 = highest. Bounded to
    [0, 1]; higher means the gene held a more consistent relative
    position. NaN if there are fewer than 2 genes to rank, if the
    baseline log fold change is missing, or if the gene has no rerun with
    a finite log fold change.
    """
    n_genes = baseline_lfc.notna().sum()
    if n_genes < 2:
        return pd.Series(float("nan"), index=baseline_lfc.index)

    baseline_pct = (baseline_lfc.rank(method="average", na_option="keep") - 1) / (n_genes - 1)

    iter_n_valid = iteration_lfc.notna().sum(axis=0)
    iter_ranks = iteration_lfc.rank(method="average", na_option="keep", axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        iter_pct = (iter_ranks - 1).div(iter_n_valid - 1, axis=1)
    iter_pct = iter_pct.where(np.isfinite(iter_pct))

    own_valid = iteration_lfc.notna()
    abs_diff = iter_pct.sub(baseline_pct, axis=0).abs().where(own_valid & iter_pct.notna())

    n_effective = abs_diff.notna().sum(axis=1)
    result = 1 - abs_diff.mean(axis=1, skipna=True)
    result = result.where(n_effective > 0)
    result = result.where(baseline_lfc.notna())
    return result


def compute_resampling_stability(
    df: pd.DataFrame,
    group_1: list[str],
    group_2: list[str],
    resample_method: str = "subsample",
    n_iterations: int = 200,
    subsample_fraction: float = 0.8,
    alpha: float = 0.05,
    lfc_threshold: float = 1.0,
    de_method: str = "welch",
    stability_threshold: float = 0.9,
    random_state: int | None = None,
) -> dict:
    """
    Rerun differential expression on perturbed sample sets and measure how
    often each gene's significance call reappears.

    Parameters
    ----------
    df : pd.DataFrame
        Log-transformed, normalized expression matrix (genes as rows,
        samples as columns), the same input ``run_differential_expression``
        expects.
    group_1, group_2 : list[str]
        Sample names (column names in ``df``) for each comparison group,
        for the baseline run and as the pool each resampling iteration
        draws from.
    resample_method : str, optional
        ``"subsample"`` (default) or ``"leave_one_out"``. See the module
        docstring for what each does.
    n_iterations : int, optional
        Number of resampling iterations. Only used for
        ``resample_method="subsample"``; ignored (and set to
        ``len(group_1) + len(group_2)``) for ``"leave_one_out"``. Default
        200.
    subsample_fraction : float, optional
        Fraction of each group kept in every subsample draw, rounded to
        the nearest integer sample count. Default 0.8. Must leave at
        least 2 samples in each group, the minimum
        ``run_differential_expression`` needs to run a per-gene test.
        Only used for ``resample_method="subsample"``.
    alpha, lfc_threshold, de_method
        Passed straight through to every ``run_differential_expression``
        call, baseline and resampled alike. See its docstring for what
        each means.
    stability_threshold : float, optional
        How consistently a gene's baseline call must reappear to count as
        stable. A baseline-significant gene is stable if it's significant
        in at least this fraction of reruns; a baseline-non-significant
        gene is stable if it's significant in at most ``1 -
        stability_threshold`` of reruns. Default 0.9. Must be in
        (0.5, 1.0] -- below 0.5 "stable" would stop meaning "reappears
        most of the time."
    random_state : int, optional
        Seed for the random subsampling draws. Ignored for
        ``"leave_one_out"``, which has no randomness. Default ``None``
        (non-reproducible draws).

    Returns
    -------
    dict
        - ``"summary"``: a flat dict with ``resample_method``,
          ``n_iterations`` (actual count run), ``n_baseline_significant``,
          ``mean_jaccard_to_baseline`` (mean, across iterations, of the
          Jaccard index between that iteration's significant-gene set and
          the baseline's; ``NaN`` for an iteration where both sets are
          empty), ``baseline_replication_rate`` (mean fraction of reruns
          in which a baseline-significant gene stayed significant; ``NaN``
          if the baseline had no significant genes), and
          ``mean_direction_stability``/``mean_rank_stability`` (mean of
          each gene-level column below, over baseline-significant genes
          only, same convention as ``baseline_replication_rate``; ``NaN``
          if the baseline had no significant genes).
        - ``"gene_stability"``: a DataFrame indexed by gene ID with:

          - ``baseline_significant`` (bool)
          - ``frac_significant``: fraction of reruns calling the gene
            significant.
          - ``direction_stability``: fraction of reruns whose log fold
            change has the same sign as the baseline's, restricted to
            reruns where that gene's rerun log fold change is a finite
            number (the denominator -- see ``_direction_stability``).
            Sign-only, no significance criterion, so it's a different
            question from ``frac_significant``. ``NaN`` if the baseline
            log fold change is 0 or missing, or the gene has no valid
            rerun.
          - ``rank_stability``: how consistently the gene holds the same
            relative position (by signed log fold change, among every
            gene) across reruns versus the baseline, one minus the mean
            per-rerun normalized-rank difference, restricted the same way
            (see ``_rank_stability``). ``1.0`` = same relative position
            every time, ``0.0`` = maximally different on average. ``NaN``
            under the same conditions as ``direction_stability``, or if
            there are fewer than 2 genes to rank.
          - ``is_stable``: unchanged from before -- still derived only
            from ``frac_significant`` and ``stability_threshold``, not
            from either new column.
        - ``"stable_genes"``, ``"sensitive_genes"``: gene IDs split by
          ``is_stable``.
        - ``"baseline_results"``: the full-data
          ``run_differential_expression`` output, for reference.

    Raises
    ------
    ValueError
        If ``resample_method`` isn't ``"subsample"`` or ``"leave_one_out"``,
        ``n_iterations`` isn't a positive integer, ``subsample_fraction``
        isn't in (0, 1), ``stability_threshold`` isn't in (0.5, 1.0], a
        group would drop below 2 samples after subsampling, or (from
        ``run_differential_expression`` itself, on the baseline call)
        ``alpha``/``lfc_threshold``/``de_method`` are invalid or a group
        is empty, has a duplicate sample, or references a sample not in
        ``df``.
    """
    if resample_method not in _RESAMPLE_METHODS:
        raise ValueError(f"resample_method must be one of {sorted(_RESAMPLE_METHODS)}; got {resample_method!r}.")
    if not (0.5 < stability_threshold <= 1.0):
        raise ValueError(f"stability_threshold must be in (0.5, 1.0]; got {stability_threshold}.")
    if resample_method == "subsample":
        if not isinstance(n_iterations, int) or n_iterations <= 0:
            raise ValueError(f"n_iterations must be a positive integer; got {n_iterations!r}.")
        if not (0 < subsample_fraction < 1):
            raise ValueError(f"subsample_fraction must be in (0, 1); got {subsample_fraction}.")

    # The baseline call also validates alpha/lfc_threshold/de_method and
    # group_1/group_2 themselves, so no need to duplicate those checks here.
    baseline_results = run_differential_expression(
        df, group_1, group_2, alpha=alpha, lfc_threshold=lfc_threshold, method=de_method,
    )
    baseline_significant = baseline_results["significant"]
    gene_index = baseline_results.index

    rng = np.random.default_rng(random_state)
    sig_counts = pd.Series(0, index=gene_index, dtype=int)
    jaccards = []

    if resample_method == "subsample":
        size_1 = round(subsample_fraction * len(group_1))
        size_2 = round(subsample_fraction * len(group_2))
        if size_1 < 2 or size_2 < 2:
            raise ValueError(
                "subsample_fraction leaves fewer than 2 samples in a group after "
                f"rounding (group_1: {len(group_1)} -> {size_1}, group_2: {len(group_2)} -> {size_2}). "
                "Use a larger subsample_fraction or a larger group."
            )

        iterations = []
        for _ in range(n_iterations):
            sample_1 = list(rng.choice(group_1, size=size_1, replace=False))
            sample_2 = list(rng.choice(group_2, size=size_2, replace=False))
            iterations.append((sample_1, sample_2))
        actual_n_iterations = n_iterations
    else:
        # Leave-one-out: drop each sample exactly once. Guard against a
        # group that only has 2 samples, since dropping one would leave a
        # single-sample group and run_differential_expression requires 2.
        if len(group_1) < 3 or len(group_2) < 3:
            raise ValueError(
                "leave_one_out needs at least 3 samples in each group (so dropping "
                f"one still leaves 2); got group_1={len(group_1)}, group_2={len(group_2)}."
            )
        iterations = []
        for sample in group_1:
            iterations.append(([s for s in group_1 if s != sample], list(group_2)))
        for sample in group_2:
            iterations.append((list(group_1), [s for s in group_2 if s != sample]))
        actual_n_iterations = len(iterations)

    baseline_sig_set = set(gene_index[baseline_significant])
    lfc_iterations = []

    for sample_1, sample_2 in iterations:
        result = run_differential_expression(
            df, sample_1, sample_2, alpha=alpha, lfc_threshold=lfc_threshold, method=de_method,
        )
        sig = result["significant"].reindex(gene_index, fill_value=False)
        sig_counts = sig_counts.add(sig.astype(int), fill_value=0)

        iter_sig_set = set(gene_index[sig])
        union = baseline_sig_set | iter_sig_set
        jaccards.append(len(baseline_sig_set & iter_sig_set) / len(union) if union else float("nan"))

        # Same rerun, no extra DE call: log_fold_change is already part of
        # `result`, just unused until now. Reindexed the same way `sig` is,
        # so a gene missing from a given rerun (shouldn't happen today, but
        # kept consistent) is NaN rather than silently misaligned.
        lfc_iterations.append(result["log_fold_change"].reindex(gene_index))

    frac_significant = sig_counts / actual_n_iterations
    iteration_lfc = pd.concat(lfc_iterations, axis=1, ignore_index=True)
    baseline_lfc = baseline_results["log_fold_change"]

    gene_stability = pd.DataFrame({
        "baseline_significant": baseline_significant,
        "frac_significant": frac_significant,
        "direction_stability": _direction_stability(baseline_lfc, iteration_lfc),
        "rank_stability": _rank_stability(baseline_lfc, iteration_lfc),
    })
    is_stable = np.where(
        gene_stability["baseline_significant"],
        gene_stability["frac_significant"] >= stability_threshold,
        gene_stability["frac_significant"] <= (1 - stability_threshold),
    )
    gene_stability["is_stable"] = is_stable

    if len(baseline_sig_set):
        baseline_sig_genes = list(baseline_sig_set)
        baseline_replication_rate = float(gene_stability.loc[baseline_sig_genes, "frac_significant"].mean())
        mean_direction_stability = float(gene_stability.loc[baseline_sig_genes, "direction_stability"].mean())
        mean_rank_stability = float(gene_stability.loc[baseline_sig_genes, "rank_stability"].mean())
    else:
        baseline_replication_rate = float("nan")
        mean_direction_stability = float("nan")
        mean_rank_stability = float("nan")

    summary = {
        "resample_method": resample_method,
        "n_iterations": actual_n_iterations,
        "n_baseline_significant": len(baseline_sig_set),
        # Every iteration's jaccard is NaN when the baseline (and every
        # resampled rerun) has zero significant genes -- a real outcome for
        # a very stringent alpha or a genuinely null dataset, not a bug.
        # np.nanmean on an all-NaN array still returns NaN correctly, but
        # emits a "Mean of empty slice" RuntimeWarning along the way; that
        # warning would otherwise leak out to anyone running a real
        # analysis that legitimately finds nothing significant, so it's
        # checked for explicitly instead of suppressed blindly.
        "mean_jaccard_to_baseline": (
            float("nan") if not jaccards or all(np.isnan(j) for j in jaccards)
            else float(np.nanmean(jaccards))
        ),
        "baseline_replication_rate": baseline_replication_rate,
        "mean_direction_stability": mean_direction_stability,
        "mean_rank_stability": mean_rank_stability,
    }

    return {
        "summary": summary,
        "gene_stability": gene_stability,
        "stable_genes": pd.Index(sorted(gene_stability.index[gene_stability["is_stable"]])),
        "sensitive_genes": pd.Index(sorted(gene_stability.index[~gene_stability["is_stable"]])),
        "baseline_results": baseline_results,
    }
