"""
Threshold sensitivity: does a concordance conclusion survive a small,
reasonable change to the significance threshold?

Method concordance (deconcord.concordance.methods.compute_de_concordance)
answers "how much do these two DE result tables agree" at one fixed alpha,
0.05 by default. That single number hides a real question: is the
agreement (or disagreement) between the two tables a stable property of
the data, or does it depend on having picked exactly that cutoff? A gene
called significant in both tables at alpha=0.05 that stops being
significant in one of them at alpha=0.04 was never a solid finding, it
was one arbitrary threshold choice away from disappearing.

This module answers that by sweeping compute_de_concordance across a
range of alpha values and reporting two things: how the aggregate
concordance metrics (Jaccard index, significant-gene counts, directional
agreement) move as alpha moves, and, per gene, what fraction of the swept
thresholds call it significant in each table. A gene significant at every
threshold in the range (fraction 1.0) or none of them (fraction 0.0) is
threshold-stable. Anything in between is threshold-sensitive: whether it
counts as a "finding" depends on which of several defensible thresholds
you happened to pick.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from deconcord.concordance.methods import compute_de_concordance


def compute_threshold_sensitivity(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    # Sequence, not list: the default is a tuple on purpose (an immutable
    # default avoids the classic mutable-default-argument footgun), and the
    # body immediately does alphas = list(alphas), so any ordered iterable
    # of floats -- tuple, list, or otherwise -- is genuinely accepted here.
    alphas: Sequence[float] = (0.01, 0.05, 0.1),
    name_a: str = "method_a",
    name_b: str = "method_b",
    lfc_col_a: str = "log_fold_change",
    pvalue_col_a: str = "adjusted_p_value",
    lfc_col_b: str = "log_fold_change",
    pvalue_col_b: str = "adjusted_p_value",
) -> dict:
    """
    Sweep method concordance across a range of significance thresholds.

    Runs ``compute_de_concordance`` once per value in ``alphas`` and
    collects the results, so you can see whether the agreement between
    ``results_a`` and ``results_b`` holds up across a reasonable range of
    cutoffs or is an artifact of the particular alpha checked.

    Parameters
    ----------
    results_a, results_b : pd.DataFrame
        DE result tables, indexed by gene ID. Same requirements as
        ``compute_de_concordance``.
    alphas : list of float, optional
        The significance thresholds to sweep. Default ``(0.01, 0.05, 0.1)``,
        a small range around the conventional 0.05. Must be non-empty,
        with every value in (0, 1] and no duplicates.
    name_a, name_b, lfc_col_a, pvalue_col_a, lfc_col_b, pvalue_col_b
        Passed straight through to ``compute_de_concordance`` at each
        alpha. See its docstring for what each one means.

    Returns
    -------
    dict
        - ``"by_alpha"``: a DataFrame indexed by alpha, one row per
          threshold swept, with columns ``genes_compared``,
          ``significant_<name_a>``, ``significant_<name_b>``,
          ``significant_in_both``, ``jaccard_index``, and
          ``directional_agreement``. ``pearson_r_lfc``/``spearman_r_lfc``
          are not included here since they don't depend on alpha at all
          (they're computed across every compared gene, not just the
          significant ones) and would just repeat the same number on
          every row.
        - ``"gene_stability"``: a DataFrame indexed by gene ID (every
          gene compared, from the widest alpha's merged table) with
          ``frac_significant_<name_a>`` and ``frac_significant_<name_b>``:
          the fraction of swept alphas at which that gene was significant
          in each table. 1.0 means significant at every threshold tried,
          0.0 means significant at none. Anything in between means the
          call flips somewhere in the range, which is exactly the
          threshold-sensitive case this function exists to surface.
        - ``"stable_genes"``: gene IDs where both fractions are 0.0 or
          1.0 in both tables, i.e. the significance call in *both*
          tables is unaffected by which alpha in the range was used.
        - ``"sensitive_genes"``: gene IDs not in ``stable_genes``, the
          ones whose concordance status depends on the exact threshold.

    Raises
    ------
    ValueError
        If ``alphas`` is empty, contains a duplicate, or contains a value
        outside (0, 1]. Any error ``compute_de_concordance`` itself would
        raise (duplicate gene IDs, missing columns, no gene overlap) is
        raised from the first alpha it's hit on.
    """
    alphas = list(alphas)
    if not alphas:
        raise ValueError("alphas must not be empty.")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"alphas must not contain duplicates; got {alphas}.")
    for a in alphas:
        if not (0 < a <= 1):
            raise ValueError(f"Every value in alphas must be in (0, 1]; got {a}.")

    sorted_alphas = sorted(alphas)
    rows = []
    per_alpha_merged = {}

    for alpha in sorted_alphas:
        result = compute_de_concordance(
            results_a, results_b, name_a=name_a, name_b=name_b,
            lfc_col_a=lfc_col_a, pvalue_col_a=pvalue_col_a,
            lfc_col_b=lfc_col_b, pvalue_col_b=pvalue_col_b,
            alpha=alpha,
        )
        summary = result["summary"]
        rows.append({
            "alpha": alpha,
            "genes_compared": summary["genes_compared"],
            f"significant_{name_a}": summary[f"significant_{name_a}"],
            f"significant_{name_b}": summary[f"significant_{name_b}"],
            "significant_in_both": summary["significant_in_both"],
            "jaccard_index": summary["jaccard_index"],
            "directional_agreement": summary["directional_agreement"],
        })
        per_alpha_merged[alpha] = result["merged"]

    by_alpha = pd.DataFrame(rows).set_index("alpha")

    # Gene-level stability: how many of the swept alphas call each gene
    # significant in each table. Every alpha's merged table has the same
    # gene set (compute_de_concordance's own intersection/dropna logic
    # doesn't depend on alpha), so any of them can supply the index.
    sig_a_col = f"significant_{name_a}"
    sig_b_col = f"significant_{name_b}"
    reference_index = per_alpha_merged[sorted_alphas[0]].index

    sig_a_counts = pd.Series(0, index=reference_index, dtype=int)
    sig_b_counts = pd.Series(0, index=reference_index, dtype=int)
    for alpha in sorted_alphas:
        merged = per_alpha_merged[alpha]
        sig_a_counts = sig_a_counts.add(merged[sig_a_col].astype(int), fill_value=0)
        sig_b_counts = sig_b_counts.add(merged[sig_b_col].astype(int), fill_value=0)

    n_alphas = len(sorted_alphas)
    gene_stability = pd.DataFrame({
        f"frac_significant_{name_a}": sig_a_counts / n_alphas,
        f"frac_significant_{name_b}": sig_b_counts / n_alphas,
    })

    is_stable = gene_stability[f"frac_significant_{name_a}"].isin([0.0, 1.0]) & \
        gene_stability[f"frac_significant_{name_b}"].isin([0.0, 1.0])

    return {
        "by_alpha": by_alpha,
        "gene_stability": gene_stability,
        "stable_genes": pd.Index(sorted(gene_stability.index[is_stable])),
        "sensitive_genes": pd.Index(sorted(gene_stability.index[~is_stable])),
    }
