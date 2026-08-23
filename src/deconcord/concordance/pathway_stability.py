"""
Pathway stability: does an enriched pathway stay enriched across
different analytical choices, or is the pathway-level conclusion itself
method-sensitive?

Method concordance (deconcord.concordance.methods.compute_de_concordance)
asks this question at the gene level. This module asks the same kind of
question one level up: given two pathway enrichment result tables (for
example, one from a DE run using Welch's t-test and another from the
same data using the moderated t-test, or one from DESeq2's significant
genes and one from edgeR's), which pathways show up as enriched in both,
and which are only enriched under one of the two choices that produced
the input gene lists?

This is deliberately simpler than gene-level concordance. Pathway
enrichment tables don't have a natural continuous "effect size" that's
comparable across sources the way log fold change is comparable across
DE tables: local enrichment reports an overlap_count against whatever
background you supplied, live g:Profiler reports an intersection_size
against its own background, and the two aren't on the same scale. So
this only compares presence/absence of significance, the same
overlap-and-Jaccard piece of compute_de_concordance, without a
directional-agreement or correlation component, since neither has a
well-defined meaning here.
"""

from __future__ import annotations

import warnings

import pandas as pd


def compute_pathway_stability(
    enrichment_a: pd.DataFrame,
    enrichment_b: pd.DataFrame,
    name_a: str = "method_a",
    name_b: str = "method_b",
    pathway_col_a: str = "pathway",
    pvalue_col_a: str = "adjusted_p_value",
    pathway_col_b: str = "pathway",
    pvalue_col_b: str = "adjusted_p_value",
    alpha: float = 0.05,
) -> dict:
    """
    Compare two pathway enrichment result tables and report which
    pathways are enriched in both, and which are specific to one.

    Parameters
    ----------
    enrichment_a, enrichment_b : pd.DataFrame
        Pathway enrichment result tables, e.g. the output of
        ``run_pathway_enrichment_analysis`` (columns ``pathway``,
        ``adjusted_p_value``, the defaults here) or
        ``run_gprofiler_enrichment`` (columns ``name``, ``p_value``,
        pass ``pathway_col_b="name", pvalue_col_b="p_value"``).
    name_a, name_b : str, optional
        Labels for each table, used in the returned summary dict's keys.
        Defaults ``"method_a"``/``"method_b"``.
    pathway_col_a, pvalue_col_a : str, optional
        Column names to use from ``enrichment_a``. Defaults
        ``"pathway"``/``"adjusted_p_value"``.
    pathway_col_b, pvalue_col_b : str, optional
        Same, for ``enrichment_b``.
    alpha : float, optional
        Adjusted (or raw, if that's what was passed as ``pvalue_col_*``)
        p-value cutoff used to call a pathway significant in each table.
        Default 0.05.

    Returns
    -------
    dict
        - ``"summary"``: a flat dict with ``pathways_compared`` (pathways
          tested in both tables), ``significant_<name_a>``,
          ``significant_<name_b>``, ``significant_in_both``, and
          ``jaccard_index`` (significant-pathway set overlap, ``NaN`` if
          neither table calls anything significant).
        - ``"stable_pathways"``: pathway names significant in both
          tables, the conclusions that survive the choice between
          whatever produced ``enrichment_a`` and ``enrichment_b``.
        - ``"only_in_<name_a>"``, ``"only_in_<name_b>"``: pathway names
          significant in one table but not the other.
        - ``"merged"``: the full per-pathway comparison table for every
          pathway tested in both.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1], either table has a duplicate
        pathway name, a required column is missing from either table, or
        no pathway was tested in both tables.

    Warns
    -----
    UserWarning
        If the matched pathway count is less than half of the smaller
        table's own pathway count — this can be a real result (different
        gene-set collections legitimately share few names), but it's also
        exactly what a naming-convention mismatch between sources looks
        like, so it's surfaced rather than left silent.
    """
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}.")

    for label, df, pathway_col, pvalue_col in (
        (name_a, enrichment_a, pathway_col_a, pvalue_col_a),
        (name_b, enrichment_b, pathway_col_b, pvalue_col_b),
    ):
        missing = [c for c in (pathway_col, pvalue_col) if c not in df.columns]
        if missing:
            raise ValueError(f"'{label}' is missing column(s) {missing}.")
        if df[pathway_col].duplicated().any():
            dupes = df.loc[df[pathway_col].duplicated(), pathway_col].unique().tolist()
            raise ValueError(f"'{label}' has duplicate pathway names in '{pathway_col}': {dupes[:10]}.")

    a = enrichment_a.set_index(pathway_col_a)[[pvalue_col_a]].rename(columns={pvalue_col_a: "padj"})
    b = enrichment_b.set_index(pathway_col_b)[[pvalue_col_b]].rename(columns={pvalue_col_b: "padj"})

    merged = a.join(b, how="inner", lsuffix=f"_{name_a}", rsuffix=f"_{name_b}")

    # Same idea as compute_de_concordance's gene-ID overlap check: a low
    # matched-pathway count can be a real, legitimate result (two truly
    # different gene-set collections, e.g. a narrow in-house GMT file
    # against g:Profiler's much broader databases, genuinely don't share
    # many pathway names) rather than a mistake -- unlike the gene-ID
    # case, there's no "should be 90%+" baseline to lean on here. Still
    # worth a nudge rather than silence, since the *other* common cause
    # (same underlying pathway, different naming convention between
    # sources -- "HALLMARK_TNFA_SIGNALING_VIA_NFKB" vs. "TNF-alpha
    # signaling via NF-kB") is a data problem, not a finding, and looks
    # identical to a genuine low-overlap result unless someone checks.
    smaller_size = min(len(a), len(b))
    if 0 < len(merged) < 0.5 * smaller_size:
        warnings.warn(
            f"Only {len(merged)} of {smaller_size} pathway names in the "
            f"smaller table matched between '{name_a}' ({len(a)} pathways) "
            f"and '{name_b}' ({len(b)} pathways). This can be a genuine "
            "result if the two tables come from different gene-set "
            "collections, but it's also exactly what a naming-convention "
            "mismatch between sources looks like (e.g. a MSigDB-style name "
            "vs. a g:Profiler term name for the same pathway) -- spot-check "
            "a few names from each table before trusting the numbers below.",
            stacklevel=2,
        )

    padj_a_col, padj_b_col = f"padj_{name_a}", f"padj_{name_b}"
    merged = merged.dropna(subset=[padj_a_col, padj_b_col])

    if merged.empty:
        raise ValueError(
            "No pathway was tested (with a non-missing p-value) in both "
            f"'{name_a}' and '{name_b}' -- nothing to compare. This usually "
            "means the two enrichment runs used different gene set "
            "collections or pathway naming."
        )

    sig_a = set(merged.index[merged[padj_a_col] < alpha])
    sig_b = set(merged.index[merged[padj_b_col] < alpha])
    sig_in_both = sig_a & sig_b
    sig_in_either = sig_a | sig_b

    stable = pd.Index(sorted(sig_in_both))
    only_in_a = pd.Index(sorted(sig_a - sig_b))
    only_in_b = pd.Index(sorted(sig_b - sig_a))

    jaccard_index = len(sig_in_both) / len(sig_in_either) if sig_in_either else float("nan")

    summary = {
        "pathways_compared": len(merged),
        f"significant_{name_a}": len(sig_a),
        f"significant_{name_b}": len(sig_b),
        "significant_in_both": len(sig_in_both),
        "jaccard_index": jaccard_index,
    }

    merged_out = merged.rename(columns={
        padj_a_col: f"adjusted_p_value_{name_a}", padj_b_col: f"adjusted_p_value_{name_b}",
    })
    merged_out[f"significant_{name_a}"] = merged_out.index.isin(sig_a)
    merged_out[f"significant_{name_b}"] = merged_out.index.isin(sig_b)

    return {
        "summary": summary,
        "stable_pathways": stable,
        f"only_in_{name_a}": only_in_a,
        f"only_in_{name_b}": only_in_b,
        "merged": merged_out,
    }
