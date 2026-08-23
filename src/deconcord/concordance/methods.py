"""
Method concordance -- do two differential expression results agree?

This is DEConcord's foundational question: given two DE result tables for
the same comparison (e.g. DESeq2 and edgeR run on the same count matrix, or
any other pair -- two DE tools, two DE methods, two preprocessing choices),
how much of the biological conclusion actually survives the choice of
method? A gene calling itself "significant" under one tool and silent under
another isn't automatically a bug in either tool; it's information about
how fragile that particular conclusion is.

Nothing here runs DESeq2 or edgeR itself -- both are R packages, and
reimplementing them would be a different (and much larger, much riskier)
project than the one DEConcord is actually about. Instead, this module
consumes DE result tables that some other tool already produced (a CSV with
a gene ID, a log fold change, and a p-value is a close-to-universal DE
output shape) and compares them. `benchmarks/run_deseq2_edger.R` is exactly
this: it runs real DESeq2 and real edgeR and writes each one's results to
CSV, which is what this module was actually validated against.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_de_concordance(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    name_a: str = "method_a",
    name_b: str = "method_b",
    lfc_col_a: str = "log_fold_change",
    pvalue_col_a: str = "adjusted_p_value",
    lfc_col_b: str = "log_fold_change",
    pvalue_col_b: str = "adjusted_p_value",
    alpha: float = 0.05,
) -> dict:
    """
    Compare two differential expression result tables for the same
    comparison and report where they agree and where they don't.

    Both tables are expected to be indexed by gene ID, with a log fold
    change column and a (multiple-testing-corrected) p-value column each --
    the column names are configurable because DESeq2 (`log2FoldChange`,
    `padj`), edgeR (`logFC`, `FDR`), and DEConcord's own DE output
    (`log_fold_change`, `adjusted_p_value`) all use different ones, and
    none of them should have to be renamed just to be compared.

    Parameters
    ----------
    results_a, results_b : pd.DataFrame
        DE result tables, indexed by gene ID.
    name_a, name_b : str, optional
        Labels for each table, used only in the returned summary dict's
        keys/values (e.g. "significant_deseq2"). Defaults "method_a"/
        "method_b".
    lfc_col_a, pvalue_col_a : str, optional
        Column names to use from ``results_a``. Default
        ``"log_fold_change"``/``"adjusted_p_value"`` (DEConcord's own DE
        output shape). Use ``"log2FoldChange"``/``"padj"`` for a raw DESeq2
        table, or ``"logFC"``/``"FDR"`` for a raw edgeR table.
    lfc_col_b, pvalue_col_b : str, optional
        Same, for ``results_b``.
    alpha : float, optional
        Adjusted p-value cutoff used to call a gene significant in each
        table. Default 0.05. This is one fixed threshold -- see
        ``deconcord.concordance`` roadmap notes for evaluating a range of
        thresholds instead of a single one.

    Returns
    -------
    dict
        - ``"summary"``: a flat dict of scalar metrics --
          ``genes_compared`` (genes with a non-missing value in both
          tables), ``significant_<name_a>``, ``significant_<name_b>``,
          ``significant_in_both``, ``jaccard_index`` (significant-gene set
          overlap), ``pearson_r_lfc``/``spearman_r_lfc`` (log fold change
          agreement across every compared gene, not just significant ones),
          and ``directional_agreement`` (among genes significant in *both*,
          the fraction where the log fold change has the same sign --
          ``NaN`` if no gene is significant in both).
        - ``"concordant_genes"``: genes significant in both tables with the
          same-signed log fold change -- the conclusions that survive the
          choice of method.
        - ``"discordant_genes"``: genes significant in both tables but with
          opposite-signed log fold change -- rare in practice, and worth
          looking at individually when it happens, since it means two
          methods actively disagree on the direction of an effect, not
          just its significance.
        - ``"only_in_<name_a>"``, ``"only_in_<name_b>"``: genes significant
          in one table but not the other -- method-dependent conclusions,
          neither confirmed nor contradicted by the other method.
        - ``"merged"``: the full per-gene comparison table (both log fold
          changes, both p-values, both significance calls) for every gene
          compared, for anything not covered by the summary above.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1], either table's index has duplicate
        gene IDs, a required column is missing from either table, or no
        gene has a non-missing log fold change and p-value in both tables
        (nothing to compare -- usually a sign that the two tables weren't
        run on the same gene set at all).

    Warns
    -----
    UserWarning
        If the matched gene count is less than half of the smaller table's
        own gene count -- a much more likely sign of an ID-format mismatch
        between the two tables (Ensembl IDs with vs. without a version
        suffix, Ensembl IDs vs. gene symbols) than of genuinely different
        genes being tested. Not raised as an error, since a real, large
        biological difference in what was tested is possible too -- just
        surfaced loudly enough to check before trusting the numbers below.
    """
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}.")

    for label, df, lfc_col, pvalue_col in (
        (name_a, results_a, lfc_col_a, pvalue_col_a),
        (name_b, results_b, lfc_col_b, pvalue_col_b),
    ):
        if df.index.duplicated().any():
            dupes = df.index[df.index.duplicated()].unique().tolist()
            raise ValueError(f"'{label}' has duplicate gene IDs in its index: {dupes[:10]}.")
        missing = [c for c in (lfc_col, pvalue_col) if c not in df.columns]
        if missing:
            raise ValueError(f"'{label}' is missing column(s) {missing}.")

    a = results_a[[lfc_col_a, pvalue_col_a]].copy()
    a.columns = ["lfc", "padj"]
    b = results_b[[lfc_col_b, pvalue_col_b]].copy()
    b.columns = ["lfc", "padj"]

    merged = a.join(b, how="inner", lsuffix=f"_{name_a}", rsuffix=f"_{name_b}")

    # A real ID-format mismatch between the two tables (Ensembl IDs with a
    # version suffix in one and without in the other, e.g. "ENSG...1" vs
    # "ENSG...", or Ensembl IDs in one and gene symbols in the other) joins
    # on the index just like a real biological difference would: silently,
    # with no error, just a smaller merged table. A *total* mismatch (zero
    # overlap) is already caught below, but a *partial* one isn't -- it
    # just looks like a slightly lower genes_compared/jaccard_index, which
    # is easy to misread as a real finding about the two methods rather
    # than a data-formatting problem. Two well-annotated tools run on the
    # same real dataset routinely match on 90%+ of the smaller table's
    # genes (see the README's Validation numbers); a match well below that
    # is a much stronger signal of an ID mismatch than of genuine
    # biological non-overlap, so it's worth a loud nudge rather than
    # silence.
    smaller_size = min(len(a), len(b))
    if 0 < len(merged) < 0.5 * smaller_size:
        warnings.warn(
            f"Only {len(merged)} of {smaller_size} gene IDs in the smaller "
            f"table matched between '{name_a}' ({len(a)} genes) and "
            f"'{name_b}' ({len(b)} genes). This usually means the two "
            "tables use different gene ID conventions (e.g. Ensembl IDs "
            "with vs. without a version suffix, or Ensembl IDs vs. gene "
            "symbols) rather than a real biological difference in what was "
            "tested -- spot-check a few IDs from each table before "
            "trusting the concordance numbers below.",
            stacklevel=2,
        )

    lfc_a_col, padj_a_col = f"lfc_{name_a}", f"padj_{name_a}"
    lfc_b_col, padj_b_col = f"lfc_{name_b}", f"padj_{name_b}"
    merged = merged.dropna(subset=[lfc_a_col, padj_a_col, lfc_b_col, padj_b_col])

    if merged.empty:
        raise ValueError(
            "No gene has a non-missing log fold change and p-value in both "
            f"'{name_a}' and '{name_b}' -- nothing to compare. This usually "
            "means the two tables weren't run on the same gene set."
        )

    sig_a = set(merged.index[merged[padj_a_col] < alpha])
    sig_b = set(merged.index[merged[padj_b_col] < alpha])
    sig_in_both = sig_a & sig_b
    sig_in_either = sig_a | sig_b

    same_direction = np.sign(merged.loc[list(sig_in_both), lfc_a_col]) == \
        np.sign(merged.loc[list(sig_in_both), lfc_b_col])
    concordant = pd.Index(sorted(g for g in sig_in_both if same_direction[g]))
    discordant = pd.Index(sorted(g for g in sig_in_both if not same_direction[g]))
    only_in_a = pd.Index(sorted(sig_a - sig_b))
    only_in_b = pd.Index(sorted(sig_b - sig_a))

    pearson_r, _ = pearsonr(merged[lfc_a_col], merged[lfc_b_col])
    spearman_r, _ = spearmanr(merged[lfc_a_col], merged[lfc_b_col])
    directional_agreement = len(concordant) / len(sig_in_both) if sig_in_both else float("nan")
    jaccard_index = len(sig_in_both) / len(sig_in_either) if sig_in_either else float("nan")

    summary = {
        "genes_compared": len(merged),
        f"significant_{name_a}": len(sig_a),
        f"significant_{name_b}": len(sig_b),
        "significant_in_both": len(sig_in_both),
        "jaccard_index": jaccard_index,
        "pearson_r_lfc": pearson_r,
        "spearman_r_lfc": spearman_r,
        "directional_agreement": directional_agreement,
    }

    merged_out = merged.rename(columns={
        lfc_a_col: f"log_fold_change_{name_a}", padj_a_col: f"adjusted_p_value_{name_a}",
        lfc_b_col: f"log_fold_change_{name_b}", padj_b_col: f"adjusted_p_value_{name_b}",
    })
    merged_out[f"significant_{name_a}"] = merged_out.index.isin(sig_a)
    merged_out[f"significant_{name_b}"] = merged_out.index.isin(sig_b)

    return {
        "summary": summary,
        "concordant_genes": concordant,
        "discordant_genes": discordant,
        f"only_in_{name_a}": only_in_a,
        f"only_in_{name_b}": only_in_b,
        "merged": merged_out,
    }
