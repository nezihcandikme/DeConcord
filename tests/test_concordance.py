import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from deconcord.concordance.methods import compute_de_concordance


def _method_a():
    return pd.DataFrame({
        "log_fold_change": [2.0, -3.0, 1.5, 0.5, 4.0, -1.0],
        "adjusted_p_value": [0.01, 0.01, 0.01, 0.2, 0.01, 0.3],
    }, index=["g1", "g2", "g3", "g4", "g5", "g6"])


def _method_b():
    return pd.DataFrame({
        "log_fold_change": [2.2, -2.8, -1.0, 0.6, 3.5, -1.2],
        "adjusted_p_value": [0.02, 0.03, 0.04, 0.01, 0.5, 0.4],
    }, index=["g1", "g2", "g3", "g4", "g5", "g6"])


# Constructed so that, at alpha=0.05:
# sig_a = {g1, g2, g3, g5}, sig_b = {g1, g2, g3, g4}
# sig_in_both = {g1, g2, g3} -- g1/g2 same-signed lfc (concordant),
# g3 opposite-signed (discordant, up in A, down in B)
# only_in_a = {g5}, only_in_b = {g4}


def test_compute_de_concordance_summary_metrics():
    result = compute_de_concordance(_method_a(), _method_b(), name_a="A", name_b="B")
    summary = result["summary"]

    assert summary["genes_compared"] == 6
    assert summary["significant_A"] == 4
    assert summary["significant_B"] == 4
    assert summary["significant_in_both"] == 3
    assert summary["jaccard_index"] == pytest.approx(3 / 5)
    assert summary["directional_agreement"] == pytest.approx(2 / 3)

    expected_pearson, _ = pearsonr(_method_a()["log_fold_change"], _method_b()["log_fold_change"])
    expected_spearman, _ = spearmanr(_method_a()["log_fold_change"], _method_b()["log_fold_change"])
    assert summary["pearson_r_lfc"] == pytest.approx(expected_pearson)
    assert summary["spearman_r_lfc"] == pytest.approx(expected_spearman)


def test_compute_de_concordance_gene_lists():
    result = compute_de_concordance(_method_a(), _method_b(), name_a="A", name_b="B")

    assert list(result["concordant_genes"]) == ["g1", "g2"]
    assert list(result["discordant_genes"]) == ["g3"]
    assert list(result["only_in_A"]) == ["g5"]
    assert list(result["only_in_B"]) == ["g4"]


def test_compute_de_concordance_merged_table_has_expected_shape():
    result = compute_de_concordance(_method_a(), _method_b(), name_a="A", name_b="B")
    merged = result["merged"]

    assert list(merged.index) == ["g1", "g2", "g3", "g4", "g5", "g6"]
    for col in [
        "log_fold_change_A", "adjusted_p_value_A", "significant_A",
        "log_fold_change_B", "adjusted_p_value_B", "significant_B",
    ]:
        assert col in merged.columns
    assert merged.loc["g1", "significant_A"] and merged.loc["g1", "significant_B"]
    assert not merged.loc["g4", "significant_A"]
    assert merged.loc["g4", "significant_B"]


def test_compute_de_concordance_default_column_names_match_deconcord_de_output():
    # No lfc_col_a/pvalue_col_a etc given -- defaults should match
    # run_differential_expression's own output column names directly.
    a = pd.DataFrame({
        "log_fold_change": [1.0, 2.0],
        "adjusted_p_value": [0.01, 0.01],
    }, index=["g1", "g2"])
    b = pd.DataFrame({
        "log_fold_change": [1.1, 2.2],
        "adjusted_p_value": [0.02, 0.02],
    }, index=["g1", "g2"])

    result = compute_de_concordance(a, b)
    assert result["summary"]["genes_compared"] == 2


def test_compute_de_concordance_deseq2_and_edger_column_names():
    deseq2 = pd.DataFrame({
        "log2FoldChange": [1.0, -2.0, 0.3],
        "padj": [0.01, 0.02, 0.5],
    }, index=["g1", "g2", "g3"])
    edger = pd.DataFrame({
        "logFC": [0.9, -1.8, 0.4],
        "FDR": [0.03, 0.01, 0.4],
    }, index=["g1", "g2", "g3"])

    result = compute_de_concordance(
        deseq2, edger, name_a="DESeq2", name_b="edgeR",
        lfc_col_a="log2FoldChange", pvalue_col_a="padj",
        lfc_col_b="logFC", pvalue_col_b="FDR",
    )

    assert result["summary"]["significant_DESeq2"] == 2
    assert result["summary"]["significant_edgeR"] == 2
    assert list(result["concordant_genes"]) == ["g1", "g2"]


def test_compute_de_concordance_drops_nan_rows_before_comparing():
    a = _method_a()
    b = _method_b().copy()
    b.loc["g6", "adjusted_p_value"] = np.nan  # e.g. DESeq2 independent-filtering NA

    result = compute_de_concordance(a, b)
    assert result["summary"]["genes_compared"] == 5
    assert "g6" not in result["merged"].index


def test_compute_de_concordance_no_significant_overlap_gives_nan_directional_agreement():
    a = pd.DataFrame({"log_fold_change": [0.1, 0.2], "adjusted_p_value": [0.9, 0.9]}, index=["g1", "g2"])
    b = pd.DataFrame({"log_fold_change": [0.1, 0.2], "adjusted_p_value": [0.9, 0.9]}, index=["g1", "g2"])

    result = compute_de_concordance(a, b)
    assert result["summary"]["significant_in_both"] == 0
    assert np.isnan(result["summary"]["directional_agreement"])
    assert np.isnan(result["summary"]["jaccard_index"])


def test_compute_de_concordance_no_gene_overlap_raises():
    a = pd.DataFrame({"log_fold_change": [1.0], "adjusted_p_value": [0.01]}, index=["g1"])
    b = pd.DataFrame({"log_fold_change": [1.0], "adjusted_p_value": [0.01]}, index=["g2"])

    with pytest.raises(ValueError, match="nothing to compare"):
        compute_de_concordance(a, b)


def test_compute_de_concordance_duplicate_index_raises():
    a = pd.DataFrame({
        "log_fold_change": [1.0, 1.0],
        "adjusted_p_value": [0.01, 0.02],
    }, index=["g1", "g1"])
    b = _method_b()

    with pytest.raises(ValueError, match="duplicate gene IDs"):
        compute_de_concordance(a, b, name_a="A")


def test_compute_de_concordance_missing_column_raises():
    a = pd.DataFrame({"log_fold_change": [1.0]}, index=["g1"])  # no p-value column
    b = _method_b()

    with pytest.raises(ValueError, match="missing column"):
        compute_de_concordance(a, b, name_a="A")


def test_compute_de_concordance_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        compute_de_concordance(_method_a(), _method_b(), alpha=0)

    with pytest.raises(ValueError, match="alpha"):
        compute_de_concordance(_method_a(), _method_b(), alpha=1.5)


def test_compute_de_concordance_warns_on_low_id_overlap():
    # Simulates a real, easy-to-hit mistake: the same 10 genes, but one
    # table's IDs carry an Ensembl-style version suffix and the other's
    # don't (e.g. one tool's annotation build vs. another's). Only 2 of
    # the 10 genes happen to match as literal strings -- well below the
    # 50% threshold -- which should warn rather than stay silent, since a
    # caller could easily misread the resulting low Jaccard index as a
    # real disagreement between methods instead of an ID-format mismatch.
    a = pd.DataFrame(
        {"log_fold_change": [1.0] * 10, "adjusted_p_value": [0.01] * 10},
        index=[f"ENSG{i:03d}.1" for i in range(10)],
    )
    b = pd.DataFrame(
        {"log_fold_change": [1.0] * 10, "adjusted_p_value": [0.01] * 10},
        # Only the first two rows use the same (unversioned) ID as `a`;
        # the rest use a disjoint ID space, standing in for "the same
        # underlying genes, named differently."
        index=[f"ENSG{i:03d}.1" for i in range(2)] + [f"OTHERID{i}" for i in range(8)],
    )

    with pytest.warns(UserWarning, match="gene ID conventions"):
        compute_de_concordance(a, b, name_a="A", name_b="B")


def test_compute_de_concordance_no_warning_on_good_overlap(recwarn):
    # The common, healthy case (matching IDs) shouldn't warn at all.
    compute_de_concordance(_method_a(), _method_b(), name_a="A", name_b="B")

    id_mismatch_warnings = [w for w in recwarn.list if "gene ID conventions" in str(w.message)]
    assert not id_mismatch_warnings
