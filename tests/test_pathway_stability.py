import pandas as pd
import pytest

from deconcord.concordance.pathway_stability import compute_pathway_stability


def _enrichment_a():
    return pd.DataFrame({
        "pathway": ["p1", "p2", "p3", "p4"],
        "adjusted_p_value": [0.01, 0.01, 0.2, 0.01],
    })


def _enrichment_b():
    return pd.DataFrame({
        "pathway": ["p1", "p2", "p3", "p5"],
        "adjusted_p_value": [0.02, 0.3, 0.01, 0.01],
    })


# At alpha=0.05: sig_a = {p1, p2, p4}, sig_b = {p1, p3, p5}
# Tested in both: p1, p2, p3 (p4 only in A, p5 only in B, excluded from
# the merge entirely since compute_pathway_stability only compares
# pathways present in both tables).
# stable (sig in both) = {p1}; only_in_a among compared = {p2}; only_in_b = {p3}


def test_compute_pathway_stability_summary_metrics():
    result = compute_pathway_stability(_enrichment_a(), _enrichment_b(), name_a="A", name_b="B")
    summary = result["summary"]

    assert summary["pathways_compared"] == 3  # p1, p2, p3 -- present in both tables
    assert summary["significant_A"] == 2  # p1, p2
    assert summary["significant_B"] == 2  # p1, p3
    assert summary["significant_in_both"] == 1  # p1
    assert summary["jaccard_index"] == pytest.approx(1 / 3)  # |{p1}| / |{p1,p2,p3}|


def test_compute_pathway_stability_pathway_lists():
    result = compute_pathway_stability(_enrichment_a(), _enrichment_b(), name_a="A", name_b="B")

    assert list(result["stable_pathways"]) == ["p1"]
    assert list(result["only_in_A"]) == ["p2"]
    assert list(result["only_in_B"]) == ["p3"]


def test_compute_pathway_stability_merged_table_has_expected_shape():
    result = compute_pathway_stability(_enrichment_a(), _enrichment_b(), name_a="A", name_b="B")
    merged = result["merged"]

    assert sorted(merged.index) == ["p1", "p2", "p3"]
    for col in ["adjusted_p_value_A", "significant_A", "adjusted_p_value_B", "significant_B"]:
        assert col in merged.columns
    assert merged.loc["p1", "significant_A"] and merged.loc["p1", "significant_B"]
    assert merged.loc["p2", "significant_A"] and not merged.loc["p2", "significant_B"]


def test_compute_pathway_stability_different_column_names():
    # Mimic comparing local enrichment output (pathway/adjusted_p_value)
    # against live g:Profiler output (name/p_value).
    local = _enrichment_a()
    live = _enrichment_b().rename(columns={"pathway": "name", "adjusted_p_value": "p_value"})

    result = compute_pathway_stability(
        local, live, name_a="local", name_b="live",
        pathway_col_b="name", pvalue_col_b="p_value",
    )
    assert result["summary"]["pathways_compared"] == 3


def test_compute_pathway_stability_no_significant_overlap_gives_nan_jaccard():
    a = pd.DataFrame({"pathway": ["p1", "p2"], "adjusted_p_value": [0.9, 0.9]})
    b = pd.DataFrame({"pathway": ["p1", "p2"], "adjusted_p_value": [0.9, 0.9]})

    result = compute_pathway_stability(a, b)
    assert result["summary"]["significant_in_both"] == 0
    import numpy as np
    assert np.isnan(result["summary"]["jaccard_index"])


def test_compute_pathway_stability_no_pathway_overlap_raises():
    a = pd.DataFrame({"pathway": ["p1"], "adjusted_p_value": [0.01]})
    b = pd.DataFrame({"pathway": ["p2"], "adjusted_p_value": [0.01]})

    with pytest.raises(ValueError, match="nothing to compare"):
        compute_pathway_stability(a, b)


def test_compute_pathway_stability_duplicate_pathway_name_raises():
    a = pd.DataFrame({"pathway": ["p1", "p1"], "adjusted_p_value": [0.01, 0.02]})
    b = _enrichment_b()

    with pytest.raises(ValueError, match="duplicate pathway names"):
        compute_pathway_stability(a, b, name_a="A")


def test_compute_pathway_stability_missing_column_raises():
    a = pd.DataFrame({"pathway": ["p1"]})  # no p-value column
    b = _enrichment_b()

    with pytest.raises(ValueError, match="missing column"):
        compute_pathway_stability(a, b, name_a="A")


def test_compute_pathway_stability_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        compute_pathway_stability(_enrichment_a(), _enrichment_b(), alpha=0)

    with pytest.raises(ValueError, match="alpha"):
        compute_pathway_stability(_enrichment_a(), _enrichment_b(), alpha=1.5)


def test_compute_pathway_stability_drops_nan_pvalue_rows():
    a = _enrichment_a()
    b = _enrichment_b().copy()
    b.loc[b["pathway"] == "p3", "adjusted_p_value"] = None

    result = compute_pathway_stability(a, b)
    assert "p3" not in result["merged"].index
    assert result["summary"]["pathways_compared"] == 2


def test_compute_pathway_stability_warns_on_low_name_overlap():
    # Stands in for a real naming-convention mismatch between two
    # enrichment sources (e.g. an MSigDB-style name in one table, a
    # g:Profiler-style term name for the same pathway in the other): only
    # 1 of 10 pathway names happens to match as a literal string.
    a = pd.DataFrame({
        "pathway": [f"PATHWAY_{i}" for i in range(10)],
        "adjusted_p_value": [0.01] * 10,
    })
    b = pd.DataFrame({
        "pathway": ["PATHWAY_0"] + [f"other name {i}" for i in range(9)],
        "adjusted_p_value": [0.01] * 10,
    })

    with pytest.warns(UserWarning, match="naming-convention mismatch"):
        compute_pathway_stability(a, b, name_a="A", name_b="B")


def test_compute_pathway_stability_no_warning_on_good_overlap(recwarn):
    compute_pathway_stability(_enrichment_a(), _enrichment_b(), name_a="A", name_b="B")

    naming_warnings = [w for w in recwarn.list if "naming-convention" in str(w.message)]
    assert not naming_warnings
