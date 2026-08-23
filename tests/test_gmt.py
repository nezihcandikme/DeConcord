import pytest

from deconcord.pathway_analysis.gmt import load_gmt
from deconcord.pathway_analysis.methods import run_pathway_enrichment_analysis


def test_load_gmt_parses_gene_sets():
    gene_sets = load_gmt("tests/fixtures/sample_gene_sets.gmt")

    assert set(gene_sets.keys()) == {"PATHWAY_A", "PATHWAY_B", "PATHWAY_C"}
    assert gene_sets["PATHWAY_A"] == {"GENE1", "GENE2", "GENE3"}
    assert gene_sets["PATHWAY_B"] == {"GENE4", "GENE5"}


def test_load_gmt_skips_blank_lines():
    # The fixture has a blank line between PATHWAY_B and PATHWAY_C.
    gene_sets = load_gmt("tests/fixtures/sample_gene_sets.gmt")

    assert len(gene_sets) == 3


def test_load_gmt_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_gmt("tests/fixtures/does_not_exist.gmt")


def test_load_gmt_line_missing_genes_raises(tmp_path):
    gmt_file = tmp_path / "broken.gmt"
    gmt_file.write_text("PATHWAY_A\tsome description\n")

    with pytest.raises(ValueError, match="line 1"):
        load_gmt(str(gmt_file))


def test_load_gmt_blank_gene_fields_raises(tmp_path):
    # Distinct from test_load_gmt_line_missing_genes_raises above: that one
    # has too few tab-separated fields entirely (caught earlier, by the
    # "at least 3 fields" check). This line has enough fields -- the shape
    # a spreadsheet export with trailing empty cells produces -- but every
    # gene slot is an empty string, so it's the same practical problem
    # (no usable genes) caught by the later, separate `if not genes` check.
    gmt_file = tmp_path / "blank_genes.gmt"
    gmt_file.write_text("PATHWAY_A\tsome description\t\t\n")

    with pytest.raises(ValueError, match="has no genes"):
        load_gmt(str(gmt_file))


def test_load_gmt_duplicate_name_raises(tmp_path):
    gmt_file = tmp_path / "duplicate.gmt"
    gmt_file.write_text(
        "PATHWAY_A\tdesc\tGENE1\tGENE2\n"
        "PATHWAY_A\tdesc again\tGENE3\tGENE4\n"
    )

    with pytest.raises(ValueError, match="PATHWAY_A"):
        load_gmt(str(gmt_file))


def test_load_gmt_empty_file_raises(tmp_path):
    gmt_file = tmp_path / "empty.gmt"
    gmt_file.write_text("\n\n")

    with pytest.raises(ValueError, match="no gene sets"):
        load_gmt(str(gmt_file))


def test_load_gmt_output_plugs_into_enrichment():
    gene_sets = load_gmt("tests/fixtures/sample_gene_sets.gmt")
    background = {f"GENE{i}" for i in range(1, 9)}
    significant = {"GENE1", "GENE2", "GENE3"}

    results_df = run_pathway_enrichment_analysis(significant, gene_sets, background)

    assert set(results_df["pathway"]) == {"PATHWAY_A", "PATHWAY_B", "PATHWAY_C"}
    row_a = results_df.loc[results_df["pathway"] == "PATHWAY_A"].iloc[0]
    assert row_a["overlap_count"] == 3
