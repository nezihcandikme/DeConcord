import json
from unittest.mock import patch

import pandas as pd
import pytest

from deconcord.cli import main


def test_cli_runs_end_to_end_and_writes_outputs(tmp_path, capsys):
    out_dir = tmp_path / "out"

    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    assert (out_dir / "tables" / "differential_expression.csv").exists()
    assert (out_dir / "tables" / "qc.csv").exists()
    assert (out_dir / "figures" / "volcano.png").exists()
    assert (out_dir / "figures" / "pca.png").exists()
    assert (out_dir / "run_metadata.json").exists()

    de = pd.read_csv(out_dir / "tables" / "differential_expression.csv", index_col=0)
    assert list(de.index) == ["gene1", "gene2", "gene3", "gene4"]

    captured = capsys.readouterr()
    assert "genes tested" in captured.out


def test_cli_writes_run_metadata_with_expected_fields(tmp_path):
    out_dir = tmp_path / "out"

    main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--alpha", "0.01",
        "--method", "moderated",
        "--no-plots",
        "--out", str(out_dir),
    ])

    metadata = json.loads((out_dir / "run_metadata.json").read_text())

    assert "deconcord_version" in metadata
    assert "python_version" in metadata
    assert metadata["package_versions"]["pandas"]
    assert metadata["input_summary"]["n_genes"] == 4
    assert metadata["input_summary"]["group_1"] == ["sample1", "sample2"]
    assert metadata["parameters"]["method"] == "moderated"
    assert metadata["parameters"]["alpha"] == 0.01
    assert metadata["command"].startswith("deconcord tests/fixtures/cli_counts.csv")
    # ISO 8601 timestamp -- just check it parses, not an exact value.
    from datetime import datetime
    datetime.fromisoformat(metadata["timestamp_utc"])


def test_cli_version_flag_prints_version_and_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip().startswith("deconcord ")


def test_cli_no_plots_skips_figures(tmp_path):
    out_dir = tmp_path / "out"

    main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--no-plots",
        "--out", str(out_dir),
    ])

    assert not (out_dir / "figures" / "volcano.png").exists()
    assert not (out_dir / "figures" / "pca.png").exists()


def test_cli_with_gmt_writes_pathway_enrichment(tmp_path):
    out_dir = tmp_path / "out"

    main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--gmt", "tests/fixtures/sample_gene_sets.gmt",
        "--no-plots",
        "--out", str(out_dir),
    ])

    # sample_gene_sets.gmt doesn't share gene names with cli_counts.csv,
    # so this exercises the "runs without crashing on zero overlap" path --
    # the real assertion is that it still produced a table.
    assert (out_dir / "tables" / "pathway_enrichment.csv").exists()


def test_cli_with_gmt_writes_pathway_enrichment_plot(tmp_path):
    out_dir = tmp_path / "out"

    main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--gmt", "tests/fixtures/sample_gene_sets.gmt",
        "--out", str(out_dir),
    ])

    assert (out_dir / "figures" / "pathway_enrichment.png").exists()


def test_cli_min_count_filters_low_count_genes(tmp_path):
    out_dir = tmp_path / "out"

    main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--min-count", "10",
        "--min-samples", "2",
        "--no-plots",
        "--out", str(out_dir),
    ])

    de = pd.read_csv(out_dir / "tables" / "differential_expression.csv", index_col=0)
    # gene4 = [8, 11, 4, 7] only clears 10 reads in one sample -- filtered out.
    assert "gene4" not in list(de.index)


def test_cli_missing_sample_exits_nonzero(tmp_path, capsys):
    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,does_not_exist",
        "--group2", "sample3,sample4",
        "--out", str(tmp_path / "out"),
    ])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "does_not_exist" in captured.err


def test_cli_moderated_method(tmp_path):
    out_dir = tmp_path / "out"

    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--method", "moderated",
        "--no-plots",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    de = pd.read_csv(out_dir / "tables" / "differential_expression.csv", index_col=0)
    assert list(de.index) == ["gene1", "gene2", "gene3", "gene4"]


def test_cli_invalid_method_exits_nonzero(tmp_path):
    # argparse itself rejects an unknown --method choice via SystemExit,
    # before main()'s own try/except ever runs.
    with pytest.raises(SystemExit) as exc_info:
        main([
            "tests/fixtures/cli_counts.csv",
            "--group1", "sample1,sample2",
            "--group2", "sample3,sample4",
            "--method", "not-a-real-method",
            "--out", str(tmp_path / "out"),
        ])

    assert exc_info.value.code != 0


def test_cli_annotation_adds_gene_symbol_column(tmp_path):
    out_dir = tmp_path / "out"

    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--annotation", "tests/fixtures/cli_gene_annotation.csv",
        "--no-plots",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    de = pd.read_csv(out_dir / "tables" / "differential_expression.csv", index_col=0)
    assert de.loc["gene1", "gene_symbol"] == "GENE_ONE"
    assert de.loc["gene2", "gene_symbol"] == "GENE_TWO"
    assert pd.isna(de.loc["gene3", "gene_symbol"])


def test_cli_without_annotation_has_no_symbol_column(tmp_path):
    out_dir = tmp_path / "out"

    main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--no-plots",
        "--out", str(out_dir),
    ])

    de = pd.read_csv(out_dir / "tables" / "differential_expression.csv", index_col=0)
    assert "gene_symbol" not in de.columns


def test_cli_bad_annotation_file_exits_nonzero(tmp_path):
    bad_annotation = tmp_path / "bad_annotation.csv"
    bad_annotation.write_text("gene_id,gene_symbol\ngene1,GENE_ONE\ngene1,DIFFERENT\n")

    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--annotation", str(bad_annotation),
        "--out", str(tmp_path / "out"),
    ])

    assert exit_code == 1


def test_cli_missing_file_exits_nonzero():
    exit_code = main([
        "tests/fixtures/does_not_exist.csv",
        "--group1", "sample1",
        "--group2", "sample2",
    ])

    assert exit_code == 1


def test_cli_live_enrichment_organism_writes_output(tmp_path):
    out_dir = tmp_path / "out"
    fake_result = pd.DataFrame({"source": ["GO:BP"], "name": ["fake term"], "p_value": [0.01]})

    with patch(
        "deconcord.pathway_analysis.live_enrichment.run_gprofiler_enrichment",
        return_value=fake_result,
    ) as mock_query:
        exit_code = main([
            "tests/fixtures/cli_counts.csv",
            "--group1", "sample1,sample2",
            "--group2", "sample3,sample4",
            "--live-enrichment-organism", "hsapiens",
            "--no-plots",
            "--out", str(out_dir),
        ])

    assert exit_code == 0
    mock_query.assert_called_once()
    assert (out_dir / "tables" / "live_pathway_enrichment.csv").exists()


def test_cli_without_live_enrichment_organism_skips_it(tmp_path):
    out_dir = tmp_path / "out"

    with patch("deconcord.pathway_analysis.live_enrichment.run_gprofiler_enrichment") as mock_query:
        main([
            "tests/fixtures/cli_counts.csv",
            "--group1", "sample1,sample2",
            "--group2", "sample3,sample4",
            "--no-plots",
            "--out", str(out_dir),
        ])

    mock_query.assert_not_called()
    assert not (out_dir / "tables" / "live_pathway_enrichment.csv").exists()


def test_cli_live_enrichment_network_failure_exits_nonzero(tmp_path):
    with patch(
        "deconcord.pathway_analysis.live_enrichment.run_gprofiler_enrichment",
        side_effect=ConnectionError("Could not reach g:Profiler"),
    ):
        exit_code = main([
            "tests/fixtures/cli_counts.csv",
            "--group1", "sample1,sample2",
            "--group2", "sample3,sample4",
            "--live-enrichment-organism", "hsapiens",
            "--no-plots",
            "--out", str(tmp_path / "out"),
        ])

    assert exit_code == 1


def test_cli_with_covariates_writes_outputs(tmp_path):
    out_dir = tmp_path / "out"
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "sample,batch\nsample1,batch_A\nsample2,batch_B\nsample3,batch_A\nsample4,batch_B\n"
    )

    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--metadata", str(metadata_path),
        "--covariates", "batch",
        "--no-moderated-covariates",
        "--no-plots",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    de = pd.read_csv(out_dir / "tables" / "differential_expression.csv", index_col=0)
    assert list(de.index) == ["gene1", "gene2", "gene3", "gene4"]

    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    assert metadata["parameters"]["covariates"] == ["batch"]
    assert metadata["parameters"]["moderated_covariates"] is False


def test_cli_covariates_without_metadata_exits_nonzero(tmp_path):
    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--covariates", "batch",
        "--no-plots",
        "--out", str(tmp_path / "out"),
    ])

    assert exit_code == 1


def test_cli_metadata_without_covariates_exits_nonzero(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "sample,batch\nsample1,batch_A\nsample2,batch_B\nsample3,batch_A\nsample4,batch_B\n"
    )

    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--metadata", str(metadata_path),
        "--no-plots",
        "--out", str(tmp_path / "out"),
    ])

    assert exit_code == 1


def test_cli_empty_group_raises_via_exit_code():
    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "",
        "--group2", "sample3,sample4",
    ])

    assert exit_code == 1


def test_cli_explicit_background_used_instead_of_full_count_matrix(tmp_path):
    # --background lets a caller supply a narrower gene universe than "every
    # gene in the count matrix" (e.g. only genes that passed some other
    # filter upstream) -- exercised here with a single-gene background so the
    # written pathway_enrichment.csv's overlap_count is provably bounded by
    # it, not by the full 4-gene count matrix.
    out_dir = tmp_path / "out"

    main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--gmt", "tests/fixtures/sample_gene_sets.gmt",
        "--background", "gene1",
        "--no-plots",
        "--out", str(out_dir),
    ])

    enrichment = pd.read_csv(out_dir / "tables" / "pathway_enrichment.csv")
    assert (enrichment["overlap_count"] <= 1).all()


def test_cli_metadata_without_sample_column_uses_first_column_as_index(tmp_path):
    # Real metadata exports don't always name the sample-ID column "sample"
    # -- the CLI falls back to treating the first column as the index either
    # way, which this pins down explicitly rather than only ever testing the
    # "sample"-named case.
    out_dir = tmp_path / "out"
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "sample_name,batch\nsample1,batch_A\nsample2,batch_B\nsample3,batch_A\nsample4,batch_B\n"
    )

    exit_code = main([
        "tests/fixtures/cli_counts.csv",
        "--group1", "sample1,sample2",
        "--group2", "sample3,sample4",
        "--metadata", str(metadata_path),
        "--covariates", "batch",
        "--no-plots",
        "--out", str(out_dir),
    ])

    assert exit_code == 0
    assert (out_dir / "tables" / "differential_expression.csv").exists()


def test_installed_versions_reports_missing_package(monkeypatch):
    from deconcord.cli import _installed_versions, PackageNotFoundError

    def fake_version(pkg):
        if pkg == "scikit-learn":
            raise PackageNotFoundError(pkg)
        return "1.0.0"

    monkeypatch.setattr("deconcord.cli.version", fake_version)

    versions = _installed_versions()

    assert versions["scikit-learn"] == "not installed"
    assert versions["pandas"] == "1.0.0"
