import numpy as np
import pandas as pd
import pytest

from deconcord.io.counts import (
    CountMatrixError,
    MissingValuesError,
    NonFiniteCountsError,
    NonNumericCountsError,
    NonUniqueColumnsError,
    NonUniqueIndexError,
    check_all_finite,
    check_all_integer,
    check_no_missing_values,
    check_numeric_dtype,
    check_unique_columns,
    check_unique_index,
    sniff_delimiter,
    validate_counts,
    load_count_matrix,
)


def test_validate_counts_invalid_raises():
    # A non-numeric column (e.g. a stray text value) is the thing
    # validate_counts should still reject outright -- fractional numeric
    # values, on the other hand, are now legitimate (see
    # test_validate_counts_fractional_counts_valid below).
    df = pd.DataFrame({"sample_A": ["not", "a", "count"]})

    with pytest.raises(CountMatrixError):
        validate_counts(df)


def test_validate_counts_fractional_counts_valid():
    # salmon/kallisto/RSEM "estimated counts" are legitimately fractional
    # (multi-mapping reads get split across the transcripts they map to),
    # not whole numbers -- these are still real counts and validate_counts
    # must accept them, not just featureCounts/htseq-count's integer output.
    df = pd.DataFrame({"sample_A": [12.4, 0.0, 340.7], "sample_B": [10.9, 1.0, 355.2]})

    validate_counts(df)


def test_validate_counts_non_numeric_column_raises():
    df = pd.DataFrame({"sample_A": ["x", "y", "z"]})

    with pytest.raises(NonNumericCountsError):
        validate_counts(df)


def test_validate_counts_missing_value_raises():
    # A NaN in an otherwise-real count matrix is one of the most common
    # real-data problems (a dropout, a merge that didn't line up every
    # gene) -- pinned down directly rather than only exercised indirectly
    # through check_no_missing_values.
    df = pd.DataFrame({"sample_A": [1.0, float("nan"), 3.0], "sample_B": [4.0, 5.0, 6.0]})

    with pytest.raises(MissingValuesError):
        validate_counts(df)


def test_check_no_missing_values_true_for_complete_data():
    df = pd.DataFrame({"sample_A": [1, 2, 3]})
    assert check_no_missing_values(df) is True


def test_check_no_missing_values_false_with_nan():
    df = pd.DataFrame({"sample_A": [1.0, float("nan"), 3.0]})
    assert check_no_missing_values(df) is False


def test_validate_counts_infinite_value_raises_specific_error():
    df = pd.DataFrame({"sample_A": [1.0, np.inf, 3.0]})

    with pytest.raises(NonFiniteCountsError):
        validate_counts(df)


def test_check_all_integer_valid():
    df = pd.DataFrame({"sample_A": [1, 2, 3]})

    assert check_all_integer(df) is True


def test_check_all_integer_invalid():
    df = pd.DataFrame({"sample_A": [1.5, 2, 3]})

    assert check_all_integer(df) is False


def test_check_numeric_dtype_true_for_fractional():
    # check_numeric_dtype (what validate_counts actually enforces) is more
    # permissive than check_all_integer on purpose.
    df = pd.DataFrame({"sample_A": [1.5, 2.0, 3.25]})

    assert check_numeric_dtype(df) is True


def test_check_numeric_dtype_false_for_text():
    df = pd.DataFrame({"sample_A": ["a", "b", "c"]})

    assert check_numeric_dtype(df) is False


def test_check_all_finite_true_for_normal_values():
    df = pd.DataFrame({"sample_A": [1.0, 2.0, 3.0]})

    assert check_all_finite(df) is True


def test_check_all_finite_false_for_inf():
    df = pd.DataFrame({"sample_A": [1.0, np.inf, 3.0]})

    assert check_all_finite(df) is False


def test_check_all_finite_ignores_nan():
    # NaN is check_no_missing_values's job, not check_all_finite's -- a
    # missing value shouldn't also trip the "contains infinity" error.
    df = pd.DataFrame({"sample_A": [1.0, np.nan, 3.0]})

    assert check_all_finite(df) is True


def test_validate_counts_valid_does_not_raise():
    df = pd.DataFrame({"sample_A": [1, 2, 3]})

    validate_counts(df)


def test_validate_counts_negative_fractional_value_raises():
    # -1.5 is invalid for a different reason now than it used to be:
    # fractional is fine (see test_validate_counts_fractional_counts_valid),
    # negative is what actually trips this one.
    df = pd.DataFrame({"sample_A": [-1.5, 2, 3]})

    with pytest.raises(CountMatrixError):
        validate_counts(df)


def test_validate_counts_multiple_errors_combined_in_message():
    # Two independent problems at once (a non-numeric column and a
    # duplicate gene ID) should both be reported together in one raised
    # error, not just the first one found -- validate_counts collects
    # every problem before raising, on purpose, so a single failed run
    # tells you everything that's wrong instead of one error at a time.
    df = pd.DataFrame(
        {"sample_A": ["x", "y"], "sample_B": [1, 2]},
        index=["gene1", "gene1"],
    )

    with pytest.raises(CountMatrixError) as exc_info:
        validate_counts(df)

    message = str(exc_info.value)
    assert "numeric" in message
    assert "unique" in message.lower() or "index" in message.lower()


def test_load_count_matrix_invalid_raises():
    with pytest.raises(CountMatrixError):
        load_count_matrix("tests/fixtures/bad_counts.csv")


def test_load_count_matrix_valid():
    df = load_count_matrix("tests/fixtures/toy_counts.csv")
    assert df.shape == (4, 3)


def test_validate_counts_warns_on_likely_transposed_matrix():
    # Wide and short: many more "genes" (columns) than "samples" (rows) —
    # the opposite of what a real count matrix looks like, and a strong
    # hint this DataFrame was loaded transposed.
    df = pd.DataFrame(
        [[1] * 50, [2] * 50],
        columns=[f"gene{i}" for i in range(50)],
    )

    with pytest.warns(UserWarning, match="transpose"):
        validate_counts(df)


def test_validate_counts_normal_shape_does_not_warn(recwarn):
    # More genes (rows) than samples (columns), the normal RNA-seq shape —
    # the orientation heuristic should stay quiet here.
    df = pd.DataFrame({"sample_A": [1, 2, 3, 4], "sample_B": [5, 6, 7, 8]})

    validate_counts(df)

    assert len(recwarn) == 0


def test_check_unique_index_true_for_unique():
    df = pd.DataFrame({"sample_A": [1, 2, 3]}, index=["gene1", "gene2", "gene3"])
    assert check_unique_index(df) is True


def test_check_unique_index_false_for_duplicates():
    df = pd.DataFrame({"sample_A": [1, 2, 3]}, index=["gene1", "gene1", "gene3"])
    assert check_unique_index(df) is False


def test_validate_counts_duplicate_gene_ids_raises():
    # Two rows sharing a gene ID is a real, easy-to-hit input mistake
    # (e.g. concatenating two annotation releases) -- validate_counts must
    # catch it rather than silently letting duplicate-indexed rows through
    # to functions that assume a unique index (like .loc lookups downstream).
    df = pd.DataFrame(
        {"sample_A": [1, 2, 3], "sample_B": [4, 5, 6]},
        index=["gene1", "gene1", "gene2"],
    )

    with pytest.raises(NonUniqueIndexError):
        validate_counts(df)


def test_validate_counts_infinite_values_raises():
    # An infinite value isn't a real count under either the strict
    # (check_all_integer) or the permissive (check_numeric_dtype) reading
    # -- see check_all_finite and test_validate_counts_infinite_value_raises_specific_error
    # for the explicit, named-exception version of this same check.
    df = pd.DataFrame({"sample_A": [1.0, np.inf, 3.0]})

    with pytest.raises(CountMatrixError):
        validate_counts(df)


def test_check_unique_columns_true_for_unique():
    df = pd.DataFrame({"sample_A": [1, 2, 3], "sample_B": [4, 5, 6]})
    assert check_unique_columns(df) is True


def test_check_unique_columns_false_for_duplicates():
    df = pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=["sample_A", "sample_A"])
    assert check_unique_columns(df) is False


def test_validate_counts_duplicate_sample_columns_raises():
    # Two columns sharing a sample name is a real, easy-to-hit mistake with
    # real data (a copy-pasted header, two lanes accidentally merged under
    # one label) -- df[["sample_A"]] for a duplicated name silently returns
    # both columns instead of one, which downstream group-mean/variance
    # calculations in the DE functions would average across without any
    # sign anything was wrong. validate_counts must catch it, mirroring the
    # existing gene-ID-side check (check_unique_index).
    df = pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=["sample_A", "sample_A"])

    with pytest.raises(NonUniqueColumnsError):
        validate_counts(df)


def test_validate_counts_empty_rows_raises():
    # Zero genes (rows) with real sample columns -- every other check in
    # validate_counts passes vacuously on an empty axis, so this needs its
    # own explicit guard rather than relying on the rest to catch it.
    df = pd.DataFrame({"sample_A": [], "sample_B": []}, dtype=int)

    with pytest.raises(CountMatrixError, match="no genes"):
        validate_counts(df)


def test_validate_counts_empty_columns_raises():
    # Zero sample columns -- the shape a real featureCounts/htseq-count
    # file produces when it's read with the wrong delimiter (tab-separated
    # data parsed as CSV collapses into one unparsed index column and zero
    # data columns). Confirmed against a real-shaped fixture in the past;
    # this pins down the fix so it can't silently regress.
    df = pd.DataFrame(index=["gene1", "gene2", "gene3"])

    with pytest.raises(CountMatrixError, match="no samples"):
        validate_counts(df)


def test_validate_counts_all_zero_column_is_valid():
    # A sample where every gene has zero counts is unusual but not
    # malformed on its own -- validate_counts only checks structural/type
    # validity. (Normalization downstream, e.g. compute_cpm, is where an
    # all-zero sample is rejected, since CPM is undefined for it.)
    df = pd.DataFrame({"sample_A": [0, 0, 0], "sample_B": [1, 2, 3]})

    validate_counts(df)


def test_sniff_delimiter_detects_comma(tmp_path):
    path = tmp_path / "counts.csv"
    path.write_text("gene_id,sample_A,sample_B\nENSG001,10,12\n")

    assert sniff_delimiter(str(path)) == ","


def test_sniff_delimiter_detects_tab(tmp_path):
    # The shape a real featureCounts/htseq-count export actually has.
    path = tmp_path / "counts.txt"
    path.write_text("gene_id\tsample_A\tsample_B\nENSG001\t10\t12\n")

    assert sniff_delimiter(str(path)) == "\t"


def test_sniff_delimiter_handles_gzip(tmp_path):
    import gzip

    path = tmp_path / "counts.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("gene_id\tsample_A\tsample_B\nENSG001\t10\t12\n")

    assert sniff_delimiter(str(path)) == "\t"


def test_load_count_matrix_auto_detects_tab_separated_file(tmp_path):
    # This is the exact real-world shape that used to load as a single,
    # useless column: a featureCounts/htseq-count-style tab-separated
    # export, loaded with no explicit sep given.
    path = tmp_path / "featurecounts_style.txt"
    path.write_text(
        "Geneid\tsample_A\tsample_B\n"
        "ENSG001\t12\t15\n"
        "ENSG002\t0\t3\n"
    )

    df = load_count_matrix(str(path))

    assert df.shape == (2, 2)
    assert list(df.columns) == ["sample_A", "sample_B"]


def test_load_count_matrix_explicit_sep_overrides_sniff(tmp_path):
    path = tmp_path / "counts.txt"
    path.write_text("gene_id;sample_A;sample_B\nENSG001;10;12\n")

    df = load_count_matrix(str(path), sep=";")

    assert df.shape == (1, 2)
