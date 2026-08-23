from __future__ import annotations

import gzip
import warnings

import numpy as np
import pandas as pd

# A real RNA-seq count matrix is lopsided: thousands of genes, a much
# smaller handful of samples. If a matrix has more columns than rows and
# very few rows, it's a decent bet someone loaded it transposed.
_ORIENTATION_WARNING_MAX_ROWS = 30


class CountMatrixError(Exception):
    """Base exception for problems found in a count matrix."""


class NonIntegerCountsError(CountMatrixError):
    """
    Raised when a count matrix contains non-integer columns.

    Not raised by ``validate_counts`` by default -- see
    ``check_numeric_dtype``/``check_all_finite`` for what actually gates
    validation now. Kept as a public exception (and ``check_all_integer``
    kept as a standalone check) for a caller who wants to enforce strict
    integer counts themselves, e.g. before assuming a matrix is exactly
    featureCounts/htseq-style raw integer output rather than salmon/
    kallisto/RSEM-style estimated (fractional) counts.
    """


class NonNumericCountsError(CountMatrixError):
    """Raised when a count matrix contains a non-numeric column."""


class NonFiniteCountsError(CountMatrixError):
    """Raised when a count matrix contains an infinite (+/-inf) value."""


class NonUniqueIndexError(CountMatrixError):
    """Raised when a count matrix has a non-unique (duplicated) gene index."""


class NonUniqueColumnsError(CountMatrixError):
    """Raised when a count matrix has a non-unique (duplicated) sample column name."""


class NegativeCountsError(CountMatrixError):
    """Raised when a count matrix contains negative values."""


class MissingValuesError(CountMatrixError):
    """Raised when a count matrix contains missing (NaN) values."""


def check_no_missing_values(df: pd.DataFrame) -> bool:
    """True if the DataFrame has no missing values."""
    return not df.isna().any().any()


def check_all_integer(df: pd.DataFrame) -> bool:
    """
    True if every column has an integer dtype.

    Not used by ``validate_counts`` by default (see
    ``check_numeric_dtype``) -- kept as a standalone, strict check for a
    caller who specifically wants to require whole-number counts (e.g.
    raw featureCounts/htseq-count output, which is always integer) rather
    than DEConcord's default, which also accepts the legitimately
    fractional "estimated counts" salmon/kallisto/RSEM produce.
    """
    for dtype in df.dtypes:
        if not np.issubdtype(dtype, np.integer):
            return False

    return True


def check_numeric_dtype(df: pd.DataFrame) -> bool:
    """
    True if every column has a numeric (integer or floating-point) dtype.

    This is the dtype check ``validate_counts`` actually enforces: a count
    has to be *a number*, but not necessarily a whole number -- raw
    featureCounts/htseq-count output is integer, but salmon/kallisto/RSEM
    report "estimated counts" that are legitimately fractional (reads that
    map to multiple transcripts get split, non-integer shares assigned to
    each), and rejecting those outright would lock out real quantification
    output for no statistical reason (nothing downstream of this — CPM,
    log2, the t-tests — requires whole numbers). Use ``check_all_integer``
    instead if a caller genuinely needs to enforce whole-number counts.
    """
    for dtype in df.dtypes:
        if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
            return False

    return True


def check_all_finite(df: pd.DataFrame) -> bool:
    """
    True if every value is finite (no ``+inf``/``-inf``).

    Missing values (``NaN``) are checked separately by
    ``check_no_missing_values`` — ``np.isfinite`` treats ``NaN`` as "not
    finite" too, so ``NaN`` is excluded here on purpose to keep the two
    error messages (missing vs. infinite) distinct instead of one
    infinite-looking value masking as the other.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    values = numeric.to_numpy(dtype=float)
    finite_values = values[~np.isnan(values)]
    return bool(np.isfinite(finite_values).all())


def check_unique_index(df: pd.DataFrame) -> bool:
    """True if the DataFrame's index (gene IDs) has no duplicates."""
    return df.index.is_unique


def check_unique_columns(df: pd.DataFrame) -> bool:
    """
    True if the DataFrame's columns (sample names) have no duplicates.

    The gene-ID side of this (``check_unique_index``) was already checked
    elsewhere; the sample side wasn't. Two columns sharing a name is a real
    failure mode with real data, not a hypothetical one (a re-exported
    matrix with two lanes accidentally merged under the same sample label,
    a copy-paste error building the header row) -- and it's a dangerous one
    to let through silently, since ``df[["sample_A"]]`` for a duplicated
    "sample_A" returns two columns instead of one, which downstream code
    (mean/variance per group in the DE functions) would average across
    without any indication anything was wrong.
    """
    return not df.columns.duplicated().any()


def check_nonnegative_counts(df: pd.DataFrame) -> bool:
    """
    True if every non-missing numeric value in the DataFrame is >= 0.

    Restricted to numeric columns (same ``select_dtypes`` guard
    ``check_all_finite`` uses) so this doesn't itself crash with a raw
    ``TypeError`` on a non-numeric column -- ``validate_counts`` runs
    every check and collects every failure before raising, including
    this one, so it has to survive being called on data that already
    failed a different check (e.g. a text column) without blowing up
    before the *real* problem (non-numeric data) gets reported.

    ``NaN`` is excluded from the comparison on purpose, the same way
    ``check_all_finite`` excludes it: a comparison against ``NaN``
    (``nan >= 0``) evaluates to ``False`` in pandas/numpy, which would
    otherwise make a single missing value trip *this* check too and
    surface as a second, redundant "counts must be non-negative" error
    alongside the real "missing values" one for the same root cause.
    ``check_no_missing_values`` is what's responsible for reporting NaN.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    return bool((numeric.fillna(0) >= 0).all().all())


def check_likely_correct_orientation(df: pd.DataFrame) -> bool:
    """
    Heuristic check that ``df`` looks like genes-as-rows, samples-as-columns
    (the orientation every other function in DEConcord assumes).

    This can't be known for certain from the numbers alone — it's a sanity
    check, not a proof. A matrix with more columns than rows and only a
    handful of rows is flagged as "probably transposed" because that shape
    is far more consistent with samples-as-rows than with a real gene list.
    A legitimately tiny custom gene panel would also trip this, so treat a
    False result as "go double check your orientation," not as evidence of
    a bug on its own.
    """
    n_rows, n_cols = df.shape
    return not (n_cols > n_rows and n_rows <= _ORIENTATION_WARNING_MAX_ROWS)


def validate_counts(df: pd.DataFrame) -> None:
    """
    Validate the structure and values of a raw count matrix.

    Expects genes as rows and samples as columns, with non-negative numeric
    values and no missing data. Values don't have to be whole numbers:
    featureCounts/htseq-count output is integer, but salmon/kallisto/RSEM
    "estimated counts" are legitimately fractional (see
    ``check_numeric_dtype``), and both are accepted here. All problems
    found are collected and raised together (rather than stopping at the
    first one) so a single failed run tells you everything that's wrong
    instead of one error at a time.

    Also emits a ``UserWarning`` (not a hard failure) if the matrix's shape
    looks like it might be transposed — see ``check_likely_correct_orientation``.

    Parameters
    ----------
    df : pd.DataFrame
        The count matrix to validate.

    Raises
    ------
    CountMatrixError
        If the matrix is empty (zero rows or zero columns), or contains
        missing values, non-numeric columns, infinite values, duplicate
        gene IDs, duplicate sample column names, negative counts, or a
        combination of these.
    """
    # Annotated explicitly as the shared base class -- left as a bare `[]`,
    # mypy infers the list's element type from whichever specific subclass
    # gets appended first, then flags every other subclass appended below
    # as incompatible with that overly narrow inferred type.
    errors: list[CountMatrixError] = []

    # A zero-row or zero-column matrix isn't "valid data with nothing wrong
    # with it" -- every check below passes vacuously on an empty axis (an
    # empty integer check, an empty non-negativity check, etc. are all
    # trivially true), so a genuinely broken load (e.g. a tab-separated
    # featureCounts file read with the wrong delimiter, which collapses
    # into one unparsed index column and zero data columns) would sail
    # through as "valid" instead of failing where the mistake actually is.
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise CountMatrixError(
            f"Count matrix has shape {df.shape} -- {'no genes (rows)' if df.shape[0] == 0 else 'no samples (columns)'} "
            "found. This usually means the file didn't parse the way you expect: "
            "double check the delimiter (a tab-separated file read as CSV collapses "
            "into a single unparsed column) and that the first column is really the "
            "gene ID, not part of the header."
        )

    if not check_no_missing_values(df):
        errors.append(MissingValuesError("Counts must not contain missing values."))

    if not check_numeric_dtype(df):
        bad_cols = df.dtypes[
            ~df.dtypes.apply(
                lambda dtype: np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
            )
        ].index.tolist()

        errors.append(NonNumericCountsError(
            f"Columns {bad_cols} must have a numeric (integer or float) data type."
        ))

    if not check_all_finite(df):
        errors.append(NonFiniteCountsError(
            "Counts must not contain infinite (+inf/-inf) values."
        ))

    if not check_unique_index(df):
        errors.append(NonUniqueIndexError("The index of the DataFrame must be unique."))

    if not check_unique_columns(df):
        dupe_cols = df.columns[df.columns.duplicated()].unique().tolist()
        errors.append(NonUniqueColumnsError(
            f"Sample column name(s) {dupe_cols} appear more than once. "
            "Each sample column must have a unique name."
        ))

    if not check_nonnegative_counts(df):
        errors.append(NegativeCountsError("Counts must be non-negative."))

    if len(errors) == 1:
        raise errors[0]
    elif errors:
        raise CountMatrixError("\n".join(str(e) for e in errors))

    if not check_likely_correct_orientation(df):
        warnings.warn(
            f"This count matrix has {df.shape[1]} columns and only "
            f"{df.shape[0]} rows. DEConcord expects genes as rows and "
            "samples as columns — if that's backwards here, transpose "
            "your input (df.T) before passing it in.",
            stacklevel=2,
        )


def sniff_delimiter(path: str) -> str:
    """
    Guess whether a file is comma- or tab-separated by counting each in its
    header line.

    Real raw bulk RNA-seq quantification tools (featureCounts, htseq-count)
    almost always write tab-separated output; a manual export, or
    DEConcord's own CSV writers, use comma. Reading the wrong one doesn't
    fail at read time — it silently collapses into one big unparsed index
    column and zero data columns (see ``validate_counts``'s empty-matrix
    check for what happens next) — so guessing right here matters more
    than for a typical "just pick a default" delimiter question. This
    covers the two delimiters actually seen in DEConcord's own use, not a
    general-purpose format sniffer (``csv.Sniffer`` exists for that, and
    is overkill for a project that only needs to tell comma from tab).

    Parameters
    ----------
    path : str
        Path to the file, plain text or ``.gz``-compressed (matched by
        the ``.gz`` suffix, same convention ``pandas`` itself uses).

    Returns
    -------
    str
        ``"\\t"`` if the header line has more tabs than commas, ``","``
        otherwise (including a tie, or a header with neither — comma is
        the more common default to fall back to).
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def load_count_matrix(path: str, sep: str | None = None) -> pd.DataFrame:
    """
    Load a count matrix from a CSV/TSV file and validate it via ``validate_counts``.

    The first column of the file is used as the gene ID index; every other
    column is treated as a sample.

    Parameters
    ----------
    path : str
        Path to the file. ``.gz``-compressed files are handled
        automatically (``pandas``' own compression inference).
    sep : str, optional
        Field delimiter. Default ``None``, which auto-detects comma vs.
        tab via ``sniff_delimiter`` — covers both a plain CSV export and
        the tab-separated output most raw bulk RNA-seq quantification
        tools (featureCounts, htseq-count) actually produce, without
        requiring the caller to know or specify which up front. Pass an
        explicit value (e.g. ``";"``) to override the sniff for anything
        else.

    Returns
    -------
    pd.DataFrame
        The validated, raw (not normalized) count matrix.

    Raises
    ------
    CountMatrixError
        If the loaded DataFrame fails validation.
    """
    if sep is None:
        sep = sniff_delimiter(path)
    df = pd.read_csv(path, index_col=0, sep=sep)
    validate_counts(df)
    return df
