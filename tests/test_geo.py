import gzip
import urllib.error
from unittest.mock import patch

import pytest

from deconcord.io.geo import (
    _series_dir_url,
    download_geo_supplementary_file,
    fetch_geo_sample_metadata,
    list_geo_supplementary_files,
)


class _FakeResponse:
    """Stand-in for what urllib.request.urlopen returns: a context manager
    whose .read() supports both the no-argument form (used to slurp an
    entire response) and the chunked form shutil.copyfileobj uses."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, size=-1):
        if size is None or size < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
        else:
            chunk = self._data[self._pos:self._pos + size]
            self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


_SAMPLE_SUPPL_HTML = """
<html><body>
<a href="?C=N;O=D">Name</a>
<a href="/geo/series/GSE52nnn/">Parent Directory</a>
<a href="GSE52778_raw_counts.csv.gz">GSE52778_raw_counts.csv.gz</a>
<a href="GSE52778_readme.txt">GSE52778_readme.txt</a>
</body></html>
"""

_SAMPLE_SERIES_MATRIX = (
    '!Sample_title\t"sample 1"\t"sample 2"\n'
    '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
    '!Sample_characteristics_ch1\t"tissue: lung"\t"tissue: liver"\n'
    '!Sample_characteristics_ch1\t"treatment: none"\t"treatment: drug"\n'
    "!series_matrix_table_begin\n"
    "some\tunrelated\tdata\n"
)


def test_series_dir_url_bucket_truncation():
    assert _series_dir_url("GSE52778") == (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE52nnn/GSE52778"
    )


def test_series_dir_url_rejects_non_gse_accession():
    with pytest.raises(ValueError, match="GEO series accession"):
        _series_dir_url("GSM12345")


@patch("urllib.request.urlopen")
def test_list_geo_supplementary_files_parses_hrefs(mock_urlopen):
    mock_urlopen.return_value = _FakeResponse(_SAMPLE_SUPPL_HTML.encode("utf-8"))

    files = list_geo_supplementary_files("GSE52778")

    assert files == ["GSE52778_raw_counts.csv.gz", "GSE52778_readme.txt"]


@patch("urllib.request.urlopen")
def test_list_geo_supplementary_files_network_failure_raises_connection_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("timed out")

    with pytest.raises(ConnectionError, match="GEO"):
        list_geo_supplementary_files("GSE52778")


@patch("urllib.request.urlopen")
def test_list_geo_supplementary_files_http_error_raises_runtime_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE00nnn/GSE00000/suppl/",
        code=404, msg="Not Found", hdrs=None, fp=None,
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        list_geo_supplementary_files("GSE00000")


@patch("urllib.request.urlopen")
def test_download_geo_supplementary_file_writes_local_file(mock_urlopen, tmp_path):
    mock_urlopen.return_value = _FakeResponse(b"gene_id,sample1,sample2\nGENE1,10,20\n")

    dest_path = download_geo_supplementary_file(
        "GSE52778", "GSE52778_raw_counts.csv", dest_dir=str(tmp_path)
    )

    assert dest_path == str(tmp_path / "GSE52778_raw_counts.csv")
    with open(dest_path, "rb") as f:
        assert f.read() == b"gene_id,sample1,sample2\nGENE1,10,20\n"


@patch("urllib.request.urlopen")
def test_download_geo_supplementary_file_network_failure_raises_connection_error(mock_urlopen, tmp_path):
    # Mirrors test_list_geo_supplementary_files_network_failure_raises_connection_error
    # above, but for download_geo_supplementary_file's own urlopen call --
    # a separate try/except around a separate network request, so it needs
    # its own direct test rather than assuming the sibling function's
    # coverage carries over.
    mock_urlopen.side_effect = urllib.error.URLError("timed out")

    with pytest.raises(ConnectionError, match="GEO"):
        download_geo_supplementary_file("GSE52778", "GSE52778_raw_counts.csv", dest_dir=str(tmp_path))


@patch("urllib.request.urlopen")
def test_fetch_geo_sample_metadata_network_failure_raises_connection_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("timed out")

    with pytest.raises(ConnectionError, match="GEO"):
        fetch_geo_sample_metadata("GSE52778")


@patch("urllib.request.urlopen")
def test_fetch_geo_sample_metadata_characteristic_without_colon_uses_fallback_name(mock_urlopen):
    # Real GEO series usually write "key: value" characteristics lines, but
    # the format doesn't enforce it -- a submitter can write a bare value
    # with no colon at all. Falls back to a positional "characteristic_N"
    # column name instead of crashing on an IndexError/ValueError trying to
    # split on a colon that isn't there.
    # No !Sample_title line on purpose: the fallback name is positional
    # ("characteristic_{len(fields) + 1}"), so an already-populated fields
    # dict (e.g. from a title column parsed first) would shift the number --
    # leaving title out keeps the expected name unambiguous.
    matrix = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_characteristics_ch1\t"lung"\t"liver"\n'
        "!series_matrix_table_begin\n"
        "some\tunrelated\tdata\n"
    )
    mock_urlopen.return_value = _FakeResponse(gzip.compress(matrix.encode("utf-8")))

    metadata = fetch_geo_sample_metadata("GSE52778")

    assert "characteristic_1" in metadata.columns
    assert metadata.loc["GSM1", "characteristic_1"] == "lung"
    assert metadata.loc["GSM2", "characteristic_1"] == "liver"


@patch("urllib.request.urlopen")
def test_fetch_geo_sample_metadata_duplicate_characteristic_names_get_suffixed(mock_urlopen):
    # Two separate !Sample_characteristics_ch1 lines that both happen to
    # use the same "key:" prefix (a real submitter mistake, e.g. two
    # differently-scoped "batch: ..." annotations) would otherwise silently
    # overwrite each other in the fields dict -- the second must be kept
    # under a suffixed name instead of clobbering the first.
    matrix = (
        '!Sample_title\t"sample 1"\t"sample 2"\n'
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_characteristics_ch1\t"batch: A"\t"batch: B"\n'
        '!Sample_characteristics_ch1\t"batch: X"\t"batch: Y"\n'
        "!series_matrix_table_begin\n"
        "some\tunrelated\tdata\n"
    )
    mock_urlopen.return_value = _FakeResponse(gzip.compress(matrix.encode("utf-8")))

    metadata = fetch_geo_sample_metadata("GSE52778")

    assert metadata.loc["GSM1", "batch"] == "A"
    assert metadata.loc["GSM2", "batch"] == "B"
    assert metadata.loc["GSM1", "batch_2"] == "X"
    assert metadata.loc["GSM2", "batch_2"] == "Y"


@patch("urllib.request.urlopen")
def test_fetch_geo_sample_metadata_parses_characteristics(mock_urlopen):
    gz_bytes = gzip.compress(_SAMPLE_SERIES_MATRIX.encode("utf-8"))
    mock_urlopen.return_value = _FakeResponse(gz_bytes)

    metadata = fetch_geo_sample_metadata("GSE52778")

    assert list(metadata.index) == ["GSM1", "GSM2"]
    assert metadata.index.name == "sample"
    assert metadata.loc["GSM1", "title"] == "sample 1"
    assert metadata.loc["GSM1", "tissue"] == "lung"
    assert metadata.loc["GSM2", "tissue"] == "liver"
    assert metadata.loc["GSM1", "treatment"] == "none"
    assert metadata.loc["GSM2", "treatment"] == "drug"


@patch("urllib.request.urlopen")
def test_fetch_geo_sample_metadata_missing_accession_line_raises(mock_urlopen):
    broken = '!Sample_title\t"sample 1"\n'
    mock_urlopen.return_value = _FakeResponse(gzip.compress(broken.encode("utf-8")))

    with pytest.raises(ValueError, match="Sample_geo_accession"):
        fetch_geo_sample_metadata("GSE52778")


def test_invalid_accession_rejected_before_any_network_call():
    with pytest.raises(ValueError):
        list_geo_supplementary_files("not-an-accession")
