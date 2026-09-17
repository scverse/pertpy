from pathlib import Path

import pytest
from scanpy import settings

from pertpy.metadata._metadata import MetaData


@pytest.fixture
def cachedir(tmp_path):
    original = settings.cachedir
    settings.cachedir = tmp_path
    yield tmp_path
    settings.cachedir = original


def test_download_metadata_skips_when_cached(cachedir, monkeypatch):
    cached = cachedir / "depmap_23Q4_info.csv"
    cached.write_text("already here")

    calls = []
    monkeypatch.setattr("pertpy.metadata._metadata._download", lambda **kwargs: calls.append(kwargs))

    path = MetaData._download_metadata("depmap_23Q4_info.csv")

    assert path == cached
    assert calls == []


def test_download_metadata_fetches_from_base_url_when_missing(cachedir, monkeypatch):
    calls = []
    monkeypatch.setattr("pertpy.metadata._metadata._download", lambda **kwargs: calls.append(kwargs))

    path = MetaData._download_metadata("gdsc1_info.csv")

    assert path == cachedir / "gdsc1_info.csv"
    assert len(calls) == 1
    assert calls[0]["url"] == "https://exampledata.scverse.org/pertpy/gdsc1_info.csv"
    assert calls[0]["output_file_name"] == "gdsc1_info.csv"
    assert Path(calls[0]["output_path"]) == cachedir
