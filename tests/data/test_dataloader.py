import pytest
import requests

from pertpy.data import _dataloader


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Keep retries instant so the tests never actually wait."""
    monkeypatch.setattr(_dataloader.time, "sleep", lambda _seconds: None)


def test_download_retries_on_transient_error_then_succeeds(monkeypatch, tmp_path):
    """A transient network error is retried and the download eventually succeeds."""
    calls = []

    def flaky_retrieve(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise requests.exceptions.ConnectionError("Connection reset by peer")

    monkeypatch.setattr(_dataloader.pooch, "retrieve", flaky_retrieve)

    target = _dataloader._download(
        url="https://example.com/pertpy/data.h5ad",
        output_path=tmp_path,
        max_retries=3,
        retry_delay=0,
    )

    assert len(calls) == 3
    assert target == tmp_path / "data.h5ad"


def test_download_raises_after_exhausting_retries(monkeypatch, tmp_path):
    """The last exception is re-raised once ``max_retries`` attempts are used up."""
    calls = []

    def always_reset(**kwargs):
        calls.append(kwargs)
        raise OSError("Connection reset by peer")

    monkeypatch.setattr(_dataloader.pooch, "retrieve", always_reset)

    with pytest.raises(OSError, match="Connection reset by peer"):
        _dataloader._download(
            url="https://example.com/pertpy/data.h5ad",
            output_path=tmp_path,
            max_retries=2,
            retry_delay=0,
        )

    assert len(calls) == 2


def test_download_does_not_retry_on_non_transient_error(monkeypatch, tmp_path):
    """Programming errors are not swallowed by the retry loop."""
    calls = []

    def boom(**kwargs):
        calls.append(kwargs)
        raise ValueError("bad argument")

    monkeypatch.setattr(_dataloader.pooch, "retrieve", boom)

    with pytest.raises(ValueError, match="bad argument"):
        _dataloader._download(
            url="https://example.com/pertpy/data.h5ad",
            output_path=tmp_path,
            max_retries=3,
            retry_delay=0,
        )

    assert len(calls) == 1
