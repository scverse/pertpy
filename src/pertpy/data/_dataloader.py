import logging
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import pooch  # type: ignore[import-untyped]
import requests
from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)
pooch.get_logger().setLevel(logging.WARNING)

# Transient network failures that are worth retrying.
# ``urllib.error.URLError`` (e.g. "Connection reset by peer") is a subclass of ``OSError``,
# while pooch's ``requests``-based downloader raises ``requests.exceptions.RequestException``.
_TRANSIENT_DOWNLOAD_ERRORS = (OSError, requests.exceptions.RequestException)

# The longest registered dataset filename is currently 38 characters.
_FILENAME_FIELD_WIDTH = 45


class _RichProgress:
    """Adapter exposing the tqdm-like interface pooch expects, backed by rich.progress."""

    def __init__(self, description: str = "[red]Downloading..."):
        self._description = description
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        self.total: int | None = None

    def _ensure_started(self) -> tuple[Progress, TaskID]:
        if self._progress is None or self._task is None:
            self._progress = Progress(refresh_per_second=3)
            self._progress.start()
            self._task = self._progress.add_task(self._description, total=self.total or None)
        return self._progress, self._task

    def update(self, n: int) -> None:
        progress, task = self._ensure_started()
        if self.total is not None and progress.tasks[task].total != self.total:
            progress.update(task, total=self.total or None)
        progress.update(task, advance=n)

    def reset(self) -> None:
        progress, task = self._ensure_started()
        progress.reset(task, total=self.total or None)

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task = None


def _download(  # pragma: no cover
    url: str,
    *,
    output_file_name: str | None = None,
    output_path: str | Path | None = None,
    block_size: int = 8192,
    overwrite: bool = False,
    is_zip: bool = False,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Path:
    """Download a dataset via pooch with a rich progress bar.

    Args:
        url: URL to download.
        output_file_name: Name of the downloaded file. Inferred from the URL if not provided.
        output_path: Directory to download/extract the files to. Defaults to the system temp dir.
        block_size: Chunk size for the HTTP stream in bytes.
        overwrite: Whether to overwrite an existing file.
        is_zip: Whether the downloaded archive should be extracted into `output_path`.
        timeout: Per-request timeout in seconds.
        max_retries: Maximum number of retries on transient network errors.
        retry_delay: Delay in seconds between retries.

    Returns:
        The path of the downloaded file, or `output_path` if `is_zip` is True.
    """
    if output_path is None:
        output_path = tempfile.gettempdir()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_file_name is None:
        output_file_name = url.rsplit("/", 1)[-1]

    target = output_path / output_file_name
    if overwrite and target.exists():
        target.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            pooch.retrieve(
                url=url,
                known_hash=None,
                fname=output_file_name,
                path=str(output_path),
                downloader=pooch.HTTPDownloader(
                    progressbar=_RichProgress(
                        description=f"[red]Downloading {output_file_name:<{_FILENAME_FIELD_WIDTH}}"
                    ),
                    chunk_size=block_size,
                    timeout=timeout,
                ),
            )
            break
        except _TRANSIENT_DOWNLOAD_ERRORS as e:
            if attempt >= max_retries:
                logger.error("Download of %s failed after %d attempts: %s", url, max_retries, e)
                raise
            logger.warning(
                "Download of %s failed (attempt %d/%d): %s. Retrying in %d seconds...",
                url,
                attempt,
                max_retries,
                e,
                retry_delay,
            )
            time.sleep(retry_delay)

    if is_zip:
        with ZipFile(target, "r") as zip_obj:
            zip_obj.extractall(path=output_path)
        return output_path

    return target
