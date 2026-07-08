import logging
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pooch
from rich.progress import Progress, TaskID

pooch.get_logger().setLevel(logging.WARNING)


class _RichProgress:
    """Adapter exposing the tqdm-like interface pooch expects, backed by rich.progress."""

    def __init__(self, description: str = "[red]Downloading..."):
        self._description = description
        self._progress: Progress | None = None
        self._task: TaskID | None = None
        self.total: int | None = None

    def _ensure_started(self) -> None:
        if self._progress is None:
            self._progress = Progress(refresh_per_second=3)
            self._progress.start()
            self._task = self._progress.add_task(self._description, total=self.total or None)

    def update(self, n: int) -> None:
        self._ensure_started()
        if self.total is not None and self._progress.tasks[self._task].total != self.total:
            self._progress.update(self._task, total=self.total or None)
        self._progress.update(self._task, advance=n)

    def reset(self) -> None:
        self._ensure_started()
        self._progress.reset(self._task, total=self.total or None)

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task = None


def _download(  # pragma: no cover
    url: str,
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
        max_retries: Unused; retained for backwards compatibility.
        retry_delay: Unused; retained for backwards compatibility.

    Returns:
        The path of the downloaded file, or `output_path` if `is_zip` is True.
    """
    del max_retries, retry_delay

    if output_path is None:
        output_path = tempfile.gettempdir()
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_file_name is None:
        output_file_name = url.rsplit("/", 1)[-1]

    target = output_path / output_file_name
    if overwrite and target.exists():
        target.unlink()

    pooch.retrieve(
        url=url,
        known_hash=None,
        fname=output_file_name,
        path=str(output_path),
        downloader=pooch.HTTPDownloader(
            progressbar=_RichProgress(),
            chunk_size=block_size,
            timeout=timeout,
        ),
    )

    if is_zip:
        with ZipFile(target, "r") as zip_obj:
            zip_obj.extractall(path=output_path)
        return output_path

    return target
