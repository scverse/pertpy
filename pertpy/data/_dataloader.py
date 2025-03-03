import tempfile
from pathlib import Path
from random import choice
from string import ascii_lowercase
from zipfile import ZipFile

import requests
from filelock import FileLock
from lamin_utils import logger
from rich.progress import Progress


def _download(  # pragma: no cover
    url: str,
    output_file_name: str = None,
    output_path: str | Path = None,
    block_size: int = 1024,
    overwrite: bool = False,
    is_zip: bool = False,
) -> None:
    """Downloads a dataset irrespective of the format.

    Args:
        url: URL to download
        output_file_name: Name of the downloaded file
        output_path: Path to download/extract the files to.
        block_size: Block size for downloads in bytes.
        overwrite: Whether to overwrite existing files.
        is_zip: Whether the downloaded file needs to be unzipped.
    """
    if output_file_name is None:
        letters = ascii_lowercase
        output_file_name = f"pertpy_tmp_{''.join(choice(letters) for _ in range(10))}"

    if output_path is None:
        output_path = tempfile.gettempdir()

    download_to_path = Path(output_path) / output_file_name

    Path(output_path).mkdir(parents=True, exist_ok=True)
    lock_path = Path(output_path) / f"{output_file_name}.lock"

    try:
        with FileLock(lock_path, timeout=300):
            if Path(download_to_path).exists() and not overwrite:
                logger.warning(f"File {download_to_path} already exists!")
                return

            temp_file_name = Path(f"{download_to_path}.part")

            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise exception for HTTP errors
                total = int(response.headers.get("content-length", 0))

                with Progress(refresh_per_second=5) as progress:
                    task = progress.add_task("[red]Downloading...", total=total)
                    with Path(temp_file_name).open("wb") as file:
                        for data in response.iter_content(block_size):
                            file.write(data)
                            progress.update(task, advance=len(data))  # Use actual data length
                        progress.update(task, completed=total, refresh=True)

                Path(temp_file_name).replace(download_to_path)

                if is_zip:
                    output_path = output_path or tempfile.gettempdir()
                    with ZipFile(download_to_path, "r") as zip_obj:
                        zip_obj.extractall(path=output_path)
            except Exception as e:
                logger.error(f"Download failed: {str(e)}")
                if Path(temp_file_name).exists():
                    Path(temp_file_name).unlink(missing_ok=True)
                raise
    finally:
        if Path(lock_path).exists():
            try:
                Path(lock_path).unlink(missing_ok=True)
            except OSError:
                pass
