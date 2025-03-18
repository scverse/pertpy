import shutil
import tempfile
import time
from pathlib import Path
from random import choice
from string import ascii_lowercase
from zipfile import ZipFile

import requests
from filelock import FileLock
from lamin_utils import logger
from requests.exceptions import RequestException
from rich.progress import Progress


def _download(  # pragma: no cover
    url: str,
    output_file_name: str = None,
    output_path: str | Path = None,
    block_size: int = 1024,
    overwrite: bool = False,
    is_zip: bool = False,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Path:
    """Downloads a dataset irrespective of the format.

    Args:
        url: URL to download
        output_file_name: Name of the downloaded file
        output_path: Path to download/extract the files to.
        block_size: Block size for downloads in bytes.
        overwrite: Whether to overwrite existing files.
        is_zip: Whether the downloaded file needs to be unzipped.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.
    """
    if output_file_name is None:
        letters = ascii_lowercase
        output_file_name = f"pertpy_tmp_{''.join(choice(letters) for _ in range(10))}"

    if output_path is None:
        output_path = tempfile.gettempdir()

    download_to_path = Path(output_path) / output_file_name

    Path(output_path).mkdir(parents=True, exist_ok=True)
    lock_path = Path(output_path) / f"{output_file_name}.lock"

    with FileLock(lock_path, timeout=300):
        if Path(download_to_path).exists() and not overwrite:
            logger.warning(f"File {download_to_path} already exists!")
            return download_to_path

        temp_file_name = Path(f"{download_to_path}.part")

        retry_count = 0
        while retry_count <= max_retries:
            try:
                head_response = requests.head(url, timeout=timeout)
                head_response.raise_for_status()
                content_length = int(head_response.headers.get("content-length", 0))

                free_space = shutil.disk_usage(output_path).free
                if content_length > free_space:
                    raise OSError(
                        f"Insufficient disk space. Need {content_length} bytes, but only {free_space} available."
                    )

                response = requests.get(url, stream=True)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))

                with Progress(refresh_per_second=5) as progress:
                    task = progress.add_task("[red]Downloading...", total=total)
                    with Path(temp_file_name).open("wb") as file:
                        for data in response.iter_content(block_size):
                            file.write(data)
                            progress.update(task, advance=len(data))
                        progress.update(task, completed=total, refresh=True)

                Path(temp_file_name).replace(download_to_path)

                if is_zip:
                    with ZipFile(download_to_path, "r") as zip_obj:
                        zip_obj.extractall(path=output_path)
                    return Path(output_path)

                return download_to_path
            except (OSError, RequestException) as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.warning(
                        f"Download attempt {retry_count}/{max_retries} failed: {str(e)}. Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Download failed after {max_retries} attempts: {str(e)}")
                    if Path(temp_file_name).exists():
                        Path(temp_file_name).unlink(missing_ok=True)
                    raise

            except Exception as e:
                logger.error(f"Download failed: {str(e)}")
                if Path(temp_file_name).exists():
                    Path(temp_file_name).unlink(missing_ok=True)
                raise
            finally:
                if Path(temp_file_name).exists():
                    Path(temp_file_name).unlink(missing_ok=True)

        return Path(download_to_path)
