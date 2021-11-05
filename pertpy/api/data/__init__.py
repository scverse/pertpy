from anndata import AnnData

from pertpy.api.data._datasets import DataSets


def burczynski_crispr(file_path: str = None) -> AnnData:
    """Downloads and returns the dataset of the Mixscape Vignette.

    https://satijalab.org/seurat/articles/mixscape_vignette.html

    Args:
        file_path: Path to the dataset

    Returns:
        :class:`~anndata.AnnData` object of the Crispr dataset
    """
    return DataSets.burczynski_crispr(file_path=file_path)


def burczynski_crohn(file_path: str = None) -> AnnData:
    """Downloads and returns the dataset of

    Args:
        file_path:

    Returns:

    """
    return DataSets.burczynski_crohn(file_path=file_path)
