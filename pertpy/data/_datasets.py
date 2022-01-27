from pathlib import Path

import muon as mu
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scanpy import settings

from pertpy.data._dataloader import _download


def burczynski_crohn() -> AnnData:
    """Bulk data with conditions ulcerative colitis (UC) and Crohn's disease (CD).

    The study assesses transcriptional profiles in peripheral blood mononuclear
    cells from 42 healthy individuals, 59 CD patients, and 26 UC patients by
    hybridization to microarrays interrogating more than 22,000 sequences.

    Args:
        file_path: Path to the dataset

    Reference:
        Burczynski et al., "Molecular classification of Crohn's disease and
        ulcerative colitis patients using transcriptional profiles in peripheral blood mononuclear cells"
        J Mol Diagn 8, 51 (2006). PMID:16436634.

    Returns:
        :class:`~anndata.AnnData` object of the Burczynski dataset
    """
    return sc.datasets.burczynski06()


def burczynski_crispr() -> MuData:
    """Dataset of the Mixscape Vignette.

    https://satijalab.org/seurat/articles/mixscape_vignette.html

    Args:
        file_path: Path to the dataset

    Returns:
        :class:`~anndata.AnnData` object of the Crispr dataset
    """
    output_file_name = "burczynski_crispr.h5mu"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/31645901",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        mudata = mu.read(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        mudata = mu.read(output_file_path)

    return mudata
