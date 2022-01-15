import scanpy as sc
from anndata import AnnData
from scanpy import settings


def burczynski_crohn(file_path: str = None) -> AnnData:
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
    adata = sc.read(
        filename=settings.datasetdir + file_path,
        backup_url="ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS1nnn/GDS1615/soft/GDS1615_full.soft.gz",
    )

    return adata

@staticmethod
def burczynski_crispr(file_path: str = None) -> AnnData:
    """Dataset of the Mixscape Vignette.

    https://satijalab.org/seurat/articles/mixscape_vignette.html

    Args:
        file_path: Path to the dataset

    Returns:
        :class:`~anndata.AnnData` object of the Crispr dataset
    """
    # TODO Add custom figshare backup URL
    adata = sc.read(filename=settings.datasetdir + file_path)

    return adata
