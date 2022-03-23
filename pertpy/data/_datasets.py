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


def sc_sim_augur() -> AnnData:
    """Simulated test dataset used the Usage example for the Augur.

    https://github.com/neurorestore/Augur

    Args:
        file_path: Path to the dataset

    Returns:
        :class:`~anndata.AnnData` object of a simulated single-cell RNA seq dataset
    """
    output_file_name = "sc_sim_augur.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/17114054",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def bhattacherjee_rna() -> AnnData:
    """Single-cell data PFC adult mice under cocaine self-administration.

    Adult mice were subject to cocaine self-administration, samples were
    collected a three time points: Maintance, 48h after cocaine withdrawal and
    15 days after cocaine withdrawal.

    Args:
        file_path: Path to the dataset

    Reference:
        Bhattacherjee A, Djekidel MN, Chen R, Chen W, Tuesta LM, Zhang Y. Cell
        type-specific transcriptional programs in mouse prefrontal cortex during
        adolescence and addiction. Nat Commun. 2019 Sep 13;10(1):4169.
        doi: 10.1038/s41467-019-12054-3. PMID: 31519873; PMCID: PMC6744514.

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "bhattacherjee_rna.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/19397624",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def sciplex3() -> AnnData:
    """SciPlex3 perturbation dataset curated for perturbation modeling.

    Args:
        file_path: Path to the dataset

    Reference:
        Srivatsan SR, McFaline-Figueroa JL, Ramani V, Saunders L, Cao J, Packer J,
        Pliner HA, Jackson DL, Daza RM, Christiansen L, Zhang F, Steemers F,
        Shendure J, Trapnell C. Massively multiplex chemical transcriptomics at
        single-cell resolution. Science. 2020 Jan 3;367(6473):45-51.
        doi: 10.1126/science.aax6234. Epub 2019 Dec 5. PMID: 31806696; PMCID: PMC7289078.

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "sciplex3.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/19122572",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
