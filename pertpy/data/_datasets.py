from pathlib import Path

import muon as mu
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from scanpy import settings

from pertpy.data._dataloader import _download


def burczynski_crohn() -> AnnData:  # pragma: no cover
    """Bulk data with conditions ulcerative colitis (UC) and Crohn's disease (CD).

    The study assesses transcriptional profiles in peripheral blood mononuclear
    cells from 42 healthy individuals, 59 CD patients, and 26 UC patients by
    hybridization to microarrays interrogating more than 22,000 sequences.

    Reference:
        Burczynski et al., "Molecular classification of Crohn's disease and
        ulcerative colitis patients using transcriptional profiles in peripheral blood mononuclear cells"
        J Mol Diagn 8, 51 (2006). PMID:16436634.

    Returns:
        :class:`~anndata.AnnData` object of the Burczynski dataset
    """
    return sc.datasets.burczynski06()


def papalexi_2021() -> MuData:  # pragma: no cover
    """Dataset of the Mixscape Vignette.

    https://satijalab.org/seurat/articles/mixscape_vignette.html

    Reference:
    Papalexi, E., Mimitou, E.P., Butler, A.W. et al. Characterizing the molecular regulation
    of inhibitory immune checkpoints with multimodal single-cell screens.
    Nat Genet 53, 322–331 (2021). https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of the Crispr dataset
    """
    output_file_name = "papalexi_2021.h5mu"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/36509460",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        mudata = mu.read(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        mudata = mu.read(output_file_path)

    return mudata


def sc_sim_augur() -> AnnData:  # pragma: no cover
    """Simulated test dataset used the Usage example for the Augur.

    https://github.com/neurorestore/Augur

    Returns:
        :class:`~anndata.AnnData` object of a simulated single-cell RNA seq dataset
    """
    output_file_name = "sc_sim_augur.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/31645886",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def bhattacherjee() -> AnnData:  # pragma: no cover
    """Processed single-cell data PFC adult mice under cocaine self-administration.

    Adult mice were subject to cocaine self-administration, samples were
    collected a three time points: Maintance, 48h after cocaine withdrawal and
    15 days after cocaine withdrawal.

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
            url="https://figshare.com/ndownloader/files/34526528",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def sciplex3_raw() -> AnnData:  # pragma: no cover
    """Raw sciplex3 perturbation dataset curated for perturbation modeling.

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
            url="https://figshare.com/ndownloader/files/33979517",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def frangieh_2021() -> AnnData:  # pragma: no cover
    """Processed perturb-CITE-seq data with multi-modal RNA and protein single-cell profiling.

    We profiled RNA and 20 surface proteins in over 218,000 cells under ~750 perturbations,
    chosen by their membership in an immune evasion program that is associated with
    immunotherapy resistance in patients.

    Reference:
        Frangieh, C.J., Melms, J.C., Thakore, P.I. et al. Multimodal pooled Perturb-CITE-seq
        screens in patient models define mechanisms of cancer immune evasion.
        Nat Genet 53, 332–341 (2021). https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of a Perturb-CITE-seq data
    """
    output_file_name = "frangieh_2021.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34013717",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def frangieh_2021_raw() -> AnnData:  # pragma: no cover
    """Raw Perturb-CITE-seq data with multi-modal RNA and protein single-cell profiling readout.

    We profiled RNA and 20 surface proteins in over 218,000 cells under ~750 perturbations,
    chosen by their membership in an immune evasion program that is associated with
    immunotherapy resistance in patients.

    Reference:
        Frangieh, C.J., Melms, J.C., Thakore, P.I. et al. Multimodal pooled Perturb-CITE-seq
        screens in patient models define mechanisms of cancer immune evasion.
        Nat Genet 53, 332–341 (2021). https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of raw Perturb-CITE-seq data
    """
    output_file_name = "frangieh_2021.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34012565",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def dixit_2016_raw() -> AnnData:  # pragma: no cover
    """Perturb-seq: scRNA-seq with pooled CRISPR-KO perturbations.

    scRNA-seq with pooled CRISPR-KO perturbations in 200,000 cells across six screens
    unstimulated BMDC, BMDC stimulated at 3hr, TFs in K562 at 7 and 13 days post transduction,
    and 13 days at a higher MOI of perturbations.

    Reference:
        Dixit A, Parnas O, Li B, Chen J et al. Perturb-Seq: Dissecting Molecular Circuits with
        Scalable Single-Cell RNA Profiling of Pooled Genetic Screens.
        Cell 2016 Dec 15;167(7):1853-1866.e17. DOI:https://doi.org/10.1016/j.cell.2016.11.038

    Returns:
        :class:`~anndata.AnnData` object of raw Perturb-seq data
    """
    output_file_name = "dixit_2016_raw.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34012565",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def dixit_2016() -> AnnData:  # pragma: no cover
    """Perturb-seq: scRNA-seq with pooled CRISPR-KO perturbations.

    scRNA-seq with pooled CRISPR-KO perturbations in 200,000 cells across six screens
    unstimulated BMDC, BMDC stimulated at 3hr, TFs in K562 at 7 and 13 days post trasnduction,
    and 13 days at a higher MOI of perturbations.

    Reference:
        Dixit A, Parnas O, Li B, Chen J et al. Perturb-Seq: Dissecting Molecular Circuits with
        Scalable Single-Cell RNA Profiling of Pooled Genetic Screens.
        Cell 2016 Dec 15;167(7):1853-1866.e17. DOI:https://doi.org/10.1016/j.cell.2016.11.038

    Returns:
        :class:`~anndata.AnnData` object of Perturb-seq data
    """
    output_file_name = "dixit_2016.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34014608",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def norman_2019() -> AnnData:  # pragma: no cover
    """Processed single-cell, pooled CRISPR screening.

    Single-cell, pooled CRISPR screening experiment comparing the transcriptional effects of
    overexpressing genes alone or in combination

    Reference:
        Norman, Thomas M et al. “Exploring genetic interaction manifolds constructed from rich
        single-cell phenotypes.” Science (New York, N.Y.) vol. 365,6455 (2019): 786-793.
        doi:10.1126/science.aax4438

    Returns:
        :class:`~anndata.AnnData` object of sc pooled CRISPR screening
    """
    output_file_name = "norman_2019.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34027562",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def norman_2019_raw() -> AnnData:  # pragma: no cover
    """Raw single-cell, pooled CRISPR screening.

    Single-cell, pooled CRISPR screening experiment comparing the transcriptional effects of
    overexpressing genes alone or in combination

    Reference:
        Norman, Thomas M et al. “Exploring genetic interaction manifolds constructed from rich
        single-cell phenotypes.” Science (New York, N.Y.) vol. 365,6455 (2019): 786-793.
        doi:10.1126/science.aax4438

    Returns:
        :class:`~anndata.AnnData` object of sc pooled CRISPR screening
    """
    output_file_name = "norman_2019_raw.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34002548",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def dialogue_example() -> AnnData:  # pragma: no cover
    """Example dataset used to feature DIALOGUE.

    https://github.com/livnatje/DIALOGUE/wiki/Example

    Returns:
        :class:`~anndata.AnnData` object
    """
    output_file_name = "dialogue_example.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34490714",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def kang_2018() -> AnnData:  # pragma: no cover
    """Processed multiplexing droplet-based single cell RNA-sequencing using genetic barcodes

    HiSeq 2500 data for sequencing of PBMCs from SLE patients and 2 controls. We collected 1M cells
    each from frozen PBMC samples that were Ficoll isolated and prepared using the 10x Single Cell
    instrument according to standard protocol. Samples A, B, and C were prepared on the instrument
    directly following thaw, while samples 2.1 and 2.2 were cultured for 6 hours with (B) or
    without (A) IFN-beta stimulation prior to loading on the 10x Single Cell instrument.

    Reference:
        Kang, H., Subramaniam, M., Targ, S. et al. Multiplexed droplet single-cell RNA-sequencing
        using natural genetic variation.
        Nat Biotechnol 36, 89–94 (2018). https://doi.org/10.1038/nbt.4042

    Returns:
        :class:`~anndata.AnnData` object of droplet-based single cell RNA-sequencing
    """
    output_file_name = "kang_2018.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34464122",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def stephenson_2021_subsampled() -> AnnData:  # pragma: no cover
    """Processed 10X 5' scRNA-seq data from PBMC of COVID-19 patients and healthy donors

    The study profiled peripheral blood mononuclear cells from 90 COVID-19 patients with different disease severity and 23 healthy control donors.
    Here the dataset was downsampled to approx. 500 cells per donor and cells were mapped to a reference atlas of healthy PBMCs from 12 studies
    with scArches.

    Reference:
        Stephenson, E., Reynolds, G., Botting, R. A., et al. (2021).
        Single-cell multi-omics analysis of the immune response in COVID-19.
        Nature Medicine, 27(5). https://doi.org/10.1038/s41591-021-01329-2


    Returns:
        :class:`~anndata.AnnData` object of scRNA-seq profiles
    """
    output_file_name = "stephenson_2021_subsampled.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/38171703",
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)
    return adata


def haber_2017_regions() -> AnnData:  # pragma: no cover
    """Raw single-cell, pooled CRISPR screening.

    Epithelial cells from the small intestine and organoids of mice.
    Some of the cells were also subject to Salmonella or Heligmosomoides polygyrus infection (day 3 and day 10).

    Reference:
        Haber, Adam L. et al. “A single-cell survey of the small intestinal epithelium” Nature vol. 551 (2017): 333-339
        doi:10.1038/nature24489

    Returns:
        :class:`~anndata.AnnData` object
    """
    output_file_name = "haber_2017_regions.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/38169900",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)
    return adata


def Adamson_2016_pilot() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.048

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Adamson_2016_pilot.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/AdamsonWeissman2016_GSM2406675_10X001.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Adamson_2016_upr_epistasis() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.048

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Adamson_2016_upr_epistasis.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/AdamsonWeissman2016_GSM2406677_10X005.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Adamson_2016_upr_perturb_seq() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.048

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Adamson_2016_upr_perturb_seq.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Aissa_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41467-021-21884-z

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Aissa_2021.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/AissaBenevolenskaya2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Chang_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41587-021-01005-3

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Chang_2021.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/ChangYe2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Datlinger_2017() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/nmeth.4177

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Datlinger_2017.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/DatlingerBock2017.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Datlinger_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41592-021-01153-z

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Datlinger_2021.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/DatlingerBock2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Dixit_2016() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.038

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Dixit_2016.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/DixitRegev2016.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Frangieh_2021_protein() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Frangieh_2021_protein.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/FrangiehIzar2021_protein.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Frangieh_2021_RNA() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Frangieh_2021_RNA.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/FrangiehIzar2021_RNA.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Gasperini_2019_atscale() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2018.11.029

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Gasperini_2019_atscale.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/GasperiniShendure2019_atscale.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Gasperini_2019_highMOI() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2018.11.029

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Gasperini_2019_highMOI.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/GasperiniShendure2019_highMOI.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Gasperini_2019_lowMOI() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2018.11.029

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Gasperini_2019_lowMOI.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/GasperiniShendure2019_lowMOI.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Gehring_2019() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41587-019-0372-z

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Gehring_2019.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/GehringPachter2019.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def McFarland_2020() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41467-020-17440-w

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "McFarland_2020.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/McFarlandTsherniak2020.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Norman_2019_filtered() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1126/science.aax4438

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Norman_2019_filtered.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/NormanWeissman2019_filtered.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Papalexi_2021_eccite_arrayed_protein() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Papalexi_2021_eccite_arrayed_protein.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/PapalexiSatija2021_eccite_arrayed_protein.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Papalexi_2021_eccite_arrayed_RNA() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Papalexi_2021_eccite_arrayed_RNA.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/PapalexiSatija2021_eccite_arrayed_RNA.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Papalexi_2021_eccite_protein() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Papalexi_2021_eccite_protein.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/PapalexiSatija2021_eccite_protein.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Papalexi_2021_eccite_RNA() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Papalexi_2021_eccite_RNA.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/PapalexiSatija2021_eccite_RNA.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Replogle_2022_K562_essential() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2022.05.013

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Replogle_2022_K562_essential.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/ReplogleWeissman2022_K562_essential.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Replogle_2022_K562_gwps() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2022.05.013

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Replogle_2022_K562_gwps.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/ReplogleWeissman2022_K562_gwps.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Replogle_2022_rpe1() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2022.05.013

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Replogle_2022_rpe1.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/ReplogleWeissman2022_rpe1.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Schiebinger_2019_GSE106340() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2019.01.006

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Schiebinger_2019_.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/SchiebingerLander2019_GSE106340.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Schiebinger_2019_GSE115943() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2019.01.006

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Schiebinger_2019_.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/SchiebingerLander2019_GSE115943.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Schraivogel_2020_TAP_SCREEN_chr11() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41592-020-0837-5

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Schraivogel_2020_TAP_SCREEN_chr11.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_11_screen.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Schraivogel_2020_TAP_SCREEN_chr8() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41592-020-0837-5

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Schraivogel_2020_TAP_SCREEN_chr8.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_8_screen.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Shifrut_2018() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.cell.2018.10.024

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Shifrut_2018.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/ShifrutMarson2018.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    
    
def Srivatsan_2020_sciplex2() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1126/science.aax6234

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Srivatsan_2020_sciplex2.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/SrivatsanTrapnell2020_sciplex2.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Srivatsan_2020_sciplex3() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1126/science.aax6234

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Srivatsan_2020_sciplex3.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Srivatsan_2020_sciplex4() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1126/science.aax6234

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Srivatsan_2020_sciplex4.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/SrivatsanTrapnell2020_sciplex4.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Tian_2019_day7neuron() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.neuron.2019.07.014

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Tian_2019_day7neuron.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/TianKampmann2019_day7neuron.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Tian_2019_iPSC() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.neuron.2019.07.014

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Tian_2019_iPSC.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/TianKampmann2019_iPSC.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Tian_2021_CRISPRa() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41593-021-00862-0

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Tian_2021_CRISPRa.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/TianKampmann2021_CRISPRa.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Tian_2021_CRISPRi() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1038/s41593-021-00862-0

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Tian_2021_CRISPRi.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/TianKampmann2021_CRISPRi.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
    

def Weinreb_2020() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1126/science.aaw3381

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Weinreb_2020.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/WeinrebKlein2020.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Xie_2017() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1016/j.molcel.2017.03.007

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "Xie_2017.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/XieHon2017.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata


def Zhao_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.
    
    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.
       
    Source paper:
        https://doi.org/10.1186/s13073-021-00894-y

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "ZhaoSims2021.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/7278143/files/ZhaoSims2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
        adata = sc.read_h5ad(filename=settings.datasetdir.__str__() + "/" + output_file_name)
    else:
        adata = sc.read_h5ad(output_file_path)

    return adata
