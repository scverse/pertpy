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


def smillie() -> AnnData:  # pragma: no cover
    """scRNA-seq data of the small intestine of mice under Ulcerative Colitis.

    Reference:
        Smillie, Christopher S et al. “Intra- and Inter-cellular Rewiring of the Human Colon during Ulcerative Colitis.”
        Cell vol. 178,3 (2019): 714-730.e22. doi:10.1016/j.cell.2019.06.029

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "smillie.h5ad"
    output_file_path = settings.datasetdir.__str__() + "/" + output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/38648585",
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


def adamson_2016_pilot() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    In a pilot experiment, single-cell RNA-seq was performed on a pool of individually
    transduced chronic myeloid leukemia cells (K562) carrying 8 distinct GBCs, analyzing
    ∼6,000 cells total.

    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.048

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "adamson_2016_pilot.h5ad"
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


def adamson_2016_upr_epistasis() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.


    In UPR epistasis experiment, Perturb-seq was applied to explore the branches of
    the mammalian UPR. Using the three-guide Perturb-seq vector, sgRNAs targeting each
    UPR sensor gene were introduced into K562 cells with dCas9-KRAB. Transduced cells
    were then pooled, sorted for vector delivery, and after 5 days of total growth,
    treated with thapsigargin. Control cells were treated with DMSO. Transcriptomes
    of ∼15,000 cells were sequenced.

    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.048

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "adamson_2016_upr_epistasis.h5ad"
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


def adamson_2016_upr_perturb_seq() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    In UPR Perturb-seq experiment, Perturb-seq was applied to a small CRISPRi library
    of 91 sgRNAs targeting 82 genes. sgRNAs were delivered via pooled transduction
    using a mixture of separately prepared lentiviruses, and ∼65,000 transcriptomes
    were collected in one large pooled experiment.

    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.048

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "adamson_2016_upr_perturb_seq.h5ad"
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


def aissa_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Source paper:
        https://doi.org/10.1038/s41467-021-21884-z

    In this studym 848 PC9 cells subjected to consecutive erlotinib treatment (for
    1, 2, 4, 9, and 11 days) and 756 control cells were analysed using Drop-seq.

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "aissa_2021.h5ad"
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


def chang_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    TraCe-seq is a method that captures at clonal resolution the origin, fate and
    differential early adaptive transcriptional programs of cells in a complex
    population in response to distinct treatments. Here, a unique TraCe-seq barcode was
    transduced into five different cell lines (PC9, MCF-10A, MDA-MB-231, NCI-H358 and NCI-H1373).
    Transduced cells were selected with puromycin only, mixed together and profiled by 10x scRNA-seq.

    Source paper:
        https://doi.org/10.1038/s41587-021-01005-3

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "chang_2021.h5ad"
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


def datlinger_2017() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    For CROP-seq, Jurkat cells were transduced with a gRNA library targeting high-level
    regulators of T cell receptor signaling and a set of transcription factors. After 10
    days of antibiotic selection and expansion, cells were stimulated with anti-CD3 and
    anti-CD28 antibodies or left untreated. Both conditions were analyzed using CROP-seq,
    measuring TCR activation for each gene knockout. The dataset comprises 5,905 high-quality
    single-cell transcriptomes with uniquely assigned gRNAs.

    Source paper:
        https://doi.org/10.1038/nmeth.4177

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "datlinger_2017.h5ad"
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


def datlinger_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

     https://github.com/sanderlab/scPerturb

     Reference:
        Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
        Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
        scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
        bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    A large-scale scifi-RNA-seq experiment was performed with 383,000 nuclei loaded into
    a single microfluidic channel of the Chromium system. Four human cell lines (HEK293T,
    Jurkat, K562 and NALM-6) were combined in an equal mixture, and technical replicates of
    each cell line with different preindexing (round1) barcodes were marked. This experiment
    resulted in 151,788 single-cell transcriptomes passing quality control.

     Source paper:
         https://doi.org/10.1038/s41592-021-01153-z

     Returns:
         :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "datlinger_2021.h5ad"
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


def dixit_2016_scperturb() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Six Perturb-seq experiments were performed, analyzing 200,000 cells. In bone
    marrow-derived dendritic cells (BMDCs), 24 transcription factors (TFs) were targeted
    and the effects pre-stimulation (0 hr) and at 3 hr post-lipopolysaccharide (LPS) were
    measured. In K562 cells, 14 TFs and 10 cell-cycle regulators were targeted in separate
    pooled experiments. For K562 TFs, experiments were performed using lower and higher MOI
    and at two time points. Reference scRNA-seq data from unperturbed cells were collected
    separately.

    Source paper:
        https://doi.org/10.1016/j.cell.2016.11.038

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "dixit_2016_scperturb.h5ad"
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


def frangieh_2021_protein() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Perturb-CITE-seq was developed for pooled CRISPR perturbation screens with multi-modal
    RNA and protein single-cell profiling readout and applied to screen patient-derived
    autologous melanoma and tumor infiltrating lymphocyte (TIL) co-cultures. RNA and 20
    surface proteins were profiled in over 218,000 cells under ~750 perturbations, chosen
    by their membership in an immune evasion program that is associated with immunotherapy
    resistance in patients.

    Source paper:
        https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "frangieh_2021_protein.h5ad"
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


def frangieh_2021_rna() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Perturb-CITE-seq was developed for pooled CRISPR perturbation screens with multi-modal
    RNA and protein single-cell profiling readout and applied to screen patient-derived
    autologous melanoma and tumor infiltrating lymphocyte (TIL) co-cultures. RNA and 20
    surface proteins were profiled in over 218,000 cells under ~750 perturbations, chosen
    by their membership in an immune evasion program that is associated with immunotherapy
    resistance in patients.

    Source paper:
        https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "frangieh_2021_rna.h5ad"
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


def gasperini_2019_atscale() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Across two experiments, the authors used dCas9-KRAB to perturb 5,920 candidate enhancers
    with no strong a priori hypothesis as to their target gene(s) in 254,974 cells of chronic
    myelogenous leukemia cell line K562, with CRISPRi as the mode of perturbation.

    Source paper:
        https://doi.org/10.1016/j.cell.2018.11.029

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "gasperini_2019_atscale.h5ad"
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


def gasperini_2019_highmoi() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    The authors used dCas9-KRAB to perturb 1,119 candidate enhancers with no strong a priori
    hypothesis as to their target gene(s) in the chronic myelogenous leukemia cell line
    K562, with CRISPRi as the mode of perturbation, where K562 cells were transduced
    at a high MOI (pilot library MOI = ∼15).

    Source paper:
        https://doi.org/10.1016/j.cell.2018.11.029

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "gasperini_2019_highmoi.h5ad"
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


def gasperini_2019_lowmoi() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    The authors used dCas9-KRAB to perturb 1,119 candidate enhancers with no strong a priori
    hypothesis as to their target gene(s) in chronic myelogenous leukemia cell line K562,
    with CRISPRi as the mode of perturbation, where K562 cells were transduced at a
    low MOI (pilot library MOI = ∼1).

    Source paper:
        https://doi.org/10.1016/j.cell.2018.11.029

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "gasperini_2019_lowmoi.h5ad"
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


def gehring_2019() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    In this study, a 96-plex perturbation experiment was conducted on live mouse neural
    stem cells (NSCs), consisting of a pair of drug-triples with 4 drugs in total at 3
    or 4 different concentractions.

    Source paper:
        https://doi.org/10.1038/s41587-019-0372-z

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "gehring_2019.h5ad"
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


def mcfarland_2020() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Here, the authors developed MIX-Seq, a method for multiplexed transcriptional profiling
    of post-perturbation responses across many cell contexts, using scRNA-seq applied to
    co-treated pools of cancer cell lines. The responses of pools of 24–99 cell lines to
    a range of different drugs were profiled, as well as to CRISPR perturbations.

    Source paper:
        https://doi.org/10.1038/s41467-020-17440-w

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "mcfarland_2020.h5ad"
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


def norman_2019_filtered() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    K562 cells stably expressing dCas9-KRAB were transduced with the CRISPRa genetic
    interactions (GIs) library. 132 gene pairs were picked from the GI map, chosen
    both within and between blocks of genes with similar interaction profiles, and
    targeted each with CRISPRa sgRNA pairs. In total, transcriptional readouts for
    287 perturbations measured across ~110,000 single cells were obtained (median
    273 cells per condition) in one pooled experiment.


    Source paper:
        https://doi.org/10.1126/science.aax4438

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "norman_2019_filtered.h5ad"
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


def papalexi_2021_eccite_arrayed_protein() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    A pilot experiment was performed using gRNA species targeting PD-L1 or IFNGR1 as
    well as NT controls on simulated THP-1 cells. Libraries were sequenced on a
    NextSeq 500 system. ADT libraries were processed with CITE-seq-Count and
    normalized across cells using the centered log ratio. Cells with high mitochondrial
    gene content (>8%) were removed.


    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "papalexi_2021_eccite_arrayed_protein.h5ad"
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


def papalexi_2021_eccite_arrayed_rna() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    A pilot experiment was performed using gRNA species targeting PD-L1 or IFNGR1 as
    well as NT controls on simulated THP-1 cells. Libraries were sequenced on a NextSeq
    500 system. mRNA libraries were quantified using Cell Ranger (2.1.1; hg19 reference)
    and normalized using standard log normalization in Seurat.

    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "papalexi_2021_eccite_arrayed_rna.h5ad"
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


def papalexi_2021_eccite_protein() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    In an ECCITE-seq experiment, THP-1 Cas9-inducible cells were transduced with virus
    containing 111 guides at a low MOI to obtain cells with one gRNA and then stimulated.
    Samples were hashed following the cell hashing protocol. Protein (ADTs) libraries
    were constructed by following 10x Genomics and ECCITE-seq protocols, and then sequenced
    together on two lanes of a NovaSeq run. The CITE-seq-Count package was used to generate
    count matrices for ADT libraries.

    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "papalexi_2021_eccite_protein.h5ad"
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


def papalexi_2021_eccite_rna() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    In an ECCITE-seq experiment, THP-1 Cas9-inducible cells were transduced with virus
    containing 111 guides at a low MOI to obtain cells with one gRNA and then stimulated.
    Samples were hashed following the cell hashing protocol. mRNA Libraries were constructed
    by following 10x Genomics and ECCITE-seq protocols, and then sequenced together on
    two lanes of a NovaSeq run. Sequencing reads from the mRNA library were mapped to the
    hg19 reference genome using Cell Ranger software (version 2.1.1).

    Source paper:
        https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "papalexi_2021_eccite_rna.h5ad"
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


def replogle_2022_k562_essential() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    For day 6 essential-scale experiment in chronic myeloid leukemia (CML) (K562) cell
    lines, library lentivirus was packaged into lentivirus in 293T cells and empirically
    measured in K562 cells to obtain viral titers. CRISPRi K562 cells were transduced
    and 20Q1 Cancer Dependency Map common essential genes were targeted at day 7 after
    transduction.

    Source paper:
        https://doi.org/10.1016/j.cell.2022.05.013

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "replogle_2022_k562_essential.h5ad"
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


def replogle_2022_k562_gwps() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Here, the authors used a compact, multiplexed CRISPR interference (CRISPRi) library
    to assay thousands of loss-of-function genetic perturbations with single-cell RNA sequencing
    in chronic myeloid leukemia (CML) (K562) cell lines. For the K562 day 8 genome-scale
    Perturb-seq experiment, library lentivirus was packaged into lentivirus in 293T cells and
    empirically measured in K562 cells to obtain viral titers. CRISPRi K562 cells were transduced
    and all expressed genes were targeted at day 8 after transduction

    Source paper:
        https://doi.org/10.1016/j.cell.2022.05.013

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "replogle_2022_k562_gwps.h5ad"
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


def replogle_2022_rpe1() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    For day 7 essential-scale Perturb-seq experiment in retinal pigment epithelial (RPE1)
    cell lines, library lentivirus was packaged into lentivirus in 293T cells and
    empirically measured in RPE1 cells to obtain viral titers. CRISPRi RPE1 cells expressing
    ZIM3 KRAB-dCas9-P2A-BFP were transduced. 20Q1 Cancer Dependency Map common essential
    genes were targeted at day 7 after transduction.

    Source paper:
        https://doi.org/10.1016/j.cell.2022.05.013

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "replogle_2022_rpe1.h5ad"
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


def schiebinger_2019_16day() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Samples were collected from established iPSC lines reprogrammed from the reprogramming mouse embryonic
    fibroblasts (MEFs), maintained in either 2i or serum conditions, at 10 time points across 16 days.
    Overall, 68,339 cells were profiled to an average depth of 38,462 reads per cell. After discarding
    cells with less than 1,000 genes detected, 65,781 cells were obtained, with a median of 2,398 genes
    and 7,387 unique transcripts per cell.


    Source paper:
        https://doi.org/10.1016/j.cell.2019.01.006

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "schiebinger_2019_16day.h5ad"
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


def schiebinger_2019_18day() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Samples were collected from established iPSC lines reprogrammed from the reprogramming mouse embryonic
    fibroblasts (MEFs), maintained in either 2i or serum conditions, over 39 time points separated by
    ∼12 hours across an 18-day time course (and every 6 hours between days 8 and 9). Overall, 259,155 cells
    were profiled.

    Source paper:
        https://doi.org/10.1016/j.cell.2019.01.006

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "Schiebinger_2019_18day.h5ad"
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


def schraivogel_2020_tap_screen_chr11() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    TAP-seq was applied to generate perturbation-based enhancer–target gene maps in K562 cells.
    They perturbed all 1,778 putatively active enhancers predicted on the basis of ENCODE data
    in two regions on chromosome 8 and 11, and identified effects on expressed protein-coding genes
    within the same regions. Thus, in each cell, 68 (chromosome 8) or 79 (chromosome 11) target genes
    were measured.

    Source paper:
        https://doi.org/10.1038/s41592-020-0837-5

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "schraivogel_2020_tap_screen_chr11.h5ad"
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


def schraivogel_2020_tap_screen_chr8() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    TAP-seq was applied to generate perturbation-based enhancer–target gene maps in K562 cells.
    They perturbed all 1,778 putatively active enhancers predicted on the basis of ENCODE data
    in two regions on chromosome 8 and 11, and identified effects on expressed protein-coding genes
    within the same regions. Thus, in each cell, 68 (chromosome 8) or 79 (chromosome 11) target genes
    were measured.

    Source paper:
        https://doi.org/10.1038/s41592-020-0837-5

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "schraivogel_2020_tap_screen_chr8.h5ad"
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


def shifrut_2018() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    The authors developed a new method, single guide RNA (sgRNA) lentiviral infection
    with Cas9 protein electroporation (SLICE), and adapted it to allow for CROP-Seq in
    primary human T cells. They used a library of 48 sgRNA, derived from GW screens,
    to explore transcriptional changes downstream of CRISPR-KO. Dataset includes CD8
    T cells from two donors, for two conditions: with TCR stimulation or No stimulation.

    Source paper:
        https://doi.org/10.1016/j.cell.2018.10.024

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "shifrut_2018.h5ad"
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


def srivatsan_2020_sciplex2() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    A549, a human lung adenocarcinoma cell line, was exposed to one of four compounds:
    dexamethasone (a corticosteroid agonist), nutlin-3a (a p53-Mdm2 antagonist),
    BMS-345541 (an inhibitor of nuclear factor κB–dependent transcription), or vorinostat
    [suberoylanilide hydroxamic acid (SAHA), an HDAC inhibitor], for 24 hours across seven
    doses in triplicate for a total of 84 drug–dose–replicate combinations and additional
    vehicle controls. Nuclei from each well were labelled and subjected to sci-RNA-seq.

    Source paper:
        https://doi.org/10.1126/science.aax6234

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "srivatsan_2020_sciplex2.h5ad"
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


def srivatsan_2020_sciplex3() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    sci-Plex was used to screen three well-characterized human cancer cell lines, A549
    (lung adenocarcinoma), K562 (chronic myelogenous leukemia), and MCF7 (mammary
    adenocarcinoma)exposed to 188 compounds, profiling ~650,000 single-cell
    transcriptomes across ~5000 independent samples in one experiment.

    Source paper:
        https://doi.org/10.1126/science.aax6234

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "srivatsan_2020_sciplex3.h5ad"
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


def srivatsan_2020_sciplex4() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    A549 and MCF7 cells were treated with pracinostat in the presence and absence of
    acetyl-CoA precursors (acetate, pyruvate, or citrate) or inhibitors of enzymes
    (ACLY, ACSS2, or PDH) involved in replenishing acetyl-CoA pools. After treatment,
    cells were harvested and processed using sci-Plex and trajectories constructed
    for each cell line. In both A549 and MCF7 cells, acetate, pyruvate, and citrate
    supplementation was capable of blocking pracinostat-treated cells from reaching
    the end of the HDAC inhibitor trajectory.

    Source paper:
        https://doi.org/10.1126/science.aax6234

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "srivatsan_2020_sciplex4.h5ad"
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


def tian_2019_day7neuron() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    The authors performed single-cell RNA sequencing of ∼20,000 day 7 neurons via 10x Genomics
    platform. Transcripts containing sgRNA sequences were further amplified to facilitate
    sgRNA identity assignment among a pool of 58 sgRNAs (two sgRNAs targeting 27 selected
    gene and four non-targeting control sgRNAs). Following sequencing, transcriptomes and
    sgRNA identities were mapped to individual cells. High data quality for neurons was evident
    from ∼91,000 mean reads per cell, the median number of ∼4,600 genes detected per cell and
    ∼8,400 cells to which a unique sgRNA could be assigned after quality control.


    Source paper:
        https://doi.org/10.1016/j.neuron.2019.07.014

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "tian_2019_day7neuron.h5ad"
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


def tian_2019_ipsc() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    The authors performed single-cell RNA sequencing of ∼20,000 iPSCs via 10x Genomics
    platform. Transcripts containing sgRNA sequences were further amplified to facilitate
    sgRNA identity assignment among a pool of 58 sgRNAs (two sgRNAs targeting 27 selected
    gene and four non-targeting control sgRNAs). Following sequencing, transcriptomes and
    sgRNA identities were mapped to individual cells. High data quality for iPSCs was evident
    from ∼84,000 mean reads per cell, the median number of ∼5,000 genes detected per cell and
    ∼15,000 cells to which a unique sgRNA could be assigned after quality control.


    Source paper:
        https://doi.org/10.1016/j.neuron.2019.07.014

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "tian_2019_iPSC.h5ad"
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


def tian_2021_crispra() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    For CRISPRa, 100 genes were inclued in the CROP-seq experiments. The CROP-seq
    libraries included 2 sgRNAs per gene plus 6 non-targeting control sgRNAs for a
    total of 206 sgRNAs. On Day 10 after CRISPRa neurons were infected by the
    CROP-seq sgRNA library, approximately 50,000 CRISPRi neurons were loaded into
    10X chips with about 25,000 input cells per lane.

    Source paper:
        https://doi.org/10.1038/s41593-021-00862-0

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "tian_2021_crispra.h5ad"
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


def tian_2021_crispri() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    For CRISPRi, 184 genes were inclued in the CROP-seq experiments. The CROP-seq
    libraries included 2 sgRNAs per gene plus 6 non-targeting control sgRNAs for a
    total of 374 sgRNAs. On Day 10 after CRISPRi neurons were infected by the
    CROP-seq sgRNA library, approximately 98,000 CRISPRi neurons were loaded into
    10X chips with about 25,000 input cells per lane.


    Source paper:
        https://doi.org/10.1038/s41593-021-00862-0

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "tian_2021_crispri.h5ad"
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


def weinreb_2020() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Source paper:
        https://doi.org/10.1126/science.aaw3381

    The authors developed a tool called LARRY (lineage and RNA recovery) and applied
    it to mouse embryonic stem cells under different cytokine conditions across time.

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "weinreb_2020.h5ad"
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


def xie_2017() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Mosaic-seq was applied to 71 constituent enhancers from 15 super-enhancers, this
    analysis of 51,448 sgRNA-induced transcriptomes finds that only a small number of
    constituents are major effectors of target gene expression。

    Source paper:
        https://doi.org/10.1016/j.molcel.2017.03.007

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "xie_2017.h5ad"
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


def zhao_2021() -> AnnData:  # pragma: no cover
    """scPerturb Single-Cell Perturbation Data.

    https://github.com/sanderlab/scPerturb

    Reference:
       Stefan Peidli, Tessa Durakis Green, Ciyue Shen, Torsten Gross, Joseph Min,
       Jake Taylor-King, Debora Marks, Augustin Luna, Nils Bluthgen, Chris Sander.
       scPerturb: Information Resource for Harmonized Single-Cell Perturbation Data.
       bioRxiv 2022.08.20.504663; doi: 10.1101/2022.08.20.504663.

    Source paper:
        https://doi.org/10.1186/s13073-021-00894-y

    This study combines multiplexed drug perturbation in acute slice culture from freshly
    resected tumors with scRNA-seq to profile transcriptome-wide drug responses in
    individual patients. They applied this approach to drug perturbations on slices derived
    from six glioblastoma (GBM) resections to identify conserved drug responses and to one
    additional GBM resection to identify patient-specific responses.

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "zhaoSims2021.h5ad"
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
