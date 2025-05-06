from pathlib import Path

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

    References:
        Burczynski et al., "Molecular classification of Crohn's disease and
        ulcerative colitis patients using transcriptional profiles in peripheral blood mononuclear cells"
        J Mol Diagn 8, 51 (2006). PMID:16436634.

    Returns:
        :class:`~anndata.AnnData` object of the Burczynski dataset
    """
    return sc.datasets.burczynski06()


def papalexi_2021() -> MuData:  # pragma: no cover
    """ECCITE-seq dataset of 11 gRNAs generated from stimulated THP-1 cell line.

    References:
        Papalexi, E., Mimitou, E.P., Butler, A.W. et al. Characterizing the molecular regulation
        of inhibitory immune checkpoints with multimodal single-cell screens.
        Nat Genet 53, 322–331 (2021). https://doi.org/10.1038/s41588-021-00778-2

    Returns:
        :class:`~mudata.MuData` object of the ECCITE-seq dataset
    """
    import mudata as md

    output_file_name = "papalexi_2021.h5mu"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/36509460",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    mdata = md.read_h5mu(output_file_path)
    mdata.pull_obs()
    mdata.pull_var()

    return mdata


def sc_sim_augur() -> AnnData:  # pragma: no cover
    """Simulated test dataset used in Augur example vignettes.

    References:
        Obtained from https://github.com/neurorestore/Augur

    Returns:
        :class:`~anndata.AnnData` object of a simulated single-cell RNA seq dataset
    """
    output_file_name = "sc_sim_augur.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/49828902",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def bhattacherjee() -> AnnData:  # pragma: no cover
    """Processed single-cell data PFC adult mice under cocaine self-administration.

    Adult mice were subject to cocaine self-administration, samples were
    collected a three time points: Maintance, 48h after cocaine withdrawal and
    15 days after cocaine withdrawal.

    References:
        Bhattacherjee A, Djekidel MN, Chen R, Chen W, Tuesta LM, Zhang Y. Cell
        type-specific transcriptional programs in mouse prefrontal cortex during
        adolescence and addiction. Nat Commun. 2019 Sep 13;10(1):4169.
        doi: 10.1038/s41467-019-12054-3. PMID: 31519873; PMCID: PMC6744514.

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "bhattacherjee_rna.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34526528",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def sciplex3_raw() -> AnnData:  # pragma: no cover
    """Raw sciplex3 perturbation dataset curated for perturbation modeling.

    References:
        Srivatsan SR et al., Trapnell C. Massively multiplex chemical transcriptomics at
        single-cell resolution. Science. 2020 Jan 3;367(6473):45-51.
        doi: 10.1126/science.aax6234. Epub 2019 Dec 5. PMID: 31806696; PMCID: PMC7289078.

    Returns:
        :class:`~anndata.AnnData` object of a single-cell RNA seq dataset
    """
    output_file_name = "sciplex3.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/33979517",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def tasccoda_example() -> AnnData:  # pragma: no cover
    """Example for the coda part of a mudata object.

    Resulting AnnData object (mudata['coda']) when preparing a dataset for processing with tascCODA.
    Created using the smillie dataset, which comprises scRNA-seq data of the small intestine of mice under Ulcerative Colitis.
    The full dataset containing the actual count data can be obtained via smillie_2019().

    References:
        Smillie, Christopher S et al. “Intra- and Inter-cellular Rewiring of the Human Colon during Ulcerative Colitis.”
        Cell vol. 178,3 (2019): 714-730.e22. doi:10.1016/j.cell.2019.06.029

    Returns:
        :class:`~anndata.AnnData` object of the dataset.
    """
    output_file_name = "tasccoda_smillie.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/38648585",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def frangieh_2021() -> AnnData:  # pragma: no cover
    """Processed perturb-CITE-seq data with multi-modal RNA and protein single-cell profiling.

    We profiled RNA and 20 surface proteins in over 218,000 cells under ~750 perturbations,
    chosen by their membership in an immune evasion program that is associated with
    immunotherapy resistance in patients.

    References:
        Frangieh, C.J., Melms, J.C., Thakore, P.I. et al. Multimodal pooled Perturb-CITE-seq
        screens in patient models define mechanisms of cancer immune evasion.
        Nat Genet 53, 332–341 (2021). https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of the Perturb-CITE-seq data.
    """
    output_file_name = "frangieh_2021.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34013717",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def frangieh_2021_raw() -> AnnData:  # pragma: no cover
    """Raw Perturb-CITE-seq data with multi-modal RNA and protein single-cell profiling readout.

    We profiled RNA and 20 surface proteins in over 218,000 cells under ~750 perturbations,
    chosen by their membership in an immune evasion program that is associated with
    immunotherapy resistance in patients.

    References:
        Frangieh, C.J., Melms, J.C., Thakore, P.I. et al. Multimodal pooled Perturb-CITE-seq
        screens in patient models define mechanisms of cancer immune evasion.
        Nat Genet 53, 332–341 (2021). https://doi.org/10.1038/s41588-021-00779-1

    Returns:
        :class:`~anndata.AnnData` object of raw Perturb-CITE-seq data.
    """
    output_file_name = "frangieh_2021_raw.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34012565",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def dixit_2016_raw() -> AnnData:  # pragma: no cover
    """Perturb-seq: scRNA-seq with pooled CRISPR-KO perturbations.

    scRNA-seq with pooled CRISPR-KO perturbations in 200,000 cells across six screens
    unstimulated BMDC, BMDC stimulated at 3hr, TFs in K562 at 7 and 13 days post transduction,
    and 13 days at a higher MOI of perturbations.

    References:
        Dixit A, Parnas O, Li B, Chen J et al. Perturb-Seq: Dissecting Molecular Circuits with
        Scalable Single-Cell RNA Profiling of Pooled Genetic Screens.
        Cell 2016 Dec 15;167(7):1853-1866.e17. DOI:https://doi.org/10.1016/j.cell.2016.11.038

    Returns:
        :class:`~anndata.AnnData` object of raw Perturb-seq data.
    """
    output_file_name = "dixit_2016_raw.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34012565",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def dixit_2016() -> AnnData:  # pragma: no cover
    """Perturb-seq: scRNA-seq with pooled CRISPR-KO perturbations.

    scRNA-seq with pooled CRISPR-KO perturbations in 200,000 cells across six screens
    unstimulated BMDC, BMDC stimulated at 3hr, TFs in K562 at 7 and 13 days post transduction,
    and 13 days at a higher MOI of perturbations.

    References:
        Dixit A, Parnas O, Li B, Chen J et al. Perturb-Seq: Dissecting Molecular Circuits with
        Scalable Single-Cell RNA Profiling of Pooled Genetic Screens.
        Cell 2016 Dec 15;167(7):1853-1866.e17. DOI:https://doi.org/10.1016/j.cell.2016.11.038

    Returns:
        :class:`~anndata.AnnData` object of Perturb-seq data
    """
    output_file_name = "dixit_2016.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34014608",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def norman_2019() -> AnnData:  # pragma: no cover
    """Processed single-cell, pooled CRISPR screening.

    Single-cell, pooled CRISPR screening experiment comparing the transcriptional effects of
    overexpressing genes alone or in combination

    References:
        Norman, Thomas M et al. “Exploring genetic interaction manifolds constructed from rich
        single-cell phenotypes.” Science (New York, N.Y.) vol. 365,6455 (2019): 786-793.
        doi:10.1126/science.aax4438

    Returns:
        :class:`~anndata.AnnData` object of single-cell pooled CRISPR screening.
    """
    output_file_name = "norman_2019.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34027562",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def norman_2019_raw() -> AnnData:  # pragma: no cover
    """Raw single-cell, pooled CRISPR screening.

    Single-cell, pooled CRISPR screening experiment comparing the transcriptional effects of
    overexpressing genes alone or in combination

    References:
        Norman, Thomas M et al. “Exploring genetic interaction manifolds constructed from rich
        single-cell phenotypes.” Science (New York, N.Y.) vol. 365,6455 (2019): 786-793.
        doi:10.1126/science.aax4438

    Returns:
        :class:`~anndata.AnnData` object of raw single-cell pooled CRISPR screening
    """
    output_file_name = "norman_2019_raw.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34002548",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def dialogue_example() -> AnnData:  # pragma: no cover
    """Example dataset used in DIALOGUE vignettes.

    References:
        https://github.com/livnatje/DIALOGUE/wiki/Example

    Returns:
        :class:`~anndata.AnnData` object
    """
    output_file_name = "dialogue_example.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/43462662",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def distance_example() -> AnnData:  # pragma: no cover
    """Example dataset used to feature distances and distance_tests.

    Processed and subsetted version of original Perturb-seq dataset by Dixit et al.

    Returns:
        :class:`~anndata.AnnData` object
    """
    output_file_name = "distances_example_data.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/39561379",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def kang_2018() -> AnnData:  # pragma: no cover
    """Processed multiplexing droplet-based single cell RNA-sequencing using genetic barcodes.

    HiSeq 2500 data for sequencing of PBMCs from SLE patients and 2 controls. We collected 1M cells
    each from frozen PBMC samples that were Ficoll isolated and prepared using the 10x Single Cell
    instrument according to standard protocol. Samples A, B, and C were prepared on the instrument
    directly following thaw, while samples 2.1 and 2.2 were cultured for 6 hours with (B) or
    without (A) IFN-beta stimulation prior to loading on the 10x Single Cell instrument.

    References:
        Kang, H., Subramaniam, M., Targ, S. et al. Multiplexed droplet single-cell RNA-sequencing
        using natural genetic variation.
        Nat Biotechnol 36, 89–94 (2018). https://doi.org/10.1038/nbt.4042

    Returns:
        :class:`~anndata.AnnData` object of droplet-based single cell RNA-sequencing
    """
    output_file_name = "kang_2018.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/34464122",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def stephenson_2021_subsampled() -> AnnData:  # pragma: no cover
    """Processed 10X 5' scRNA-seq data from PBMC of COVID-19 patients and healthy donors.

    The study profiled peripheral blood mononuclear cells from 90 COVID-19 patients with different disease severity and 23 healthy control donors.
    Here the dataset was downsampled to approx. 500 cells per donor and cells were mapped to a reference atlas of healthy PBMCs from 12 studies
    with scArches.

    References:
        Stephenson, E., Reynolds, G., Botting, R. A., et al. (2021).
        Single-cell multi-omics analysis of the immune response in COVID-19.
        Nature Medicine, 27(5). https://doi.org/10.1038/s41591-021-01329-2

    Returns:
        :class:`~anndata.AnnData` object of scRNA-seq profiles
    """
    output_file_name = "stephenson_2021_subsampled.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/38171703",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def haber_2017_regions() -> AnnData:  # pragma: no cover
    """Raw single-cell, pooled CRISPR screening.

    Epithelial cells from the small intestine and organoids of mice.
    Some of the cells were also subject to Salmonella or Heligmosomoides polygyrus infection (day 3 and day 10).

    References:
        Haber, Adam L. et al. “A single-cell survey of the small intestinal epithelium” Nature vol. 551 (2017): 333-339
        doi:10.1038/nature24489

    Returns:
        :class:`~anndata.AnnData` object
    """
    output_file_name = "haber_2017_regions.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/54169301",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def adamson_2016_pilot() -> AnnData:  # pragma: no cover
    """6000 chronic myeloid leukemia (K562) cells carrying 8 distinct GBCs.

    In a pilot experiment, single-cell RNA-seq was performed on a pool of individually transduced chronic
    myeloid leukemia cells (K562) carrying 8 distinct guide barcodes, analyzing ∼6,000 cells total.

    References:
        Publication: https://www.sciencedirect.com/science/article/pii/S0092867416316609?via%3Dihub \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "adamson_2016_pilot.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/AdamsonWeissman2016_GSM2406675_10X001.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def adamson_2016_upr_epistasis() -> AnnData:  # pragma: no cover
    """15000 K562 cells with UPR sensor genes knocked out and treated with thapsigargin.

    In UPR epistasis experiment, Perturb-seq was applied to explore the branches of
    the mammalian UPR. Using the three-guide Perturb-seq vector, sgRNAs targeting each
    UPR sensor gene were introduced into K562 cells with dCas9-KRAB. Transduced cells
    were then pooled, sorted for vector delivery, and after 5 days of total growth,
    treated with thapsigargin. Control cells were treated with DMSO. Transcriptomes
    of ∼15,000 cells were sequenced.

    References:
        Publication: https://www.sciencedirect.com/science/article/pii/S0092867416316609?via%3Dihub \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb preparedsingle-cell perturbation data
    """
    output_file_name = "adamson_2016_upr_epistasis.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/AdamsonWeissman2016_GSM2406677_10X005.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def adamson_2016_upr_perturb_seq() -> AnnData:  # pragma: no cover
    """Transcriptomics measurements of 65000 cells that were subject to 91 sgRNAs targeting 82 genes.

    In UPR Perturb-seq experiment, Perturb-seq was applied to a small CRISPRi library
    of 91 sgRNAs targeting 82 genes. sgRNAs were delivered via pooled transduction
    using a mixture of separately prepared lentiviruses, and ∼65,000 transcriptomes
    were collected in one large pooled experiment.

    References:
        Publication: https://www.sciencedirect.com/science/article/pii/S0092867416316609?via%3Dihub \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "adamson_2016_upr_perturb_seq.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def aissa_2021() -> AnnData:  # pragma: no cover
    """Transcriptomics of 848 P99 cells subject to consecutive erlotinib and 756 control cells.

    In this study 848 PC9 cells subjected to consecutive erlotinib treatment (for
    1, 2, 4, 9, and 11 days) and 756 control cells were analysed using Drop-seq.

    References:
        Publication: https://www.nature.com/articles/s41467-021-21884-z \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "aissa_2021.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/AissaBenevolenskaya2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def chang_2021() -> AnnData:  # pragma: no cover
    """Transcriptomics of 5 different cell lines that were induced with a unique TraCe-seq barcode.

    TraCe-seq is a method that captures at clonal resolution the origin, fate and
    differential early adaptive transcriptional programs of cells in a complex
    population in response to distinct treatments. Here, a unique TraCe-seq barcode was
    transduced into five different cell lines (PC9, MCF-10A, MDA-MB-231, NCI-H358 and NCI-H1373).
    Transduced cells were selected with puromycin only, mixed together and profiled by 10x scRNA-seq.

    References:
        https://www.nature.com/articles/s41587-021-01005-3

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "chang_2021.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/ChangYe2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def datlinger_2017() -> AnnData:  # pragma: no cover
    """Transcriptomics measurements of 5905 Jurkat cells induced with anti-CD3 and anti-CD28 antibodies.

    For CROP-seq, Jurkat cells were transduced with a gRNA library targeting high-level
    regulators of T cell receptor signaling and a set of transcription factors. After 10
    days of antibiotic selection and expansion, cells were stimulated with anti-CD3 and
    anti-CD28 antibodies or left untreated. Both conditions were analyzed using CROP-seq,
    measuring TCR activation for each gene knockout. The dataset comprises 5,905 high-quality
    single-cell transcriptomes with uniquely assigned gRNAs.

    References:
        Publication: https://www.nature.com/articles/nmeth.4177 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "datlinger_2017.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/DatlingerBock2017.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def datlinger_2021() -> AnnData:  # pragma: no cover
    """Transcriptomics measurements of 151788 nuclei of four cell lines.

    A large-scale scifi-RNA-seq experiment was performed with 383,000 nuclei loaded into
    a single microfluidic channel of the Chromium system. Four human cell lines (HEK293T,
    Jurkat, K562 and NALM-6) were combined in an equal mixture, and technical replicates of
    each cell line with different preindexing (round1) barcodes were marked. This experiment
    resulted in 151,788 single-cell transcriptomes passing quality control.

    Source paper:
        Publication: https://doi.org/10.1038/s41592-021-01153-z \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
         :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "datlinger_2021.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/DatlingerBock2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def frangieh_2021_protein() -> AnnData:  # pragma: no cover
    """CITE-seq data of 218000 cells under 750 perturbations (only the surface protein data).

    Perturb-CITE-seq was developed for pooled CRISPR perturbation screens with multi-modal
    RNA and protein single-cell profiling readout and applied to screen patient-derived
    autologous melanoma and tumor infiltrating lymphocyte (TIL) co-cultures. RNA and 20
    surface proteins were profiled in over 218,000 cells under ~750 perturbations, chosen
    by their membership in an immune evasion program that is associated with immunotherapy
    resistance in patients.

    References:
        Publication: https://doi.org/10.1038/s41588-021-00779-1 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "frangieh_2021_protein.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/FrangiehIzar2021_protein.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def frangieh_2021_rna() -> AnnData:  # pragma: no cover
    """CITE-seq data of 218000 cells under 750 perturbations (only the transcriptomics data).

    Perturb-CITE-seq was developed for pooled CRISPR perturbation screens with multi-modal
    RNA and protein single-cell profiling readout and applied to screen patient-derived
    autologous melanoma and tumor infiltrating lymphocyte (TIL) co-cultures. RNA and 20
    surface proteins were profiled in over 218,000 cells under ~750 perturbations, chosen
    by their membership in an immune evasion program that is associated with immunotherapy
    resistance in patients.

    Source paper:
        Publication: https://doi.org/10.1038/s41588-021-00779-1 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "frangieh_2021_rna.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/FrangiehIzar2021_RNA.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def gasperini_2019_atscale() -> AnnData:  # pragma: no cover
    """Transcriptomics of 254974 cells of chronic K562 cells with CRISPRi perturbations.

    Across two experiments, the authors used dCas9-KRAB to perturb 5,920 candidate enhancers
    with no strong a priori hypothesis as to their target gene(s) in 254,974 cells of chronic
    myelogenous leukemia cell line K562, with CRISPRi as the mode of perturbation.

    References:
        Publication: https://doi.org/10.1016/j.cell.2018.11.029 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "gasperini_2019_atscale.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/GasperiniShendure2019_atscale.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def gasperini_2019_highmoi() -> AnnData:  # pragma: no cover
    """K562 perturbed cells with 1119 candidate enhancers (only the high MOI part).

    The authors used dCas9-KRAB to perturb 1,119 candidate enhancers with no strong a priori
    hypothesis as to their target gene(s) in the chronic myelogenous leukemia cell line
    K562, with CRISPRi as the mode of perturbation, where K562 cells were transduced
    at a high MOI (pilot library MOI = ∼15).

    References:
        Publication: https://doi.org/10.1016/j.cell.2018.11.029 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "gasperini_2019_highmoi.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/GasperiniShendure2019_highMOI.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def gasperini_2019_lowmoi() -> AnnData:  # pragma: no cover
    """K562 perturbed cells with 1119 candidate enhancers (only the low MOI part).

    The authors used dCas9-KRAB to perturb 1,119 candidate enhancers with no strong a priori
    hypothesis as to their target gene(s) in chronic myelogenous leukemia cell line K562,
    with CRISPRi as the mode of perturbation, where K562 cells were transduced at a
    low MOI (pilot library MOI = ∼1).

    References:
        Publication: https://doi.org/10.1016/j.cell.2018.11.029 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "gasperini_2019_lowmoi.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/GasperiniShendure2019_lowMOI.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def gehring_2019() -> AnnData:  # pragma: no cover
    """96-plex perturbation experiment on live mouse neural stem cells.

    In this study, a 96-plex perturbation experiment was conducted on live mouse neural
    stem cells (NSCs), consisting of a pair of drug-triples with 4 drugs in total at 3
    or 4 different concentractions.

    References:
        Publication: https://doi.org/10.1038/s41587-019-0372-z \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of a scPerturb prepared single-cell dataset
    """
    output_file_name = "gehring_2019.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/GehringPachter2019.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def mcfarland_2020() -> AnnData:  # pragma: no cover
    """Response of various cell lines to a range of different drugs and CRISPRi perturbations.

    Here, the authors developed MIX-Seq, a method for multiplexed transcriptional profiling
    of post-perturbation responses across many cell contexts, using scRNA-seq applied to
    co-treated pools of cancer cell lines. The responses of pools of 24–99 cell lines to
    a range of different drugs were profiled, as well as to CRISPR perturbations.

    References:
        Publication: https://doi.org/10.1038/s41467-020-17440-w \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "mcfarland_2020.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/McFarlandTsherniak2020.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def replogle_2022_k562_essential() -> AnnData:  # pragma: no cover
    """K562 cells transduced with CRISPRi (day 7 after transduction).

    For day 6 essential-scale experiment in chronic myeloid leukemia (CML) (K562) cell
    lines, library lentivirus was packaged into lentivirus in 293T cells and empirically
    measured in K562 cells to obtain viral titers. CRISPRi K562 cells were transduced
    and 20Q1 Cancer Dependency Map common essential genes were targeted at day 7 after transduction.

    References:
        Publication: https://doi.org/10.1016/j.cell.2022.05.013 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "replogle_2022_k562_essential.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/ReplogleWeissman2022_K562_essential.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def replogle_2022_k562_gwps() -> AnnData:  # pragma: no cover
    """K562 cells transduced with CRISPRi (day 8 after transduction).

    Here, the authors used a compact, multiplexed CRISPR interference (CRISPRi) library
    to assay thousands of loss-of-function genetic perturbations with single-cell RNA sequencing
    in chronic myeloid leukemia (CML) (K562) cell lines. For the K562 day 8 genome-scale
    Perturb-seq experiment, library lentivirus was packaged into lentivirus in 293T cells and
    empirically measured in K562 cells to obtain viral titers. CRISPRi K562 cells were transduced
    and all expressed genes were targeted at day 8 after transduction

    References:
        Publication: https://doi.org/10.1016/j.cell.2022.05.013 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "replogle_2022_k562_gwps.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/ReplogleWeissman2022_K562_gwps.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def replogle_2022_rpe1() -> AnnData:  # pragma: no cover
    """RPE1 cells transduced with CRISPRi (day 7 after transduction).

    For day 7 essential-scale Perturb-seq experiment in retinal pigment epithelial (RPE1)
    cell lines, library lentivirus was packaged into lentivirus in 293T cells and
    empirically measured in RPE1 cells to obtain viral titers. CRISPRi RPE1 cells expressing
    ZIM3 KRAB-dCas9-P2A-BFP were transduced. 20Q1 Cancer Dependency Map common essential
    genes were targeted at day 7 after transduction.

    References:
        Publication: https://doi.org/10.1016/j.cell.2022.05.013 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "replogle_2022_rpe1.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/ReplogleWeissman2022_rpe1.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def schiebinger_2019_16day() -> AnnData:  # pragma: no cover
    """Transcriptomes of 65781 iPSC cells collected over 10 time points in 2i or serum conditions (16-day time course).

    Samples were collected from established iPSC lines reprogrammed from the reprogramming mouse embryonic
    fibroblasts (MEFs), maintained in either 2i or serum conditions, at 10 time points across 16 days.
    Overall, 68,339 cells were profiled to an average depth of 38,462 reads per cell. After discarding
    cells with less than 1,000 genes detected, 65,781 cells were obtained, with a median of 2,398 genes
    and 7,387 unique transcripts per cell.


    References:
        Publication: https://doi.org/10.1016/j.cell.2019.01.006 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "schiebinger_2019_16day.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/SchiebingerLander2019_GSE106340.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def schiebinger_2019_18day() -> AnnData:  # pragma: no cover
    """Transcriptomes of 259155 iPSC cells collected over 39 time points in 2i or serum conditions (18-day time course).

    Samples were collected from established iPSC lines reprogrammed from the reprogramming mouse embryonic
    fibroblasts (MEFs), maintained in either 2i or serum conditions, over 39 time points separated by
    ∼12 hours across an 18-day time course (and every 6 hours between days 8 and 9). Overall, 259,155 cells
    were profiled.

    References:
        Publication: https://doi.org/10.1016/j.cell.2019.01.006 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "Schiebinger_2019_18day.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/SchiebingerLander2019_GSE115943.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def schraivogel_2020_tap_screen_chr11() -> AnnData:  # pragma: no cover
    """TAP-seq applied to K562 cells (only chromosome 11).

    TAP-seq was applied to generate perturbation-based enhancer–target gene maps in K562 cells.
    They perturbed all 1,778 putatively active enhancers predicted on the basis of ENCODE data
    in two regions on chromosome 8 and 11, and identified effects on expressed protein-coding genes
    within the same regions. Thus, in each cell, 68 (chromosome 8) or 79 (chromosome 11) target genes were measured.

    References:
        Publication: https://doi.org/10.1038/s41592-020-0837-5 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "schraivogel_2020_tap_screen_chr11.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_11_screen.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def schraivogel_2020_tap_screen_chr8() -> AnnData:  # pragma: no cover
    """TAP-seq applied to K562 cells (only chromosome 8).

    TAP-seq was applied to generate perturbation-based enhancer–target gene maps in K562 cells.
    They perturbed all 1,778 putatively active enhancers predicted on the basis of ENCODE data
    in two regions on chromosome 8 and 11, and identified effects on expressed protein-coding genes
    within the same regions. Thus, in each cell, 68 (chromosome 8) or 79 (chromosome 11) target genes were measured.

    References:
        Publication: https://doi.org/10.1038/s41592-020-0837-5 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "schraivogel_2020_tap_screen_chr8.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/SchraivogelSteinmetz2020_TAP_SCREEN__chromosome_8_screen.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def shifrut_2018() -> AnnData:  # pragma: no cover
    """CD8 T-cells from two donors for two conditions (SLICE and CROP-seq).

    The authors developed a new method, single guide RNA (sgRNA) lentiviral infection
    with Cas9 protein electroporation (SLICE), and adapted it to allow for CROP-Seq in
    primary human T cells. They used a library of 48 sgRNA, derived from GW screens,
    to explore transcriptional changes downstream of CRISPR-KO. Dataset includes CD8
    T cells from two donors, for two conditions: with TCR stimulation or No stimulation.

    References:
        Publication: https://doi.org/10.1016/j.cell.2018.10.024 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "shifrut_2018.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/13350497/files/ShifrutMarson2018.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def srivatsan_2020_sciplex2() -> AnnData:  # pragma: no cover
    """A549 cells exposed to four compounds.

    A549, a human lung adenocarcinoma cell line, was exposed to one of four compounds:
    dexamethasone (a corticosteroid agonist), nutlin-3a (a p53-Mdm2 antagonist),
    BMS-345541 (an inhibitor of nuclear factor κB–dependent transcription), or vorinostat
    [suberoylanilide hydroxamic acid (SAHA), an HDAC inhibitor], for 24 hours across seven
    doses in triplicate for a total of 84 drug–dose–replicate combinations and additional
    vehicle controls. Nuclei from each well were labelled and subjected to sci-RNA-seq.

    References:
        Publication: https://doi.org/10.1126/science.aax6234 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "srivatsan_2020_sciplex2.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/record/10044268/files/SrivatsanTrapnell2020_sciplex2.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def srivatsan_2020_sciplex3() -> AnnData:  # pragma: no cover
    """Transcriptomes of 650000 A549, K562, and mCF7 cells exposed to 188 compounds.

    sci-Plex was used to screen three well-characterized human cancer cell lines, A549
    (lung adenocarcinoma), K562 (chronic myelogenous leukemia), and MCF7 (mammary
    adenocarcinoma)exposed to 188 compounds, profiling ~650,000 single-cell
    transcriptomes across ~5000 independent samples in one experiment.

    References:
        Publication: https://doi.org/10.1126/science.aax6234 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "srivatsan_2020_sciplex3.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def srivatsan_2020_sciplex4() -> AnnData:  # pragma: no cover
    """A549 and MCF7 cells treated with pracinostat.

    A549 and MCF7 cells were treated with pracinostat in the presence and absence of
    acetyl-CoA precursors (acetate, pyruvate, or citrate) or inhibitors of enzymes
    (ACLY, ACSS2, or PDH) involved in replenishing acetyl-CoA pools. After treatment,
    cells were harvested and processed using sci-Plex and trajectories constructed
    for each cell line. In both A549 and MCF7 cells, acetate, pyruvate, and citrate
    supplementation was capable of blocking pracinostat-treated cells from reaching
    the end of the HDAC inhibitor trajectory.

    References:
        Publication: https://doi.org/10.1126/science.aax6234 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "srivatsan_2020_sciplex4.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/SrivatsanTrapnell2020_sciplex4.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def tian_2019_day7neuron() -> AnnData:  # pragma: no cover
    """Transcriptomes of 20000 day 7 neurons targeted by 58 gRNAs.

    The authors performed single-cell RNA sequencing of ∼20,000 day 7 neurons via 10x Genomics
    platform. Transcripts containing sgRNA sequences were further amplified to facilitate
    sgRNA identity assignment among a pool of 58 sgRNAs (two sgRNAs targeting 27 selected
    gene and four non-targeting control sgRNAs). Following sequencing, transcriptomes and
    sgRNA identities were mapped to individual cells. High data quality for neurons was evident
    from ∼91,000 mean reads per cell, the median number of ∼4,600 genes detected per cell and
    ∼8,400 cells to which a unique sgRNA could be assigned after quality control.

    References:
        Publication: https://doi.org/10.1016/j.neuron.2019.07.014 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "tian_2019_day7neuron.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/TianKampmann2019_day7neuron.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def tian_2019_ipsc() -> AnnData:  # pragma: no cover
    """Transcriptomics of 20000 iPSCs targeted by 58 sgRNAs.

    The authors performed single-cell RNA sequencing of ∼20,000 iPSCs via 10x Genomics
    platform. Transcripts containing sgRNA sequences were further amplified to facilitate
    sgRNA identity assignment among a pool of 58 sgRNAs (two sgRNAs targeting 27 selected
    gene and four non-targeting control sgRNAs). Following sequencing, transcriptomes and
    sgRNA identities were mapped to individual cells. High data quality for iPSCs was evident
    from ∼84,000 mean reads per cell, the median number of ∼5,000 genes detected per cell and
    ∼15,000 cells to which a unique sgRNA could be assigned after quality control.

    References:
        Publication: https://doi.org/10.1016/j.neuron.2019.07.014 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "tian_2019_iPSC.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/TianKampmann2019_iPSC.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def tian_2021_crispra() -> AnnData:  # pragma: no cover
    """CROP-seq of 50000 neurons treated with 374 gRNAs (CRISPRa only).

    For CRISPRa, 100 genes were inclued in the CROP-seq experiments. The CROP-seq
    libraries included 2 sgRNAs per gene plus 6 non-targeting control sgRNAs for a
    total of 206 sgRNAs. On Day 10 after CRISPRa neurons were infected by the
    CROP-seq sgRNA library, approximately 50,000 CRISPRi neurons were loaded into
    10X chips with about 25,000 input cells per lane.

    References:
        Publication: https://doi.org/10.1038/s41593-021-00862-0 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb single-cell perturbation data
    """
    output_file_name = "tian_2021_crispra.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/TianKampmann2021_CRISPRa.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def tian_2021_crispri() -> AnnData:  # pragma: no cover
    """CROP-seq of 98000 neurons treated with 374 gRNAs (CRISPRi only).

    For CRISPRi, 184 genes were inclued in the CROP-seq experiments. The CROP-seq
    libraries included 2 sgRNAs per gene plus 6 non-targeting control sgRNAs for a
    total of 374 sgRNAs. On Day 10 after CRISPRi neurons were infected by the
    CROP-seq sgRNA library, approximately 98,000 CRISPRi neurons were loaded into
    10X chips with about 25,000 input cells per lane.

    References:
        Publication: https://doi.org/10.1038/s41593-021-00862-0 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "tian_2021_crispri.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/TianKampmann2021_CRISPRi.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def weinreb_2020() -> AnnData:  # pragma: no cover
    """Mouse embryonic stem cells under different cytokines across time.

    The authors developed a tool called LARRY (lineage and RNA recovery) and applied
    it to mouse embryonic stem cells under different cytokine conditions across time.

    References:
        Publication: https://www.science.org/doi/10.1126/science.aaw3381 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "weinreb_2020.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/WeinrebKlein2020.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def xie_2017() -> AnnData:  # pragma: no cover
    """Single-cell transcriptomics of 51448 cells generated with Mosaic-seq.

    Mosaic-seq was applied to 71 constituent enhancers from 15 super-enhancers, this
    analysis of 51,448 sgRNA-induced transcriptomes finds that only a small number of
    constituents are major effectors of target gene expression.

    References:
        Publication: https://doi.org/10.1016/j.molcel.2017.03.007 \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "xie_2017.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/XieHon2017.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def zhao_2021() -> AnnData:  # pragma: no cover
    """Multiplexed drug perturbation from freshly resected tumors.

    This study combines multiplexed drug perturbation in acute slice culture from freshly
    resected tumors with scRNA-seq to profile transcriptome-wide drug responses in
    individual patients. They applied this approach to drug perturbations on slices derived
    from six glioblastoma (GBM) resections to identify conserved drug responses and to one
    additional GBM resection to identify patient-specific responses.

    References:
        Publication: https://doi.org/10.1186/s13073-021-00894-y \
        Obtained from scperturb: http://projects.sanderlab.org/scperturb/

    Returns:
        :class:`~anndata.AnnData` object of scPerturb prepared single-cell perturbation data
    """
    output_file_name = "zhaoSims2021.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://zenodo.org/records/10044268/files/ZhaoSims2021.h5ad?download=1",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def cinemaot_example() -> AnnData:  # pragma: no cover:
    """Subsampled CINEMA-OT example dataset.

    Ex vivo stimulation of human peripheral blood mononuclear cells (PBMC) with interferon. This is a subsampled
    dataset containing 1000 cells, either without stimulation or stimulated with IFNb. The full dataset is available
    via the cinemaot_full() function.


    Returns:
        :class:`~anndata.AnnData` object of PBMCs stimulated with interferon.
    """
    output_file_name = "cinemaot_example.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/42362796?private_link=270b0d2c7f1ea57c366d",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def dong_2023() -> AnnData:  # pragma: no cover
    """Complete CINEMA-OT dataset.

    Ex vivo stimulation of human peripheral blood mononuclear cells (PBMC) with interferon. This is the full dataset
    containing 9209 cells that were stimulated with IFNb, IFNg, IFNb+IFNg, or left unstimulated. A subsampled version
    of the dataset is available via cinemaot_example().

    References:
        Preprint: https://doi.org/10.1101/2022.07.31.502173
        Dataset available here: https://datadryad.org/stash/dataset/doi:10.5061/dryad.4xgxd25g1

    Returns:
        :class:`~anndata.AnnData` object of PBMCs stimulated with interferon.
    """
    output_file_name = "dong_2023.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/43068190",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def smillie_2019() -> AnnData:  # pragma: no cover
    """scRNA-seq data of the small intestine of mice under Ulcerative Colitis.

    The resulting AnnData when preparing this dataset for processing with tascCODA is available via tasccoda_example().

    References:
        Smillie, Christopher S et al. “Intra- and Inter-cellular Rewiring of the Human Colon during Ulcerative Colitis.”
        Cell vol. 178,3 (2019): 714-730.e22. doi:10.1016/j.cell.2019.06.029

    Returns:
        :class:`~anndata.AnnData` object of the dataset.
    """
    output_file_name = "smillie_2019.h5ad.zip"
    output_file_path = settings.datasetdir / Path(output_file_name).with_suffix("")
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/43317285",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=True,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def combosciplex() -> AnnData:  # pragma: no cover
    """scRNA-seq subset of the combinatorial experiment of sciplex3.

    References:
        Srivatsan SR et al., Trapnell C. Massively multiplex chemical transcriptomics at
        single-cell resolution. Science. 2020 Jan 3;367(6473):45-51.
        doi: 10.1126/science.aax6234. Epub 2019 Dec 5. PMID: 31806696; PMCID: PMC7289078.

    Returns:
        :class:`~anndata.AnnData` object of the dataset.
    """
    output_file_name = "combosciplex.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/44229635",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def sciplex_gxe1() -> AnnData:  # pragma: no cover
    """sci-Plex-GxE profiling of A172 dCas9-KRAB (HPRT1 or MMR knockout) with 6-TG/TMZ and A172 dCas9-SunTag (HPRT1 knockout) with 6-TG.

    References:
        McFaline-Figueroa JL et al., Trapnell C. Multiplex single-cell chemical genomics reveals
        the kinase dependence of the response to targeted therapy. Cell Genomics. 2024 Volume 4, Issue 2.
        doi: 10.1016/j.xgen.2023.100487

    Returns:
        :class:`~anndata.AnnData` object of the dataset.
    """
    output_file_name = "sciPlexGxE_1_GSM7056148.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/45372454",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def zhang_2021() -> AnnData:  # pragma: no cover
    """Single-cell RNA-seq of TNBC patients' immune cells exposed to paclitaxel alone or combined with the anti-PD-L1 atezolizumab.

    This analysis, involving 22 patients, identifies immune subtypes predictive of therapeutic
    responses and underscores potential limitations of combining paclitaxel with atezolizumab in treatment protocols.

    The script that generated this specific AnnData object:
    https://github.com/tessadgreen/ThesisCode/blob/main/Chapter3/drug_response/import_zhang_data.ipynb

    This dataset does not contain the single-cell ATAC-seq data that was also measured for the paper.

    References:
        Zhang Y et al., Liu Z. Single-cell analyses reveal key immune cell subsets associated with response to PD-L1 blockade in triple-negative breast cancer.
        Cancer Cell. 2021 Volume 39, Issue 12. doi: https://doi.org/10.1016/j.ccell.2021.09.010

    Returns:
        :class:`~anndata.AnnData` object of the dataset.
    """
    output_file_name = "zhang_2021.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/46457872",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


def hagai_2018() -> AnnData:  # pragma: no cover
    """Cross-species analysis of primary dermal fibroblasts and bone marrow-derived phagocytes, stimulated with dsRNA and IFNB.

    The study explores immune response variations across humans, macaques, mice, and rats.

    Referenences:
        Hagai, T., Chen, X., Miragaia, R.J. et al. Gene expression variability across cells and species shapes innate immunity.
        Nature 563, 197–202 (2018). https://doi.org/10.1038/s41586-018-0657-2

    Returns:
        :class:`~anndata.AnnData` object of the dataset.
    """
    output_file_name = "hagai_2018.h5ad"
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists():
        _download(
            url="https://figshare.com/ndownloader/files/46978846",
            output_file_name=output_file_name,
            output_path=settings.datasetdir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata
