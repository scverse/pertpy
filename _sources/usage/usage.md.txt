# Usage

Import the pertpy API as follows:

```python
import pertpy as pt
```

You can then access the respective modules like:

```python
pt.pl.cool_fancy_plot()
```

```{eval-rst}
.. currentmodule:: pertpy
```

## Data

```{eval-rst}
.. autosummary::
    :toctree: data

    data.adamson_2016_pilot
    data.adamson_2016_upr_epistasis
    data.adamson_2016_upr_perturb_seq
    data.aissa_2021
    data.bhattacherjee
    data.burczynski_crohn
    data.chang_2021
    data.datlinger_2017
    data.datlinger_2021
    data.dialogue_example
    data.dixit_2016
    data.dixit_2016_raw
    data.dixit_2016_scperturb
    data.frangieh_2021
    data.frangieh_2021_protein
    data.frangieh_2021_raw
    data.frangieh_2021_rna
    data.gasperini_2019_atscale
    data.gasperini_2019_highmoi
    data.gasperini_2019_lowmoi
    data.gehring_2019
    data.haber_2017_regions
    data.kang_2018
    data.mcfarland_2020
    data.norman_2019
    data.norman_2019_raw
    data.papalexi_2021
    data.replogle_2022_k562_essential
    data.replogle_2022_k562_gwps
    data.replogle_2022_rpe1
    data.sc_sim_augur
    data.schiebinger_2019_16day
    data.schiebinger_2019_18day
    data.schraivogel_2020_tap_screen_chr8
    data.schraivogel_2020_tap_screen_chr11
    data.sciplex3_raw
    data.shifrut_2018
    data.smillie
    data.srivatsan_2020_sciplex2
    data.srivatsan_2020_sciplex3
    data.srivatsan_2020_sciplex4
    data.stephenson_2021_subsampled
    data.tian_2019_day7neuron
    data.tian_2019_ipsc
    data.tian_2021_crispra
    data.tian_2021_crispri
    data.weinreb_2020
    data.xie_2017
    data.zhao_2021
```

## Preprocessing

### Guide Assignment

Simple functions for:

Assigning guides based on thresholds. Each cell is assigned to the most expressed gRNA if it has at least the specified number of counts.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: preprocessing
    :nosignatures:

    preprocessing.GuideAssignment
```

Example implementation:

```python
import pertpy as pt
import scanpy as sc

mdata = pt.data.papalexi_2021()
gdo = mdata.mod['gdo']
gdo.layers['counts'] = gdo.X.copy()
sc.pp.log1p(gdo)

ga = pt.pp.GuideAssignment()
ga.assign_by_threshold(gdo, 5, layer="counts", output_layer="assigned_guides")

pt.pl.guide.heatmap(gdo, layer='assigned_guides')
```

## Tools

### Cell type prioritization

#### Augur

The Python implementation of [Augur R package](https://github.com/neurorestore/Augur)
Skinnider, M.A., Squair, J.W., Kathe, C. et al. [Cell type prioritization in single-cell data](https://doi.org/10.1038/s41587-020-0605-1). Nat Biotechnol 39, 30–34 (2021).

Augur aims to rank or prioritize cell types according to their response to experimental perturbations given high dimensional single-cell sequencing data.
The basic idea is that in the space of molecular measurements cells reacting heavily to induced perturbations are
more easily separated into perturbed and unperturbed than cell types with little or no response.
This separability is quantified by measuring how well experimental labels (eg. treatment and control) can be predicted within each cell type.
Augur trains a machine learning model predicting experimental labels for each cell type in multiple cross validation runs and
then prioritizes cell type response according to metric scores measuring the accuracy of the model.
For categorical data the area under the curve is the default metric and for numerical data the concordance correlation coefficient
is used as a proxy for how accurate the model is which in turn approximates perturbation response.

Example implementation:

```python
import pertpy as pt

adata = pt.dt.sc_sim_augur()
ag = pt.tl.Augur(estimator="random_forest_classifier")
adata = ag.load(adata)
adata, results = ag.predict(adata)

# metrics for each cell type
results['summary_metrics']
```

See [augur tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/augur.html) for a more elaborate tutorial.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.Augur
```

### Pooled CRISPR screens

#### Mixscape

A Python implementation of [Mixscape](https://satijalab.org/seurat/articles/mixscape_vignette.html)
Papalexi et al. [Characterizing the molecular regulation of inhibitory immune checkpoints with multimodal single-cell screens](https://www.nature.com/articles/s41588-021-00778-2).

Mixscape first tries to remove confounding sources of variation such as cell cycle or replicate effect by embedding the cells into a perturbation space (the perturbation signature).
Next, it determines which targeted cells were affected by the genetic perturbation (=KO) and which targeted cells were not (=NP) with the use of mixture models.
Finally, it visualizes similarities and differences across different perturbations.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.Mixscape
```

Example implementation:

```python
import pertpy as pt

mdata = pt.dt.papalexi_2021()
ms = pt.tl.Mixscape()
ms.perturbation_signature(mdata['rna'], 'perturbation', 'NT', 'replicate')
ms.mixscape(adata=mdata['rna'], control='NT', labels='gene_target', layer='X_pert')
ms.lda(adata=mdata['rna'], labels='gene_target', layer='X_pert')
pt.pl.ms.lda(adata=mdata['rna'])
```

See [mixscape tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/mixscape.html) for a more elaborate tutorial.

### Compositional analysis

#### Milo

A Python implementation of Milo for differential abundance testing on KNN graphs, to ease interoperability with scverse pipelines for single-cell analysis.
See [Differential abundance testing on single-cell data using k-nearest neighbor graphs](https://www.nature.com/articles/s41587-021-01033-z) for details on the statistical framework.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.Milo
```

See [milo tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/milo.html) for a more elaborate tutorial.

Example implementation:

```python
import pertpy as pt
import scanpy as sc

adata = pt.data.stephenson_2021_subsampled()
adata.obs['COVID_severity'] = adata.obs['Status_on_day_collection_summary'].copy()
adata.obs[['patient_id', 'COVID_severity']].drop_duplicates()
adata = adata[adata.obs['Status'] != 'LPS'].copy()

milo = pt.tl.Milo()
mdata = milo.load(adata)
sc.pp.neighbors(mdata['rna'], use_rep='X_scVI', n_neighbors=150, n_pcs=10)
milo.make_nhoods(mdata['rna'], prop=0.1)
mdata = milo.count_nhoods(mdata, sample_col="patient_id")
mdata['rna'].obs['Status'] = mdata['rna'].obs['Status'].cat.reorder_categories(['Healthy', 'Covid'])
milo.da_nhoods(mdata, design='~Status')
```

#### scCODA and tascCODA

Reimplementation of scCODA for identification of compositional changes in high-throughput sequencing count data and tascCODA for sparse, tree-aggregated modeling of high-throughput sequencing data.
See [scCODA is a Bayesian model for compositional single-cell data analysis](https://www.nature.com/articles/s41467-021-27150-6) for statistical methodology and benchmarking performance of scCODA and [tascCODA: Bayesian Tree-Aggregated Analysis of Compositional Amplicon and Single-Cell Data](https://www.frontiersin.org/articles/10.3389/fgene.2021.766405/full) for statistical methodology and benchmarking performance of tascCODA.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.Sccoda
    tools.Tasccoda
```

Example implementation:

```python
import pertpy as pt

haber_cells = pt.dt.haber_2017_regions()
sccoda_model = pt.tl.Sccoda()
sccoda_data = sccoda_model.load(haber_cells,
                                type="cell_level",
                                generate_sample_level=True,
                                cell_type_identifier="cell_label",
                                sample_identifier="batch",
                                covariate_obs=["condition"])
sccoda_data.mod["coda_salm"] = sccoda_data["coda"][sccoda_data["coda"].obs["condition"].isin(["Control", "Salmonella"])].copy()

sccoda_data = sccoda_model.prepare(sccoda_data, modality_key="coda_salm", formula="condition", reference_cell_type="Goblet")
sccoda_model.run_nuts(sccoda_data, modality_key="coda_salm")
sccoda_model.summary(sccoda_data, modality_key="coda_salm")
pt.pl.coda.effects_barplot(sccoda_data, modality_key="coda_salm", parameter="Final Parameter")
```

### Multi-cellular or gene programs

#### DIALOGUE

A **work in progress (!)** Python implementation of DIALOGUE for the discovery of multicellular programs.
See [DIALOGUE maps multicellular programs in tissue from single-cell or spatial transcriptomics data](https://www.nature.com/articles/s41587-022-01288-0) for more details on the methodology.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.Dialogue
```

See [dialogue tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/dialogue.html) for a more elaborate tutorial.

```python
import pertpy as pt
import scanpy as sc

adata = pt.dt.dialogue_example()
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)



dl = pt.tl.Dialogue(sample_id = "clinical.status",
                   celltype_key = "cell.subtypes",
                   n_counts_key = "nCount_RNA",
                   n_mpcs = 3)
adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(
    adata,
    normalize=True
)
all_results, new_mcps = dl.multilevel_modeling(ct_subs=ct_subs,
                                     mcp_scores=mcps,
                                     ws_dict=ws,
                                     confounder="gender",
                                   )
```

### Distances and Permutation Tests

General purpose functions for distances and permutation tests. Reimplements
functions from [scperturb](http://projects.sanderlab.org/scperturb/) package.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.Distance
    tools.DistanceTest
```

See [Distance tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/distances.html)
and [Permutation test tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/distance_tests.html) for a more elaborate tutorial.

```python
import pertpy as pt

adata = pt.dt.distance_example_data()

# Pairwise distances
distance = pt.tl.Distance(metric='edistance', obsm_key='X_pca')
pairwise_edistance = distance.pairwise(adata, groupby='perturbation')

# E-test (Permutation test using E-distance)
etest = pt.tl.PermutationTest(metric='edistance', obsm_key='X_pca', correction='holm-sidak')
tab = etest(adata, groupby='perturbation', contrast='control')
```

### MetaData

MetaData provides tooling to fetch and add more metadata to perturbations by querying a couple of databases.
We are currently implementing several sources with more to come.

CellLineMetaData aims to retrieve various types of information related to cell lines, including cell line annotation,
bulk RNA and protein expression data.

Available databases for cell line metadata:

-   The Cancer Dependency Map Project at Broad
-   The Cancer Dependency Map Project at Sanger

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.CellLineMetaData
```

### Response prediction

#### scGen

Reimplementation of scGen for perturbation response prediction of scRNA-seq data in Jax.
See [scGen predicts single-cell perturbation responses](https://www.nature.com/articles/s41592-019-0494-8) for more details.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.SCGEN
```

Example implementation:

```python
import pertpy as pt

train = pt.dt.kang_2018()

train_new = train[~((train.obs["cell_type"] == "CD4T") &
                    (train.obs["condition"] == "stimulated"))]
train_new = train_new.copy()

pt.tl.SCGEN.setup_anndata(train_new, batch_key="condition", labels_key="cell_type")
model = pt.tl.SCGEN(train_new)
model.train(
    max_epochs=100,
    batch_size=32
)

pred, delta = model.predict(
    ctrl_key='control',
    stim_key='stimulated',
    celltype_to_predict='CD4T'
)
pred.obs['condition'] = 'pred'
```

See [augur tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/scgen_perturbation_prediction.html) for a more elaborate tutorial.

### Representation

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.kernel_pca
```

## Plots

### Preprocessing

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.guide

```

### Cell type prioritization

#### Augur

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.ag.dp_scatter
    plot.ag.important_features
    plot.ag.lollipop
    plot.ag.scatterplot

```

### Pooled CRISPR screens

#### Mixscape

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.ms.violin
    plot.ms.perturbscore
    plot.ms.heatmap
    plot.ms.barplot
    plot.ms.lda
```

### Compositional analysis

#### Milo

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.milo.nhood_graph
    plot.milo.nhood
    plot.milo.da_beeswarm
    plot.milo.nhood_counts_by_cond
```

#### scCODA and tascCODA

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.coda.stacked_barplot
    plot.coda.effects_barplot
    plot.coda.boxplots
    plot.coda.rel_abundance_dispersion_plot
    plot.coda.draw_tree
    plot.coda.draw_effects
    plot.coda.effects_umap
```

### Response prediction

#### scGen

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.scg.reg_mean_plot
    plot.scg.reg_var_plot
    plot.scg.binary_classifier
```
