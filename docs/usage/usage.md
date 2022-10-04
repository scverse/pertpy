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

    data.burczynski_crohn
    data.papalexi_2021
    data.kang_2018
    data.sc_sim_augur
    data.bhattacherjee
    data.sciplex3_raw
    data.frangieh_2021
    data.frangieh_2021_raw
    data.dixit_2016
    data.dixit_2016_raw
    data.norman_2019
    data.norman_2019_raw
    data.dialogue_example
```

## Tools

### Augurpy

The Python implementation of [Augur R package](https://github.com/neurorestore/Augur) Skinnider, M.A., Squair, J.W., Kathe, C. et al. [Cell type prioritization in single-cell data](https://doi.org/10.1038/s41587-020-0605-1). Nat Biotechnol 39, 30–34 (2021).

Augurpy aims to rank or prioritize cell types according to their response to experimental perturbations given high dimensional single-cell sequencing data.
The basic idea is that in the space of molecular measurements cells reacting heavily to induced perturbations are more easily separated into perturbed and unperturbed than cell types with little or no response.
This separability is quantified by measuring how well experimental labels (eg. treatment and control) can be predicted within each cell type. Augurpy trains a machine learning model predicting experimental labels for each cell type in multiple cross validation runs and then prioritizes cell type response according to metric scores measuring the accuracy of the model. For categorical data the area under the curve is the default metric and for numerical data the concordance correlation coefficient is used as a proxy for how accurate the model is which in turn approximates perturbation response.

Example implementation

```python
import pertpy as pt

adata = pt.dt.sc_sim_augur()
ag = pt.tl.Augurpy(estimator="random_forest_classifier")
adata = ag.load(adata)
adata, results = ag.predict(adata)

# metrics for each cell type
results['summary_metrics']
```

See [augurpy tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/augurpy.html) for a more elaborate tutorial.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.Augurpy
```

### Mixscape

A Python implementation of [Mixscape](https://satijalab.org/seurat/articles/mixscape_vignette.html) Papalexi et al. [Characterizing the molecular regulation of inhibitory immune checkpoints with multimodal single-cell screens](https://www.nature.com/articles/s41588-021-00778-2).

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

See [mixscape tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/mixscape.html) for a more elaborate tutorial.

### Milopy

Basic python implementation of Milo for differential abundance testing on KNN graphs, to ease interoperability with scanpy pipelines for single-cell analysis. See [preprint](https://www.biorxiv.org/content/10.1101/2020.11.23.393769v1) for details on the statistical framework.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.Milopy
```

See [milopy tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/milopy.html) for a more elaborate tutorial.

### Representation

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.kernel_pca
```

## Plots

### Augurpy

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.ag.dp_scatter
    plot.ag.important_features
    plot.ag.lollipop
    plot.ag.scatterplot

```

### Mixscape

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.ms.violin
    plot.ms.perturbscore
    plot.ms.heatmap
    plot.ms.barplot
    plot.ms.lda
```

### Milopy

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.milo.nhood_graph
    plot.milo.nhood
    plot.milo.da_beeswarm
    plot.milo.nhood_counts_by_cond
```
