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

See [here](tutorials) for a more elaborate tutorial.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools
    :nosignatures:

    tools.Augurpy
```

### Perturbation signature

The python implementation of Mixscape's [CalcPerturbSig](https://satijalab.org/seurat/reference/calcperturbsig).

To calculate perturbation signature, for each cell, we identify `n_neighbors` cells from the control pool with
the most similar mRNA expression profiles. The perturbation signature is calculated by subtracting
the averaged mRNA expression profile of the control neighbors from the mRNA expression profile
of each cell.

```{eval-rst}
.. currentmodule:: pertpy
```

```{eval-rst}
.. autosummary::
    :toctree: tools

    tools.pert_sign
    tools.Mixscape
```

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
```
