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

## Tools

### Augurpy

The Python implementation of [Augur R package](https://github.com/neurorestore/Augur) Skinnider, M.A., Squair, J.W., Kathe, C. et al. [Cell type prioritization in single-cell data](https://doi.org/10.1038/s41587-020-0605-1). Nat Biotechnol 39, 30–34 (2021).

Augurpy aims to rank or prioritize cell types according to their response to experimental perturbations given high dimensional single-cell sequencing data.
The basic idea is that in the space of molecular measurements cells reacting heavily to induced perturbations are more easily separated into perturbed and unperturbed than cell types with little or no response.
This separability is quantified by measuring how well experimental labels (eg. treatment and control) can be predicted within each cell type. Augurpy trains a machine learning model predicting experimental labels for each cell type in multiple cross validation runs and then prioritizes cell type response according to metric scores measuring the accuracy of the model. For categorical data the area under the curve is the default metric and for numerical data the concordance correlation coefficient is used as a proxy for how accurate the model is which in turn approximates perturbation response.

Example implementation

```python
import pertpy as pt

adata = '<data_you_want_to_analyse.h5ad>'
adata = pt.tl.ag.read_load.load(adata, label_col = "<label_col_name>", cell_type_col = "<cell_type_col_name>")
classifier = pt.tl.ag.estimator.create_estimator(classifier='random_forest_classifier')
adata, results = pt.tl.ag.evaluate.predict(adata, classifier)

# metrics for each cell type
results['summary_metrics']
```

See [here](tutorials) for a more elaborate tutorial.

```{eval-rst}
.. currentmodule:: pertpy
```

#### Loading Data

```{eval-rst}
.. autosummary::
    :toctree: read_load

    tools.augurpy.read_load.load
```

#### Estimator

```{eval-rst}
.. autosummary::
    :toctree: estimator

    tools.augurpy.estimator.create_estimator

    :template: param_class.rst
    :toctree: estimator

    tools.augurpy.estimator.Params


```

#### Evaluation

```{eval-rst}
.. autosummary::
    :toctree: evaluate

    tools.augurpy.evaluate.predict
```

#### Differential Prioritization

```{eval-rst}
.. autosummary::
    :toctree: differential_prioritization

    tools.augurpy.differential_prioritization.predict_differential_prioritization
```

## Plots

### Augurpy

```{eval-rst}
.. autosummary::
    :toctree: plot

    plot.augurpy.differential_prioritization.dp_scatter
    plot.augurpy.important_features.important_features
    plot.augurpy.lollipop.lollipop
    plot.augurpy.scatterplot.scatterplot

```

## Data

```{eval-rst}
.. autosummary::
    :toctree: data

    data.burczynski_crohn
    data.burczynski_crispr
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
