Usage
=====


Import the pertpy API as follows:

.. code:: python

   import pertpy as pp

You can then access the respective modules like:

.. code:: python

   pp.pl.cool_fancy_plot()


.. currentmodule:: pertpy

Tools
-----

Augurpy
~~~~~~~

the python implementation of `Augur R package <https://github.com/neurorestore/Augur>`_ Skinnider, M.A., Squair, J.W., Kathe, C. et al. `Cell type prioritization in single-cell data <https://doi.org/10.1038/s41587-020-0605-1>`_. Nat Biotechnol 39, 30–34 (2021).

Augurpy aims to rank or prioritize cell types according to the their response to experimental perturbations given high dimensional single-cell sequencing data.
The basic idea is that in the space of molecular measurements cells reacting heavily to induced perturbations are more easily separated into perturbed and unperturbed than cell types with little or no response.
This seperability is quantified by measuring how well experimental labels (eg. treatment and control) can be predicted within each cell type. Augurpy trains a machine learning model predicting experimental labels for each cell type in multiple cross validation runs and then prioritizes cell type response according to metric scores measuring the accuracy of the model. For categorical data the area under the curve is the default metric and for numerical data the concordance correlation coefficient is used as a proxy for how accurate the model is which in turn approximates perturbation response.

Example implementation

.. code:: python

   import pertpy as pp

   adata = '<data_you_want_to_analyse.h5ad>'
   adata = pp.tl.ag.read_load.load(adata, label_col = "<label_col_name>", cell_type_col = "<cell_type_col_name>")
   classifier = pp.tl.ag.estimator.create_estimator(classifier='random_forest_classifier')
   adata, results = pp.tl.ag.evaluate.predict(adata, classifier)

   # metrics for each cell type
   results['summary_metrics']

See `here <tutorials>`_ for a more elaborate tutorial.


.. currentmodule:: pertpy

Loading Data
++++++++++++

.. autosummary::
    :toctree: read_load

    tools.augurpy.read_load.load

Estimator
+++++++++

.. autosummary::
    :toctree: estimator

    tools.augurpy.estimator.create_estimator
    tools.augurpy.estimator.Params

Evaluation
++++++++++

.. autosummary::
    :toctree: evaluate

    tools.augurpy.evaluate.predict

Differential Prioritization
+++++++++++++++++++++++++++

.. autosummary::
    :toctree: differential_prioritization

    tools.augurpy.differential_prioritization.predict_differential_prioritization

Plots
-----

Augurpy
~~~~~~~

.. autosummary::
    :toctree: plot

    plot.augurpy.differential_prioritization.dp_scatter
    plot.augurpy.important_features.important_features
    plot.augurpy.lollipop.lollipop
    plot.augurpy.scatterplot.scatterplot


Data
----

.. autosummary::
    :toctree: data

    data.burczynski_crohn

