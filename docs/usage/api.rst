===
API
===

Import the pertpy API as follows:

.. code:: python

   import pertpy.api as py

You can then access the respective modules like:

.. code:: python

   py.pl.cool_fancy_plot()


.. currentmodule:: pertpy.api

Tools
~~~~~

Augurpy
#######

Import augurpy as follows:

.. code:: python

   import augurpy as ag

You can then access the respective modules like:

.. code:: python

   ag.function()

Example implementation

.. code:: python

   import augurpy as ag

   adata = '<data_you_want_to_analyse.h5ad>'
   adata = ag.load(adata, label_col = "<label_col_name>", cell_type_col = "<cell_type_col_name>")
   classifier = ag.create_estimator(classifier='random_forest_classifier')
   adata, results = ag.predict(adata, classifier)

   # metrics for each cell type
   results['summary_metrics']

See `here <tutorials>`_ for a more elaborate tutorial.


.. currentmodule:: pertpy.tools.augurpy

Loading Data
>>>>>>>>>>>>

.. module:: pertpy.tools.augurpy

.. autosummary::
    :toctree: read_load

    read_load.load

Estimator
>>>>>>>>>

.. autosummary::
    :toctree: estimator

    estimator.create_estimator
    estimator.Params

Evaluation
>>>>>>>>>>

.. autosummary::
    :toctree: evaluate

    evaluate.predict

Differential Prioritization
>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. autosummary::
    :toctree: differential_prioritization

    differential_prioritization.predict_differential_prioritization

Plots
>>>>>

.. autosummary::
    :toctree: pl

    pl.plot_differential_prioritization.plot_differential_prioritization
    pl.important_features.important_features
    pl.lollipop.lollipop
    pl.scatterplot.scatterplot


Data
~~~~~

.. module:: pertpy.data

.. autosummary::
    :toctree: data

    data.burczynski_crohn
