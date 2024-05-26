---
orphan: true
---

# Tutorials

The easiest way to get familiar with pertpy is to follow along with our tutorials.
Many are also designed to work seamlessly in Google colab.

:::{note}
For questions about the usage of pertpy use the [scverse discourse](https://discourse.scverse.org/).
:::

## Quick start

```{eval-rst}
.. nbgallery::

   notebooks/guide_rna_assignment
   notebooks/mixscape
   notebooks/augur
   notebooks/sccoda
   notebooks/sccoda_extended
   notebooks/tasccoda
   notebooks/milo
   notebooks/dialogue
   notebooks/enrichment
   notebooks/distances
   notebooks/distance_tests
   notebooks/cinemaot
   notebooks/scgen_perturbation_prediction
   notebooks/perturbation_space
   notebooks/differential_gene_expression
   notebooks/metadata_annotation
   notebooks/ontology_mapping
```

### Glossary

```{eval-rst}
.. tab-set::

    .. tab-item:: AnnData

        `AnnData <https://github.com/scverse/anndata>`_ is short for Annotated Data and is the primary datastructure that pertpy uses.
        It is based on the principle of a single Numpy matrix X embraced by two Pandas Dataframes.
        All rows are called observations (in our case cells or similar) and the columns
        are known as variables (any feature such as e.g. genes or similar).
        For a more in depth introduction please read the `AnnData paper <https://doi.org/10.1101/2021.12.16.473007>`_.


    .. tab-item:: scanpy

        For a more in depth introduction please read the `Scanpy paper <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0>`_.
```
