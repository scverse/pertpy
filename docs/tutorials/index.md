---
orphan: true
---

# Tutorials

The easiest way to get familiar with pertpy is to follow along with our tutorials.
Many are also designed to work seamlessly in Google colab.

:::{note}
For questions about the usage of pertpy use the [scverse discourse](https://discourse.scverse.org/).
:::

## Quick start: Tool specific tutorials

### Data transformation

```{eval-rst}
.. nbgallery::

   notebooks/guide_rna_assignment
   notebooks/mixscape
   notebooks/perturbation_space
   notebooks/metadata_annotation
   notebooks/ontology_mapping
```

### Knowledge inference

```{eval-rst}
.. nbgallery::

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
   notebooks/differential_gene_expression
```

## Use cases

Our use cases showcase a variety of pertpy tools applied to one dataset.
They are designed to give you a sense of how to use pertpy in a real-world scenario.
The use cases featured here are those we present in the pertpy [preprint](https://www.biorxiv.org/content/10.1101/2024.08.04.606516v1).

```{eval-rst}
.. nbgallery::

   notebooks/norman_use_case
   notebooks/mcfarland_use_case
   notebooks/zhang_use_case
```

## Glossary

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
