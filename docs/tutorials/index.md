---
orphan: true
---

# Tutorials

The easiest way to get familiar with pertpy is to follow along with our tutorials.
Many are also designed to work seamlessly in Binder, a free cloud computing platform.

:::{note}
For questions about the usage of pertpy use [Github Discussions].
:::

## Quick start

```{eval-rst}
.. nbgallery::

   notebooks/mixscape
   notebooks/augurpy

```

### Glossary

```{eval-rst}
.. tab-set::

    .. tab-item:: AnnData

        `AnnData <https://github.com/theislab/anndata>`_ is short for Annotated Data and is the primary datastructure that pertpy uses.
        It is based on the principle of a single Numpy matrix X embraced by two Pandas Dataframes.
        All rows are called observations (in our case cells or similar) and the columns
        are known as variables (any feature such as e.g. genes or similar).
        For a more in depth introduction please read the `AnnData paper <https://doi.org/10.1101/2021.12.16.473007>`_.


    .. tab-item:: scanpy

        For a more in depth introduction please read the `Scanpy paper <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0>`_.
```

[github discussions]: https://github.com/theislab/pertpy/discussions
