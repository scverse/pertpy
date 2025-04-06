import os
import re
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy import ndarray
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from scipy.sparse import issparse

from ._base import LinearModelBase
from ._checks import check_is_integer_matrix


class PyDESeq2(LinearModelBase):
    """Differential expression test using a PyDESeq2."""

    def __init__(
        self, adata: AnnData, design: str | ndarray, *, mask: str | None = None, layer: str | None = None, **kwargs
    ):
        super().__init__(adata, design, mask=mask, layer=layer, **kwargs)
        # work around pydeseq2 issue with sparse matrices
        # see also https://github.com/owkin/PyDESeq2/issues/25
        if issparse(self.data):
            if self.layer is None:
                self.adata.X = self.adata.X.toarray()
            else:
                self.adata.layers[self.layer] = self.adata.layers[self.layer].toarray()

    def _check_counts(self):
        check_is_integer_matrix(self.data)

    def fit(self, **kwargs) -> pd.DataFrame:
        """Fit dds model using pydeseq2.

        Note: this creates its own AnnData object for downstream processing.

        Args:
            **kwargs: Keyword arguments specific to DeseqDataSet(), except for `n_cpus` which will use all available CPUs minus one if the argument is not passed.
        """
        try:
            usable_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            usable_cpus = os.cpu_count()

        inference = DefaultInference(n_cpus=kwargs.pop("n_cpus", usable_cpus))

        dds = DeseqDataSet(
            adata=self.adata,
            design=self.design,  # initialize using design matrix, not formula
            refit_cooks=True,
            inference=inference,
            **kwargs,
        )

        dds.deseq2()
        self.dds = dds

    def _test_single_contrast(self, contrast, alpha=0.05, **kwargs) -> pd.DataFrame:
        """Conduct a specific test and returns a Pandas DataFrame.

        Args:
            contrast: list of three strings of the form `["variable", "tested level", "reference level"]`.
            alpha: p value threshold used for controlling fdr with independent hypothesis weighting
            **kwargs: extra arguments to pass to DeseqStats()
        """
        contrast = np.array(contrast)
        stat_res = DeseqStats(self.dds, contrast=contrast, alpha=alpha, **kwargs)
        # Calling `.summary()` is required to fill the `results_df` data frame
        stat_res.summary()
        res_df = (
            pd.DataFrame(stat_res.results_df)
            .rename(columns={"pvalue": "p_value", "padj": "adj_p_value", "log2FoldChange": "log_fc"})
            .sort_values("p_value")
        )
        res_df.index.name = "variable"
        res_df = res_df.reset_index()
        return res_df
