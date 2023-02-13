from __future__ import annotations

from typing import Literal

import decoupler as dc
import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData


class DifferentialGeneExpression:
    """Support for differential gene expression for scverse."""

    def pseudobulk(
        self,
        adata: AnnData,
        sample_col: str,
        groups_col: str,
        obs: pd.DataFrame = None,
        layer: str = None,
        use_raw: bool = False,
        min_prop: float = 0.2,
        min_counts: int = 1000,
        min_samples: int = 2,
        dtype: npt.DTypeLike = np.float32,
    ) -> AnnData:
        """Generate Pseudobulk for DE analysis.

        Wraps decoupler's get_pseudobulk function.
        See: https://decoupler-py.readthedocs.io/en/latest/generated/decoupler.get_pseudobulk.html#decoupler.get_pseudobulk
        for more details

        Args:
            adata: Input AnnData object.
            sample_col: Column of obs where to extract the samples names.
            groups_col: Column of obs where to extract the groups names.
            obs: If provided, metadata dataframe.
            layer: If provided, which layer to use.
            use_raw: Use raw attribute of adata if present.
            min_prop: Minimum proportion of cells with non-zero values.
            min_counts: Minimum number of cells per sample.
            min_samples: Minimum number of samples per feature.
            dtype: Type of float used.

        Returns:
            Returns new AnnData object with unormalized pseudobulk profiles per sample and group.
        """
        pseudobulk_adata = dc.get_pseudobulk(
            adata,
            sample_col=sample_col,
            groups_col=groups_col,
            obs=obs,
            layer=layer,
            use_raw=use_raw,
            min_prop=min_prop,
            min_counts=min_counts,
            min_smpls=min_samples,
            dtype=dtype,
        )

        return pseudobulk_adata

    def de_analysis(
        self,
        adata: AnnData,
        groupby: str,
        method: Literal["t-test", "wilcoxon", "pydeseq2", "deseq2", "edger"],
        *formula: str | None,
        contrast: str | None,
        inplace: bool = True,
        key_added: str | None,
    ) -> pd.DataFrame:
        """Perform differential expression analysis.

        Args:
            adata: single-cell or pseudobulk AnnData object
            groupby: Column in adata.obs that contains the factor to test, e.g. `treatment`.
                     For simple statistical tests (t-test, wilcoxon), it is sufficient to specify groupby.
                     Linear models require to specify a formula.
                     In that case, the `groupby` column is used to compute the contrast.
            method: Which method to use to perform the DE test.
            formula: model specification for linear models. E.g. `~ treatment + sex + age`.
                     MUST contain the factor specified in `groupby`.
            contrast: See e.g. https://www.statsmodels.org/devel/contrasts.html for more information.
            inplace: if True, save the result in `adata.varm[key_added]`
            key_added: Key under which the result is saved in `adata.varm` if inplace is True.
                       If set to None this defaults to `de_{method}_{groupby}`.
        Returns:
            Depending on the method a Pandas DataFrame containing at least:
            * gene_id
            * log2 fold change
            * mean expression
            * unadjusted p-value
            * adjusted p-value
        """
        pass
