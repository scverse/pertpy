import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from fast_array_utils.conv import to_dense
from joblib import delayed, effective_n_jobs
from scipy.sparse import issparse
from tqdm.auto import tqdm

from pertpy._parallel import _MAX_BLOCK_ELEMENTS, _block_slices, _parallelize_with_joblib
from pertpy._types import CSBase

from ._base import LinearModelBase
from ._checks import check_is_numeric_matrix


def _fit_block(regression_model, endog_block: np.ndarray, exog, kwargs: dict) -> list:
    """Fit one model per column of `endog_block`."""
    return [regression_model(endog_block[:, i], exog, **kwargs).fit() for i in range(endog_block.shape[1])]


class Statsmodels(LinearModelBase):
    """Differential expression test using a statsmodels linear regression."""

    def _check_counts(self):
        check_is_numeric_matrix(self.data)

    def fit(
        self,
        regression_model: type[sm.OLS] | type[sm.GLM] = sm.OLS,
        *,
        n_jobs: int | None = None,
        **kwargs,
    ) -> None:
        """Fit the specified regression model.

        Args:
            regression_model: A statsmodels regression model class, either OLS or GLM.
            n_jobs: Number of jobs to distribute the variables over, passed to `joblib.Parallel`.
                None means one job, -1 all available cores.
            **kwargs: Additional arguments for fitting the specific method. In particular, this
                is where you can specify the family for GLM.

        Examples:
            >>> import statsmodels.api as sm
            >>> import pertpy as pt
            >>> model = pt.tl.Statsmodels(adata, design="~condition")
            >>> model.fit(sm.GLM, family=sm.families.NegativeBinomial(link=sm.families.links.Log()))
            >>> results = model.test_contrasts(np.array([0, 1]))
        """
        data = self.data
        # Slicing out variables is significantly more efficient in csc format.
        if isinstance(data, CSBase):
            data = data.tocsc()

        n_workers = effective_n_jobs(n_jobs if n_jobs is not None else 1)
        blocks = _block_slices(
            self.adata.n_vars,
            # Across processes, larger blocks amortize sending the design matrix.
            n_blocks=self.adata.n_vars if n_workers == 1 else 4 * n_workers,
            max_block_size=_MAX_BLOCK_ELEMENTS // max(self.adata.n_obs, 1),
        )
        self.models = [
            model
            for block_models in _parallelize_with_joblib(
                (
                    delayed(_fit_block)(regression_model, to_dense(data[:, block]), self.design, dict(kwargs))
                    for block in blocks
                ),
                total=len(blocks),
                n_jobs=n_jobs,
            )
            for model in block_models
        ]
        self._fitted = True

    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        res = []
        for var, mod in zip(tqdm(self.adata.var_names), self.models, strict=False):
            t_test = mod.t_test(contrast)
            res.append(
                {
                    "variable": var,
                    "p_value": t_test.pvalue,
                    "t_value": t_test.tvalue.item(),
                    "sd": t_test.sd.item(),
                    "log_fc": t_test.effect.item(),
                }
            )
        return (
            pd.DataFrame(res)
            .sort_values("p_value")
            .assign(adj_p_value=lambda x: statsmodels.stats.multitest.fdrcorrection(x["p_value"])[1])
        )
