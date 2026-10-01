import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from fast_array_utils.conv import to_dense
from joblib import delayed, effective_n_jobs
from scipy import stats
from scipy.sparse import issparse
from statsmodels.tools.sm_exceptions import SingularMatrixWarning
from statsmodels.tools.tools import pinv_extended, recipr

from pertpy._parallel import _MAX_BLOCK_ELEMENTS, _block_slices, _parallelize_with_joblib
from pertpy._types import CSBase

from ._base import LinearModelBase
from ._checks import check_is_numeric_matrix


def _fit_block(regression_model, endog_block: np.ndarray, exog, kwargs: dict) -> list:
    """Fit one model per column of `endog_block`."""
    return [regression_model(endog_block[:, i], exog, **kwargs).fit() for i in range(endog_block.shape[1])]


class _OLSFits(Sequence):
    """OLS fits of every column of `endog` on the same design, of which the statsmodels results are only built when accessed.

    All fits share the pseudo-inverse of the design, so their coefficients and residual variances are computed at once.
    """

    def __init__(self, endog, exog):
        self.endog, self.exog = endog, exog
        design = np.asarray(exog, dtype=np.float64)
        pinv_exog, singular_values = pinv_extended(design)
        rank = np.linalg.matrix_rank(np.diag(singular_values))
        if rank < design.shape[1]:
            warnings.warn(
                "The design matrix is rank-deficient. The model parameters are not uniquely determined.",
                SingularMatrixWarning,
                stacklevel=3,
            )
        self.normalized_cov_params = pinv_exog @ pinv_exog.T
        self.df_resid = float(design.shape[0] - rank)
        self.params = np.empty((endog.shape[1], design.shape[1]))
        self.scale = np.empty(endog.shape[1])
        for block in _block_slices(endog.shape[1], max_block_size=_MAX_BLOCK_ELEMENTS // max(design.shape[0], 1)):
            values = to_dense(endog[:, block])
            self.params[block] = (pinv_exog @ values).T
            resid = values - design @ self.params[block].T
            self.scale[block] = np.sum(resid * resid, axis=0) / self.df_resid

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]
        return sm.OLS(to_dense(self.endog[:, [index]])[:, 0], self.exog).fit()


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
                Unused by OLS without further arguments, which fits all variables at once.
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

        if regression_model is sm.OLS and not kwargs:
            self.models: Sequence = _OLSFits(data, self.design)
            self._fitted = True
            return

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
        r_matrix = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
        if isinstance(self.models, _OLSFits):
            effect = self.models.params @ r_matrix[0]
            sd = np.sqrt(self.models.scale * (r_matrix @ self.models.normalized_cov_params @ r_matrix.T).item())
            df_resid = np.full(len(effect), self.models.df_resid)
            use_t = np.ones(len(effect), dtype=bool)
        else:
            results = [getattr(mod, "_results", mod) for mod in self.models]
            effect = np.array([np.dot(r_matrix, res.params.ravel(order="F")).item() for res in results])
            sd = np.array([np.sqrt(res.cov_params(r_matrix=r_matrix)).item() for res in results])
            df_resid = np.array([getattr(res, "df_resid_inference", res.df_resid) for res in results])
            use_t = np.array([getattr(res, "use_t", False) for res in results], dtype=bool)
        t_value = effect * recipr(sd)
        p_value = np.where(use_t, stats.t.sf(np.abs(t_value), df_resid), stats.norm.sf(np.abs(t_value))) * 2
        return (
            pd.DataFrame(
                {
                    "variable": self.adata.var_names,
                    "p_value": p_value,
                    "t_value": t_value,
                    "sd": sd,
                    "log_fc": effect,
                }
            )
            .sort_values("p_value")
            .assign(adj_p_value=lambda x: statsmodels.stats.multitest.fdrcorrection(x["p_value"])[1])
        )
