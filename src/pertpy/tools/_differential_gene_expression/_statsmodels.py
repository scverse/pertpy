import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData
from fast_array_utils.conv import to_dense
from joblib import delayed, effective_n_jobs
from scipy import stats
from scipy.sparse import issparse
from statsmodels.tools.sm_exceptions import SingularMatrixWarning
from statsmodels.tools.tools import pinv_extended, recipr

from pertpy._parallel import _MAX_BLOCK_ELEMENTS, _block_slices, _numba_threads, _parallelize_with_joblib
from pertpy._types import CSBase
from pertpy.tools._mixed_models import _random_intercept_contrast, _random_intercept_response_fits, random_effect_codes

from ._base import LinearModelBase
from ._checks import check_is_numeric_matrix
from ._simple_tests import fdr_correction


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


class _MixedFits(Sequence):
    """REML fits of every column of `endog` on the same design with a random intercept per level of `groups`, of which the statsmodels results are only built when accessed.

    Contrasts of the fixed effects are tested with Satterthwaite degrees of freedom like `lmerTest`.
    Columns whose fit is degenerate, such as constant ones, hold NaN.
    """

    def __init__(self, endog, exog, groups: np.ndarray, n_groups: int):
        self.endog, self.exog, self.groups = endog, exog, groups
        design = np.ascontiguousarray(exog, dtype=np.float64)
        n_vars, p = endog.shape[1], design.shape[1]
        self.params = np.full((n_vars, p), np.nan)
        self.inverses = np.full((n_vars, p, p), np.nan)
        self.gammas = np.full(n_vars, np.nan)
        self.scale = np.full(n_vars, np.nan)
        self.residual_sums = np.full((n_vars, n_groups), np.nan)
        self.design_sums = np.zeros((n_groups, p))
        self.sizes = np.zeros(n_groups)
        for block in _block_slices(n_vars, max_block_size=_MAX_BLOCK_ELEMENTS // max(design.shape[0], 1)):
            responses = np.ascontiguousarray(to_dense(endog[:, block]).T, dtype=np.float64)
            with _numba_threads(responses.shape[0]):
                fits = _random_intercept_response_fits(responses, design, groups, n_groups)
            (
                self.params[block],
                self.inverses[block],
                self.gammas[block],
                self.scale[block],
                self.residual_sums[block],
            ) = fits[:5]
            self.design_sums, self.sizes = fits[5:]

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]
        return sm.MixedLM(to_dense(self.endog[:, [index]])[:, 0], self.exog, groups=self.groups).fit()

    def contrast(self, contrast: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate, standard error and Satterthwaite degrees of freedom of `contrast` for every column."""
        with _numba_threads(len(self)):
            return _random_intercept_contrast(
                np.ascontiguousarray(contrast, dtype=np.float64),
                self.params,
                self.inverses,
                self.gammas,
                self.scale,
                self.residual_sums,
                self.design_sums,
                self.sizes,
            )


class Statsmodels(LinearModelBase):
    """Differential expression test using a statsmodels linear regression.

    A random intercept written as ``(1 | variable)`` in the design, e.g. ``"~condition + (1 | donor)"``, turns the linear regression into a linear mixed model.
    It is fitted by REML and its contrasts are tested with Satterthwaite degrees of freedom like ``lmerTest``, so effects that are constant within a donor are judged by the number of donors rather than the number of cells.
    This is how to test single cells of a cell type while accounting for the samples they come from.
    """

    _supports_random_effects = True

    def __init__(
        self, adata: AnnData, design: str | np.ndarray, *, mask: str | None = None, layer: str | None = None, **kwargs
    ):
        super().__init__(adata, design, mask=mask, layer=layer, **kwargs)
        if len(self.random_effects) > 1:
            raise ValueError(
                f"Statsmodels supports a single random intercept, but the design has {len(self.random_effects)}: "
                f"{self.random_effects}."
            )
        self.groups: np.ndarray | None = None
        self.n_groups = 0
        if self.random_effects:
            self.groups, self.n_groups = random_effect_codes(self.adata.obs, self.random_effects[0])

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
                A design with a random intercept requires OLS, which then fits the linear mixed model.
            n_jobs: Number of jobs to distribute the variables over, passed to `joblib.Parallel`.
                None means one job, -1 all available cores.
                Unused by OLS without further arguments and by the mixed model, which fit all variables at once.
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

        if self.groups is not None:
            if regression_model is not sm.OLS or kwargs:
                raise ValueError("A random intercept is only supported for OLS without further fit arguments.")
            self.models: Sequence = _MixedFits(data, self.design, self.groups, self.n_groups)
            self._fitted = True
            return
        if regression_model is sm.OLS and not kwargs:
            self.models = _OLSFits(data, self.design)
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
        if isinstance(self.models, _MixedFits):
            effect, sd, df_resid = self.models.contrast(r_matrix[0])
            use_t = np.ones(len(effect), dtype=bool)
        elif isinstance(self.models, _OLSFits):
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
        return fdr_correction(
            pd.DataFrame(
                {
                    "variable": self.adata.var_names,
                    "p_value": p_value,
                    "t_value": t_value,
                    "sd": sd,
                    "log_fc": effect,
                }
            ).sort_values("p_value")
        )
