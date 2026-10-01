import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fast_array_utils.conv import to_dense
from joblib import delayed, effective_n_jobs
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

from pertpy._parallel import _MAX_BLOCK_ELEMENTS, _block_slices, _parallelize_with_joblib
from pertpy._types import CSBase
from pertpy.tools._milo_glmm import parse_random_effects

from ._base import LinearModelBase
from ._checks import check_is_numeric_matrix


def _fit_block(endog_block: np.ndarray, exog: np.ndarray, groups: np.ndarray, kwargs: dict) -> list:
    """Fit one mixed model per column of `endog_block`.

    The default gradient-based optimizers of statsmodels regularly reach the optimum but fail their gradient
    tolerance check. Unless an optimizer was requested explicitly, such fits are repeated with derivative-free
    optimizers, as the statsmodels convergence warning suggests.
    Per-variable warnings are silenced here and reported once, via the `converged` column of the results.
    Variables whose model cannot be fit at all (e.g. constant expression) yield `None`.
    """
    models = []
    for i in range(endog_block.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = sm.MixedLM(endog_block[:, i], exog, groups=groups)
                result = model.fit(**kwargs)
                if not result.converged and "method" not in kwargs:
                    result = model.fit(**kwargs, method=["powell", "nm"])
                models.append(result)
            except (np.linalg.LinAlgError, ValueError):
                models.append(None)
    return models


def _between_within_df(design: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Classify design columns and compute between-within denominator degrees of freedom.

    A column is *between-group* if it is constant within every group (e.g. a condition assigned per donor,
    or the intercept), otherwise *within-group* (e.g. a condition that varies across the cells of a donor).

    Returns:
        A boolean mask of the between-group columns, the between-group degrees of freedom
        (number of groups minus the rank of the between-group columns) and the within-group degrees of freedom
        (number of observations minus the rank of the group indicators together with the within-group columns).
    """
    codes = pd.factorize(groups)[0]
    n_groups = int(codes.max()) + 1
    group_means = np.zeros((n_groups, design.shape[1]))
    np.add.at(group_means, codes, design)
    group_means /= np.bincount(codes, minlength=n_groups)[:, None]
    is_between = np.all(np.isclose(design, group_means[codes]), axis=0)

    df_between = n_groups - np.linalg.matrix_rank(group_means[:, is_between])
    indicators = np.eye(n_groups)[codes]
    df_within = design.shape[0] - np.linalg.matrix_rank(np.column_stack([indicators, design[:, ~is_between]]))
    return is_between, int(df_between), int(df_within)


class LinearMixedModel(LinearModelBase):
    """Differential expression test using a linear mixed model with a random intercept.

    Fits one :class:`statsmodels.regression.mixed_linear_model.MixedLM` per variable.
    This allows testing all cells while accounting for the sample they come from,
    or testing pseudobulk data whose design requires a random effect.
    As for :class:`~pertpy.tools.Statsmodels`, the data should be on a continuous scale,
    e.g. log-normalized expression.

    The random intercept is specified in the formula using the ``(1 | variable)`` syntax of lme4,
    for example ``"~condition + (1 | donor)"``. Exactly one random intercept is supported.

    Fixed effects are tested with a t-test on the requested contrast. Because statsmodels only provides
    a normal approximation, which is anti-conservative when there are few groups, the denominator degrees
    of freedom follow the between-within method: effects that are constant within each group
    (e.g. a condition assigned per donor) are tested against the number of groups minus the number of such
    effects, effects varying within groups against the within-group residual degrees of freedom.
    Contrasts mixing both kinds use the smaller of the two.

    Examples:
        >>> import pertpy as pt
        >>> model = pt.tl.LinearMixedModel(adata, design="~condition + (1 | donor)")  # doctest: +SKIP
        >>> model.fit()  # doctest: +SKIP
        >>> res = model.test_contrasts(
        ...     model.contrast(column="condition", baseline="control", group_to_compare="treated")
        ... )  # doctest: +SKIP
    """

    def __init__(self, adata, design, *, mask=None, layer=None, **kwargs):
        if not isinstance(design, str):
            raise TypeError(
                "LinearMixedModel requires a formula with a random intercept, e.g. '~condition + (1 | donor)'."
            )
        fixed_design, random_effects = parse_random_effects(design)
        if len(random_effects) != 1:
            raise ValueError(
                f"LinearMixedModel requires exactly one random intercept in the formula, e.g. '~condition + (1 | donor)', "
                f"but {design!r} has {len(random_effects)}."
            )
        (random_effect,) = random_effects
        if random_effect not in adata.obs.columns:
            raise ValueError(f"Random effect variable {random_effect!r} is not a column of adata.obs.")
        if adata.obs[random_effect].isna().any():
            raise ValueError(f"Random effect variable {random_effect!r} contains missing values.")
        if adata.obs[random_effect].nunique() < 2:
            raise ValueError(f"Random effect variable {random_effect!r} needs at least two levels.")

        super().__init__(adata, fixed_design, mask=mask, layer=layer, **kwargs)
        self.random_effect = random_effect
        self.groups = adata.obs[random_effect].astype(str).to_numpy()
        self.is_between, self.df_between, self.df_within = _between_within_df(
            np.asarray(self.design, dtype=float), self.groups
        )

    @classmethod
    def _comparison_design(cls, column: str, paired_by: str | None) -> str:
        if paired_by is None:
            raise ValueError(
                "LinearMixedModel.compare_groups requires `paired_by`, which enters the model as the random intercept."
            )
        return f"~{column} + (1 | {paired_by})"

    def _check_counts(self):
        check_is_numeric_matrix(self.data)

    def fit(self, *, reml: bool = True, n_jobs: int | None = None, **kwargs) -> None:
        """Fit one linear mixed model per variable.

        Args:
            reml: Whether to fit by restricted maximum likelihood (recommended) instead of maximum likelihood.
            n_jobs: Number of jobs to distribute the variables over, passed to `joblib.Parallel`.
                None means one job, -1 all available cores.
            **kwargs: Additional arguments passed to :meth:`statsmodels.regression.mixed_linear_model.MixedLM.fit`,
                e.g. `method`.

        Examples:
            >>> import pertpy as pt
            >>> model = pt.tl.LinearMixedModel(adata, design="~condition + (1 | donor)")  # doctest: +SKIP
            >>> model.fit()  # doctest: +SKIP
        """
        data = self.data
        # Slicing out variables is significantly more efficient in csc format.
        if isinstance(data, CSBase):
            data = data.tocsc()

        exog = np.asarray(self.design, dtype=float)
        fit_kwargs = {"reml": reml, **kwargs}
        n_workers = effective_n_jobs(n_jobs if n_jobs is not None else 1)
        blocks = _block_slices(
            self.adata.n_vars,
            n_blocks=self.adata.n_vars if n_workers == 1 else 4 * n_workers,
            max_block_size=_MAX_BLOCK_ELEMENTS // max(self.adata.n_obs, 1),
        )
        self.models = [
            model
            for block_models in _parallelize_with_joblib(
                (delayed(_fit_block)(to_dense(data[:, block]), exog, self.groups, fit_kwargs) for block in blocks),
                total=len(blocks),
                n_jobs=n_jobs,
            )
            for model in block_models
        ]
        n_failed = sum(model is None or not model.converged for model in self.models)
        if n_failed:
            warnings.warn(
                f"The mixed model did not converge (or could not be fit) for {n_failed} of {len(self.models)} variables. "
                "See the `converged` column of the results.",
                UserWarning,
                stacklevel=2,
            )
        self._fitted = True

    def _contrast_df(self, contrast: np.ndarray) -> int:
        """Denominator degrees of freedom for a contrast, following the between-within method."""
        involved = ~np.isclose(contrast, 0)
        df = min(
            self.df_between if (involved & self.is_between).any() else np.inf,
            self.df_within if (involved & ~self.is_between).any() else np.inf,
        )
        if df < 1:
            raise ValueError(
                f"The contrast cannot be tested: it involves effects that are constant within `{self.random_effect}`, "
                f"but there are not enough levels of `{self.random_effect}` left to estimate them "
                f"(between-group degrees of freedom: {self.df_between})."
            )
        return int(df)

    def _test_single_contrast(self, contrast, **kwargs) -> pd.DataFrame:
        contrast = np.asarray(contrast, dtype=float)
        df = self._contrast_df(contrast)
        res = []
        for var, mod in zip(tqdm(self.adata.var_names), self.models, strict=False):
            row = {"variable": var, "p_value": np.nan, "t_value": np.nan, "sd": np.nan, "log_fc": np.nan}
            if mod is not None:
                # MixedLMResults.t_test expects a 2D restriction matrix over the fixed effects.
                t_test = mod.t_test(np.atleast_2d(contrast))
                effect, sd = float(np.ravel(t_test.effect)[0]), float(np.ravel(t_test.sd)[0])
                t_value = effect / sd if sd > 0 else np.nan
                row |= {
                    "p_value": 2 * stats.t.sf(abs(t_value), df) if np.isfinite(t_value) else np.nan,
                    "t_value": t_value,
                    "sd": sd,
                    "log_fc": effect,
                }
            row["converged"] = mod is not None and bool(mod.converged)
            res.append(row)

        res_df = pd.DataFrame(res).sort_values("p_value")
        tested = res_df["p_value"].notna().to_numpy()
        adj = np.full(len(res_df), np.nan)
        if tested.any():
            adj[tested] = fdrcorrection(res_df["p_value"].to_numpy()[tested])[1]
        return res_df.assign(adj_p_value=adj)
