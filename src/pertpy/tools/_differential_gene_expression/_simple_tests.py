"""Simple tests such as t-test, wilcoxon."""

import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels
from anndata import AnnData
from fast_array_utils.conv import to_dense
from joblib import delayed
from numba import njit, prange
from pandas.core.api import DataFrame
from scipy import special
from scipy.sparse import csc_array, diags, issparse

from pertpy._parallel import _MAX_BLOCK_ELEMENTS, _block_slices, _parallelize_with_joblib, _spawn_rngs

from ._base import MethodBase

_RNG_KWARGS = ("rng", "random_state")

TestStatistic = Callable[[np.ndarray, np.ndarray], float] | Callable[[np.ndarray, np.ndarray], np.ndarray]


def _var_block(x, block: slice, *, dense: bool = True):
    """Get a block of variables (columns), as a dense array if `dense`."""
    return to_dense(x[:, block]) if dense else x[:, block]


def _csc_columns(x) -> tuple[np.ndarray, np.ndarray]:
    """Column pointers and stored values of `x` in CSC format, as int64 and float64."""
    x = csc_array(x)
    return x.indptr.astype(np.int64, copy=False), x.data.astype(np.float64, copy=False)


@njit(parallel=True, cache=True)
def _column_sums(indptr: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum of the stored values of every column of a CSC matrix, with compensated summation since columns are long."""
    sums = np.empty(len(indptr) - 1)
    for j in prange(len(sums)):
        total = compensation = 0.0
        for value in values[indptr[j] : indptr[j + 1]]:
            updated = total + value
            compensation += (total - updated) + value if abs(total) >= abs(value) else (value - updated) + total
            total = updated
        sums[j] = total + compensation
    return sums


def _column_means(x) -> np.ndarray:
    """Mean of every column in float64, from the stored entries if `x` is sparse."""
    n = x.shape[0]
    if not issparse(x):
        return np.asarray(x).mean(axis=0, dtype=np.float64)
    return _column_sums(*_csc_columns(x)) / n


def _mean_var(x) -> tuple[np.ndarray, np.ndarray]:
    """Mean and unbiased variance of every column in float64, from the stored entries if `x` is sparse."""
    n = x.shape[0]
    mean = _column_means(x)
    if issparse(x):
        indptr, values = _csc_columns(x)
        stored = np.diff(indptr)
        squares = _column_sums(indptr, (values - np.repeat(mean, stored)) ** 2) + (n - stored) * mean**2
    else:
        squares = np.sum((np.asarray(x, dtype=np.float64) - mean) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mean, np.where(n == 1, 0.0, squares / n * (n / (n - 1)))


def _t_test_ind(x0, x1) -> dict[str, np.ndarray]:
    """:func:`scipy.stats.ttest_ind` with its default arguments for every column, from the column sums of sparse input."""
    (mean0, var0), (mean1, var1) = _mean_var(x0), _mean_var(x1)
    n0, n1 = x0.shape[0], x1.shape[0]
    df = n0 + n1 - 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean0 - mean1) / np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / df * (1.0 / n0 + 1.0 / n1))
    return {"p_value": 2 * special.stdtr(df, -np.abs(t)), "statistic": t}


@njit(parallel=True, cache=True)
def _rank_sums(  # noqa: PLR0917
    indptr0: np.ndarray, data0: np.ndarray, n0: int, indptr1: np.ndarray, data1: np.ndarray, n1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rank sum of the first group and tie term ``sum(t**3 - t)`` of every column of two CSC matrices.

    The zeros that are not stored enter as one tied block, so only the stored entries are sorted.
    """
    n_cols = len(indptr0) - 1
    rank_sum = np.empty(n_cols)
    tie_term = np.empty(n_cols)
    for j in prange(n_cols):
        stored0, stored1 = indptr0[j + 1] - indptr0[j], indptr1[j + 1] - indptr1[j]
        values = np.zeros(stored0 + stored1 + 1)
        values[:stored0] = data0[indptr0[j] : indptr0[j + 1]]
        values[stored0:-1] = data1[indptr1[j] : indptr1[j + 1]]
        weight = np.ones(len(values))
        weight[-1] = n0 + n1 - stored0 - stored1
        weight_first = np.zeros(len(values))
        weight_first[:stored0] = 1.0
        weight_first[-1] = n0 - stored0
        order = np.argsort(values)
        if np.isnan(values[order[-1]]):
            rank_sum[j] = tie_term[j] = np.nan
            continue
        total = ties = ranked = 0.0
        start = 0
        while start < len(order):
            count = count_first = 0.0
            end = start
            while end < len(order) and values[order[end]] == values[order[start]]:
                count += weight[order[end]]
                count_first += weight_first[order[end]]
                end += 1
            total += count_first * (ranked + (count + 1) / 2)
            ties += count**3 - count
            ranked += count
            start = end
        rank_sum[j], tie_term[j] = total, ties
    return rank_sum, tie_term


def _mann_whitney_u(x0, x1) -> dict[str, np.ndarray]:
    """:func:`scipy.stats.mannwhitneyu` with its default arguments for every column, ranking only the nonzero entries."""
    n0, n1 = x0.shape[0], x1.shape[0]
    n = n0 + n1
    rank_sum, tie_term = _rank_sums(*_csc_columns(x0), n0, *_csc_columns(x1), n1)
    statistic = rank_sum - n0 * (n0 + 1) / 2
    numerator = statistic - n0 * n1 / 2
    numerator -= 0.5 * np.sign(numerator)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = numerator / np.sqrt(n0 * n1 / 12 * ((n + 1) - tie_term / (n * (n - 1))))
    return {"p_value": np.clip(2 * special.ndtr(-np.abs(z)), 0, 1), "statistic": statistic}


def _run_vectorized_test(test, x0_block, x1_block, paired: bool, kwargs: dict) -> dict:
    """Test a block of variables at once, returning one array of values per metric."""
    return {
        "log_fc": np.log2(_column_means(x1_block)) - np.log2(_column_means(x0_block)),
        **{
            metric: np.atleast_1d(np.asarray(value))
            for metric, value in test(x0_block, x1_block, paired, **kwargs).items()
        },
    }


def _run_test(test, x0_var: np.ndarray, x1_var: np.ndarray, paired: bool, kwargs: dict) -> dict:
    """Test a single variable."""
    return {
        "log_fc": np.log2(np.mean(x1_var)) - np.log2(np.mean(x0_var)),
        **test(x0_var, x1_var, paired, **kwargs),
    }


def _log2_fold_change(x0: np.ndarray, x1: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log2 fold change of the means with a pseudocount, computed along `axis`."""
    return np.log2(np.mean(x1, axis=axis) + 1e-8) - np.log2(np.mean(x0, axis=axis) + 1e-8)


def fdr_correction(
    df: pd.DataFrame, pvalue_col: str = "p_value", *, key_added: str = "adj_p_value", inplace: bool = False
):
    """Adjust p-values in a DataFrame with test results using FDR correction."""
    if not inplace:
        df = df.copy()

    pvalues = df[pvalue_col].to_numpy(dtype=float)
    tested = ~np.isnan(pvalues)
    df[key_added] = np.nan
    df.loc[tested, key_added] = statsmodels.stats.multitest.fdrcorrection(pvalues[tested])[1]

    if not inplace:
        return df


class SimpleComparisonBase(MethodBase):
    #: Counterpart of :func:`_test` that tests a block of variables at once, which is orders of magnitude faster.
    #: If None, :func:`_test` is called per variable instead.
    _test_vectorized: Callable[..., dict[str, np.ndarray]] | None = None
    _test_vectorized_sparse: bool = False

    @staticmethod
    @abstractmethod
    def _test(x0: np.ndarray, x1: np.ndarray, paired: bool, **kwargs) -> dict[str, float]:
        """Perform a statistical test between values in x0 and x1.

        If `paired` is True, x0 and x1 must be of the same length and ordered such that
        paired elements have the same position.

        Args:
            x0: Array with baseline values.
            x1: Array with values to compare.
            paired: Indicates whether to perform a paired test
            **kwargs: kwargs passed to the test function

        Returns:
            A dictionary metric -> value.
            This allows to return values for different metrics (e.g. p-value + test statistic).
        """
        ...

    def _compare_single_group(
        self, baseline_idx: np.ndarray, comparison_idx: np.ndarray, *, paired: bool, n_jobs: int | None = None, **kwargs
    ) -> DataFrame:
        """Perform a single comparison between two groups.

        Args:
            baseline_idx: Numeric indices indicating which observations are in the baseline group.
            comparison_idx: Numeric indices indicating which observations are in the comparison/treatment group
            paired: Whether to perform a paired test. Note that in the case of a paired test,
                the indices must be ordered such that paired observations appear at the same position.
            n_jobs: Number of jobs to distribute the variables over, passed to `joblib.Parallel`.
            **kwargs: kwargs passed to the test function
        """
        if paired:
            assert len(baseline_idx) == len(comparison_idx), "For a paired test, indices must be of the same length"

        x0 = self.data[baseline_idx, :]
        x1 = self.data[comparison_idx, :]

        # In the following, we are doing a lot of column slicing -- which is significantly
        # more efficient in csc format.
        if issparse(self.data):
            x0 = x0.tocsc()
            x1 = x1.tocsc()

        res = (
            self._compare_blockwise(x0, x1, paired=paired, n_jobs=n_jobs, **kwargs)
            if type(self)._test_vectorized is not None
            else self._compare_per_variable(x0, x1, paired=paired, n_jobs=n_jobs, **kwargs)
        )
        res.insert(0, "variable", self.adata.var_names.to_numpy())
        return res.sort_values("p_value")

    def _compare_blockwise(self, x0, x1, *, paired: bool, n_jobs: int | None, **kwargs) -> DataFrame:
        """Test all variables with the vectorized test, in blocks of bounded size."""
        test = type(self)._test_vectorized
        blocks = _block_slices(x0.shape[1], max_block_size=_MAX_BLOCK_ELEMENTS // max(x0.shape[0] + x1.shape[0], 1))
        if not blocks:
            return pd.DataFrame()
        dense = not type(self)._test_vectorized_sparse
        # Densifying here rather than in the workers keeps the full matrices from being sent around.
        if len(blocks) == 1:
            results = [
                _run_vectorized_test(
                    test,
                    _var_block(x0, blocks[0], dense=dense),
                    _var_block(x1, blocks[0], dense=dense),
                    paired,
                    dict(kwargs),
                )
            ]
        else:
            results = list(
                _parallelize_with_joblib(
                    (
                        delayed(_run_vectorized_test)(
                            test,
                            _var_block(x0, block, dense=dense),
                            _var_block(x1, block, dense=dense),
                            paired,
                            dict(kwargs),
                        )
                        for block in blocks
                    ),
                    total=len(blocks),
                    n_jobs=n_jobs,
                )
            )
        return pd.DataFrame({metric: np.concatenate([res[metric] for res in results]) for metric in results[0]})

    def _compare_per_variable(self, x0, x1, *, paired: bool, n_jobs: int | None, **kwargs) -> DataFrame:
        """Test one variable at a time, distributing the variables over jobs."""
        test = type(self)._test
        n_vars = x0.shape[1]
        # One stream per variable keeps results independent of n_jobs.
        rng_kwarg = next((key for key in _RNG_KWARGS if key in kwargs), None)
        rngs = _spawn_rngs(kwargs[rng_kwarg], n_vars) if rng_kwarg is not None else None
        var_kwargs = [dict(kwargs)] * n_vars if rngs is None else [{**kwargs, rng_kwarg: rng} for rng in rngs]
        results = _parallelize_with_joblib(
            (
                delayed(_run_test)(
                    test,
                    _var_block(x0, slice(i, i + 1)).ravel(),
                    _var_block(x1, slice(i, i + 1)).ravel(),
                    paired,
                    var_kwargs[i],
                )
                for i in range(n_vars)
            ),
            total=n_vars,
            n_jobs=n_jobs,
        )
        return pd.DataFrame(list(results))

    @classmethod
    def compare_groups(
        cls,
        adata: AnnData,
        column: str,
        baseline: str,
        groups_to_compare: str | Sequence[str],
        *,
        paired_by: str | None = None,
        mask: str | None = None,
        layer: str | None = None,
        n_jobs: int | None = None,
        fit_kwargs: Mapping = MappingProxyType({}),
        test_kwargs: Mapping = MappingProxyType({}),
    ) -> DataFrame:
        """Perform a comparison between groups.

        Args:
            adata: Data with observations to compare.
            column: Column in `adata.obs` that contains the groups to compare.
            baseline: Reference group.
            groups_to_compare: Groups to compare against the baseline. If None, all other groups
                are compared.
            paired_by: Column in `adata.obs` to use for pairing. If None, an unpaired test is performed.
            mask: Mask to apply to the data.
            layer: Layer to use for the comparison.
            n_jobs: Number of jobs to distribute the variables over, passed to `joblib.Parallel`.
                None means one job, -1 all available cores.
                `TTest` and `WilcoxonTest` test all variables at once and only use several jobs if the data does not fit into memory in one block.
            fit_kwargs: Unused argument for compatibility with the `MethodBase` interface, do not specify.
            test_kwargs: Additional kwargs passed to the test function.
        """
        if len(fit_kwargs):
            warnings.warn("fit_kwargs not used for simple tests.", UserWarning, stacklevel=2)
        paired = paired_by is not None
        model = cls(adata, mask=mask, layer=layer)
        if groups_to_compare is None:
            # compare against all other
            groups_to_compare = sorted(set(model.adata.obs[column]) - {baseline})
        if isinstance(groups_to_compare, str):
            groups_to_compare = [groups_to_compare]

        def _get_idx(column, value):
            mask = model.adata.obs[column] == value
            if paired:
                dummies = pd.get_dummies(model.adata.obs[paired_by], sparse=True).sparse.to_coo().tocsr()
                if not np.all(np.sum(dummies, axis=0) == 2):
                    raise ValueError("Pairing is only possible with exactly two values per group")
                # Use matrix multiplication to only retreive those dummy entries that are associated with the current `value`.
                # Convert to COO matrix to get rows/cols
                # row indices refers to the indices of rows that have `column == value` (equivalent to np.where(mask)[0])
                # col indices refers to the numeric index of each "pair" in obs_names
                ind_mat = diags(mask.values, dtype=bool) @ dummies
                if not np.all(np.sum(ind_mat, axis=0) == 1):
                    raise ValueError("Pairing is only possible with exactly two values per group")
                ind_mat = ind_mat.tocoo()
                return ind_mat.row[np.argsort(ind_mat.col)]
            else:
                return np.where(mask)[0]

        res_dfs = []
        baseline_idx = _get_idx(column, baseline)
        for group_to_compare in groups_to_compare:
            comparison_idx = _get_idx(column, group_to_compare)
            res_dfs.append(
                model._compare_single_group(
                    baseline_idx, comparison_idx, paired=paired, n_jobs=n_jobs, **test_kwargs
                ).assign(comparison=f"{group_to_compare}_vs_{baseline if baseline is not None else 'rest'}")
            )
        return fdr_correction(pd.concat(res_dfs))


class WilcoxonTest(SimpleComparisonBase):
    """Perform a unpaired or paired Wilcoxon test.

    (the former is also known as "Mann-Whitney U test", the latter as "wilcoxon signed rank test")
    """

    @staticmethod
    def _test(x0: np.ndarray, x1: np.ndarray, paired: bool, **kwargs) -> dict[str, float]:
        """Perform an unpaired or paired Wilcoxon/Mann-Whitney-U test."""
        test_result = scipy.stats.wilcoxon(x0, x1, **kwargs) if paired else scipy.stats.mannwhitneyu(x0, x1, **kwargs)

        return {
            "p_value": test_result.pvalue,
            "statistic": test_result.statistic,
        }

    _test_vectorized_sparse = True

    @staticmethod
    def _test_vectorized(x0, x1, paired: bool, **kwargs) -> dict[str, np.ndarray]:
        """Perform an unpaired or paired Wilcoxon/Mann-Whitney-U test for each column of x0 and x1.

        Without arguments, the unpaired test uses the normal approximation with tie and continuity correction for samples larger than 8, which only needs to rank the nonzero values.
        """
        if not (paired or kwargs) and min(x0.shape[0], x1.shape[0]) > 8:
            return _mann_whitney_u(x0, x1)
        x0, x1 = to_dense(x0), to_dense(x1)
        test_result = (
            scipy.stats.wilcoxon(x0, x1, axis=0, **kwargs)
            if paired
            else scipy.stats.mannwhitneyu(x0, x1, axis=0, **kwargs)
        )

        return {
            "p_value": test_result.pvalue,
            "statistic": test_result.statistic,
        }


class TTest(SimpleComparisonBase):
    """Perform a unpaired or paired T-test."""

    @staticmethod
    def _test(x0: np.ndarray, x1: np.ndarray, paired: bool, **kwargs) -> dict[str, float]:
        test_result = scipy.stats.ttest_rel(x0, x1, **kwargs) if paired else scipy.stats.ttest_ind(x0, x1, **kwargs)

        return {
            "p_value": test_result.pvalue,
            "statistic": test_result.statistic,
        }

    _test_vectorized_sparse = True

    @staticmethod
    def _test_vectorized(x0, x1, paired: bool, **kwargs) -> dict[str, np.ndarray]:
        """Perform an unpaired or paired T-test for each column of x0 and x1."""
        if not (paired or kwargs):
            return _t_test_ind(x0, x1)
        x0, x1 = to_dense(x0), to_dense(x1)
        test_result = (
            scipy.stats.ttest_rel(x0, x1, axis=0, **kwargs)
            if paired
            else scipy.stats.ttest_ind(x0, x1, axis=0, **kwargs)
        )

        return {
            "p_value": test_result.pvalue,
            "statistic": test_result.statistic,
        }


class PermutationTest(SimpleComparisonBase):
    """Perform a permutation test.

    The permutation test relies on another test statistic (e.g. t-statistic or your own) to obtain a p-value through
    random permutations of the data and repeated generation of the test statistic.

    For paired tests, each paired observation is permuted together and distributed randomly between the two groups.
    For unpaired tests, all observations are permuted independently.

    The null hypothesis for the unpaired test is that all observations come from the same underlying distribution and
    have been randomly assigned to one of the samples.

    The null hypothesis for the paired permutation test is that the observations within each pair are drawn from the
    same underlying distribution and that their assignment to a sample is random.
    """

    @classmethod
    def compare_groups(
        cls,
        adata: AnnData,
        column: str,
        baseline: str,
        groups_to_compare: str | Sequence[str],
        *,
        paired_by: str | None = None,
        mask: str | None = None,
        layer: str | None = None,
        n_jobs: int | None = None,
        n_permutations: int = 1000,
        test_statistic: TestStatistic = _log2_fold_change,
        fit_kwargs: Mapping = MappingProxyType({}),
        test_kwargs: Mapping = MappingProxyType({}),
    ) -> DataFrame:
        """Perform a permutation test comparison between groups.

        Args:
            adata: Data with observations to compare.
            column: Column in `adata.obs` that contains the groups to compare.
            baseline: Reference group.
            groups_to_compare: Groups to compare against the baseline.
                If None, all other groups are compared.
            paired_by: Column in `adata.obs` to use for pairing.
                If None, an unpaired test is performed.
            mask: Mask to apply to the data.
            layer: Layer to use for the comparison.
            n_jobs: Number of jobs to distribute the variables over, passed to `joblib.Parallel`.
                None means one job, -1 all available cores.
                Permuting is expensive and each variable is permuted independently, so this is the method that benefits the most from several jobs.
            n_permutations: Number of permutations to perform.
            test_statistic: A callable that takes two arrays (x0, x1) and returns a float statistic.
                Defaults to log2 fold change with pseudocount: log2(mean(x1) + 1e-8) - log2(mean(x0) + 1e-8).
                The callable should have signature: test_statistic(x0, x1) -> float.
            fit_kwargs: Unused argument for compatibility with the `MethodBase` interface, do not specify.
            test_kwargs: Additional kwargs passed to the permutation test function (not the test statistic).
                The permutation test function is `scipy.stats.permutation_test`.
                We refer to its documentation for available options.
                Note that `test_statistic` and `n_permutations` are set by this function and should not be provided here.
                Passing `rng` makes the result reproducible and independent of `n_jobs`, since each variable is tested with its own stream derived from it.

        Examples:
            >>> # Difference in means (log fold change)
            >>> PermutationTest.compare_groups(
            ...     adata,
            ...     column="condition",
            ...     baseline="A",
            ...     groups_to_compare="B",
            ...     test_statistic=lambda x, y: np.log2(np.mean(y)) - np.log2(np.mean(x)),
            ...     n_permutations=1000,
            ...     test_kwargs={"rng": 0},
            ... )
        """
        enhanced_test_kwargs = dict(test_kwargs)
        enhanced_test_kwargs.update({"test_statistic": test_statistic, "n_permutations": n_permutations})

        return super().compare_groups(
            adata=adata,
            column=column,
            baseline=baseline,
            groups_to_compare=groups_to_compare,
            paired_by=paired_by,
            mask=mask,
            layer=layer,
            n_jobs=n_jobs,
            fit_kwargs=fit_kwargs,
            test_kwargs=enhanced_test_kwargs,
        )

    @staticmethod
    def _test(
        x0: np.ndarray,
        x1: np.ndarray,
        paired: bool,
        test_statistic: TestStatistic = _log2_fold_change,
        n_permutations: int = 1000,
        **kwargs,
    ) -> dict[str, float]:
        """Perform a permutation test.

        This function uses a simple test statistic function to compute p-values through permutations.

        Args:
            x0: Array with baseline values.
            x1: Array with values to compare.
            paired: Whether to perform a paired test.
            test_statistic: A callable that takes two arrays (x0, x1) and returns a float statistic.
                Please refer to the examples below for usage.
                The callable should have signature: test_statistic(x0, x1) -> float.
            n_permutations: Number of permutations to perform.
            **kwargs: Additional kwargs passed to scipy.stats.permutation_test.

        Examples:
            >>> # Difference in means (log fold change)
            >>> PermutationTest._test(x0, x1, paired=False)
            >>>
            >>> # Difference in medians
            >>> median_diff = lambda x, y: np.median(y) - np.median(x)
            >>> PermutationTest._test(x0, x1, paired=False, test_statistic=median_diff)
        """
        test_result = scipy.stats.permutation_test(
            [x0, x1],
            statistic=test_statistic,
            n_resamples=n_permutations,
            permutation_type=("samples" if paired else "independent"),
            **kwargs,
        )

        return {
            "p_value": test_result.pvalue,
            "statistic": test_result.statistic,
        }
