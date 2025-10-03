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
from pandas.core.api import DataFrame
from scipy.sparse import diags, issparse
from tqdm.auto import tqdm

from ._base import MethodBase


def fdr_correction(
    df: pd.DataFrame, pvalue_col: str = "p_value", *, key_added: str = "adj_p_value", inplace: bool = False
):
    """Adjust p-values in a DataFrame with test results using FDR correction."""
    if not inplace:
        df = df.copy()

    df[key_added] = statsmodels.stats.multitest.fdrcorrection(df[pvalue_col].values)[1]

    if not inplace:
        return df


class SimpleComparisonBase(MethodBase):
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
        self, baseline_idx: np.ndarray, comparison_idx: np.ndarray, *, paired: bool, **kwargs
    ) -> DataFrame:
        """Perform a single comparison between two groups.

        Args:
            baseline_idx: Numeric indices indicating which observations are in the baseline group.
            comparison_idx: Numeric indices indicating which observations are in the comparison/treatment group
            paired: Whether to perform a paired test. Note that in the case of a paired test,
                the indices must be ordered such that paired observations appear at the same position.
            **kwargs: kwargs passed to the test function
        """
        if paired:
            assert len(baseline_idx) == len(comparison_idx), "For a paired test, indices must be of the same length"

        x0 = self.data[baseline_idx, :]
        x1 = self.data[comparison_idx, :]

        # In the following loop, we are doing a lot of column slicing -- which is significantly
        # more efficient in csc format.
        if issparse(self.data):
            x0 = x0.tocsc()
            x1 = x1.tocsc()

        res: list[dict[str, float]] = []
        for var in tqdm(self.adata.var_names):
            tmp_x0 = x0[:, self.adata.var_names == var]
            tmp_x0 = np.asarray(tmp_x0.todense()).flatten() if issparse(tmp_x0) else tmp_x0.flatten()
            tmp_x1 = x1[:, self.adata.var_names == var]
            tmp_x1 = np.asarray(tmp_x1.todense()).flatten() if issparse(tmp_x1) else tmp_x1.flatten()
            test_result = self._test(tmp_x0, tmp_x1, paired, **kwargs)
            mean_x0 = np.mean(tmp_x0)
            mean_x1 = np.mean(tmp_x1)
            res.append({"variable": var, "log_fc": np.log2(mean_x1) - np.log2(mean_x0), **test_result})
        return pd.DataFrame(res).sort_values("p_value")

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
                model._compare_single_group(baseline_idx, comparison_idx, paired=paired, **test_kwargs).assign(
                    comparison=f"{group_to_compare}_vs_{baseline if baseline is not None else 'rest'}"
                )
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


class TTest(SimpleComparisonBase):
    """Perform a unpaired or paired T-test."""

    @staticmethod
    def _test(x0: np.ndarray, x1: np.ndarray, paired: bool, **kwargs) -> dict[str, float]:
        test_result = scipy.stats.ttest_rel(x0, x1, **kwargs) if paired else scipy.stats.ttest_ind(x0, x1, **kwargs)

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
        n_permutations: int = 1000,
        test_statistic: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.log2(np.mean(y) + 1e-8)
        - np.log2(np.mean(x) + 1e-8),
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
            n_permutations: Number of permutations to perform.
            test_statistic: A callable that takes two arrays (x0, x1) and returns a float statistic.
                Defaults to log2 fold change with pseudocount: log2(mean(x1) + 1e-8) - log2(mean(x0) + 1e-8).
                The callable should have signature: test_statistic(x0, x1) -> float.
            fit_kwargs: Unused argument for compatibility with the `MethodBase` interface, do not specify.
            test_kwargs: Additional kwargs passed to the permutation test function (not the test statistic).
                The permutation test function is `scipy.stats.permutation_test`.
                We refer to its documentation for available options.
                Note that `test_statistic` and `n_permutations` are set by this function and should not be provided here.

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
            fit_kwargs=fit_kwargs,
            test_kwargs=enhanced_test_kwargs,
        )

    @staticmethod
    def _test(
        x0: np.ndarray,
        x1: np.ndarray,
        paired: bool,
        test_statistic: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.log2(np.mean(y) + 1e-8)
        - np.log2(np.mean(x) + 1e-8),
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
            statistic=lambda x0_perm, x1_perm: test_statistic(x0_perm, x1_perm),
            n_resamples=n_permutations,
            permutation_type=("samples" if paired else "independent"),
            **kwargs,
        )

        return {
            "p_value": test_result.pvalue,
            "statistic": test_result.statistic,
        }
