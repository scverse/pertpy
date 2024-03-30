from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from anndata import AnnData
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class AugurpyPlot:
    """Plotting functions for Augurpy."""

    @staticmethod
    def dp_scatter(results: pd.DataFrame, top_n=None, ax: Axes = None) -> Figure | Axes:
        """Plot result of differential prioritization.

        Args:
            results: Results after running differential prioritization.
            top_n: optionally, the number of top prioritized cell types to label in the plot
            ax: optionally, axes used to draw plot
            return_figure: if `True` returns figure of the plot

        Returns:
            Axes of the plot.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")

            >>> data_15 = ag_rfc.load(adata, condition_label="Maintenance_Cocaine", treatment_label="withdraw_15d_Cocaine")
            >>> adata_15, results_15 = ag_rfc.predict(data_15, random_state=None, n_threads=4)
            >>> adata_15_permute, results_15_permute = ag_rfc.predict(data_15, augur_mode="permute", n_subsamples=100, random_state=None, n_threads=4)

            >>> data_48 = ag_rfc.load(adata, condition_label="Maintenance_Cocaine", treatment_label="withdraw_48h_Cocaine")
            >>> adata_48, results_48 = ag_rfc.predict(data_48, random_state=None, n_threads=4)
            >>> adata_48_permute, results_48_permute = ag_rfc.predict(data_48, augur_mode="permute", n_subsamples=100, random_state=None, n_threads=4)

            >>> pvals = ag_rfc.predict_differential_prioritization(augur_results1=results_15, augur_results2=results_48, \
                permuted_results1=results_15_permute, permuted_results2=results_48_permute)
            >>> ag_rfc.plot_dp_scatter(pvals)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Augur

        ag = Augur("random_forest_classifier")

        return ag.plot_dp_scatter(results=results, top_n=top_n, ax=ax)

    @staticmethod
    def important_features(
        data: dict[str, Any], key: str = "augurpy_results", top_n=10, ax: Axes = None
    ) -> Figure | Axes:
        """Plot a lollipop plot of the n features with largest feature importances.

        Args:
            results: results after running `predict()` as dictionary or the AnnData object.
            key: Key in the AnnData object of the results
            top_n: n number feature importance values to plot. Default is 10.
            ax: optionally, axes used to draw plot
            return_figure: if `True` returns figure of the plot, default is `False`

        Returns:
            Axes of the plot.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> v_adata, v_results = ag_rfc.predict(
            ...     loaded_data, subsample_size=20, select_variance_features=True, n_threads=4
            ... )
            >>> ag_rfc.plot_important_features(v_results)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Augur

        ag = Augur("random_forest_classifier")

        return ag.plot_important_features(data=data, key=key, top_n=top_n, ax=ax)

    @staticmethod
    def lollipop(data: dict[str, Any], key: str = "augurpy_results", ax: Axes = None) -> Figure | Axes | None:
        """Plot a lollipop plot of the mean augur values.

        Args:
            results: results after running `predict()` as dictionary or the AnnData object.
            key: Key in the AnnData object of the results
            ax: optionally, axes used to draw plot
            return_figure: if `True` returns figure of the plot

        Returns:
            Axes of the plot.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> v_adata, v_results = ag_rfc.predict(
            ...     loaded_data, subsample_size=20, select_variance_features=True, n_threads=4
            ... )
            >>> ag_rfc.plot_lollipop(v_results)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Augur

        ag = Augur("random_forest_classifier")

        return ag.plot_lollipop(data=data, key=key, ax=ax)

    @staticmethod
    def scatterplot(results1: dict[str, Any], results2: dict[str, Any], top_n=None) -> Figure | Axes:
        """Create scatterplot with two augur results.

        Args:
            results1: results after running `predict()`
            results2: results after running `predict()`
            top_n: optionally, the number of top prioritized cell types to label in the plot
            return_figure: if `True` returns figure of the plot

        Returns:
            Axes of the plot.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> h_adata, h_results = ag_rfc.predict(loaded_data, subsample_size=20, n_threads=4)
            >>> v_adata, v_results = ag_rfc.predict(
            ...     loaded_data, subsample_size=20, select_variance_features=True, n_threads=4
            ... )
            >>> ag_rfc.plot_scatterplot(v_results, h_results)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Augur

        ag = Augur("random_forest_classifier")

        return ag.plot_scatterplot(results1=results1, results2=results2, top_n=top_n)
