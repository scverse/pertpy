from __future__ import annotations

from typing import Any

import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class AugurpyPlot:
    """Plotting functions for Augurpy."""

    @staticmethod
    def dp_scatter(results: pd.DataFrame, top_n=None, ax: Axes = None, return_figure: bool = False) -> Figure | Axes:
        """Plot result of differential prioritization.

        Args:
            results: Results after running differential prioritization.
            top_n: optionally, the number of top prioritized cell types to label in the plot
            ax: optionally, axes used to draw plot
            return_figure: if `True` returns figure of the plot

        Returns:
        """
        x = results["mean_augur_score1"]
        y = results["mean_augur_score2"]

        if ax is None:
            fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=results.z, cmap="Greens")

        # adding optional labels
        top_n_index = results.sort_values(by="pval").index[:top_n]
        for idx in top_n_index:
            ax.annotate(
                results.loc[idx, "cell_type"],
                (results.loc[idx, "mean_augur_score1"], results.loc[idx, "mean_augur_score2"]),
            )

        # add diagonal
        limits = max(ax.get_xlim(), ax.get_ylim())
        (diag_line,) = ax.plot(limits, limits, ls="--", c=".3")

        # formatting and details
        plt.xlabel("Augur scores 1")
        plt.ylabel("Augur scores 2")
        legend1 = ax.legend(*scatter.legend_elements(), loc="center left", title="z-scores", bbox_to_anchor=(1, 0.5))
        ax.add_artist(legend1)

        return fig if return_figure else ax

    @staticmethod
    def important_features(
        data: dict[str, Any], key: str = "augurpy_results", top_n=10, ax: Axes = None, return_figure: bool = False
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
        """
        if isinstance(data, AnnData):
            results = data.uns[key]
        else:
            results = data
        # top_n features to plot
        n_features = (
            results["feature_importances"]
            .groupby("genes", as_index=False)
            .feature_importances.mean()
            .sort_values(by="feature_importances")[-top_n:]
        )

        if ax is None:
            fig, ax = plt.subplots()
        y_axes_range = range(1, top_n + 1)
        ax.hlines(
            y_axes_range,
            xmin=0,
            xmax=n_features["feature_importances"],
        )

        # drawing the markers (circle)
        ax.plot(n_features["feature_importances"], y_axes_range, "o")

        # formatting and details
        plt.xlabel("Mean Feature Importance")
        plt.ylabel("Gene")
        plt.yticks(y_axes_range, n_features["genes"])

        return fig if return_figure else ax

    @staticmethod
    def lollipop(
        data: dict[str, Any], key: str = "augurpy_results", ax: Axes = None, return_figure: bool = False
    ) -> Figure | Axes:
        """Plot a lollipop plot of the mean augur values.

        Args:
            results: results after running `predict()` as dictionary or the AnnData object.
            key: Key in the AnnData object of the results
            ax: optionally, axes used to draw plot
            return_figure: if `True` returns figure of the plot

        Returns:
            Axes of the plot.
        """
        if isinstance(data, AnnData):
            results = data.uns[key]
        else:
            results = data
        if ax is None:
            fig, ax = plt.subplots()
        y_axes_range = range(1, len(results["summary_metrics"].columns) + 1)
        ax.hlines(
            y_axes_range,
            xmin=0,
            xmax=results["summary_metrics"].sort_values("mean_augur_score", axis=1).loc["mean_augur_score"],
        )

        # drawing the markers (circle)
        ax.plot(
            results["summary_metrics"].sort_values("mean_augur_score", axis=1).loc["mean_augur_score"],
            y_axes_range,
            "o",
        )

        # formatting and details
        plt.xlabel("Mean Augur Score")
        plt.ylabel("Cell Type")
        plt.yticks(y_axes_range, results["summary_metrics"].sort_values("mean_augur_score", axis=1).columns)

        return fig if return_figure else ax

    @staticmethod
    def scatterplot(
        results1: dict[str, Any], results2: dict[str, Any], top_n=None, return_figure: bool = False
    ) -> Figure | Axes:
        """Create scatterplot with two augur results.

        Args:
            results1: results after running `predict()`
            results2: results after running `predict()`
            top_n: optionally, the number of top prioritized cell types to label in the plot
            return_figure: if `True` returns figure of the plot

        Returns:
            Axes of the plot.
        """
        cell_types = results1["summary_metrics"].columns

        fig, ax = plt.subplots()
        ax.scatter(
            results1["summary_metrics"].loc["mean_augur_score", cell_types],
            results2["summary_metrics"].loc["mean_augur_score", cell_types],
        )

        # adding optional labels
        top_n_cell_types = (
            (results1["summary_metrics"].loc["mean_augur_score"] - results2["summary_metrics"].loc["mean_augur_score"])
            .sort_values(ascending=False)
            .index[:top_n]
        )
        for txt in top_n_cell_types:
            ax.annotate(
                txt,
                (
                    results1["summary_metrics"].loc["mean_augur_score", txt],
                    results2["summary_metrics"].loc["mean_augur_score", txt],
                ),
            )

        # adding diagonal
        limits = max(ax.get_xlim(), ax.get_ylim())
        (diag_line,) = ax.plot(limits, limits, ls="--", c=".3")

        # formatting and details
        plt.xlabel("Augur scores 1")
        plt.ylabel("Augur scores 2")

        return fig if return_figure else ax
