from typing import Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def lollipop(results: dict, ax: Axes = None, return_figure: bool = False) -> Union[Figure, Axes]:
    """Plot a lollipop plot of the mean augur values.

    Args:
        results: results after running `predict()`
        ax: optionally, axes used to draw plot
        return_figure: if `True` returns figure of the plot

    Returns:
        Axes of the plot.
    """
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
        results["summary_metrics"].sort_values("mean_augur_score", axis=1).loc["mean_augur_score"], y_axes_range, "o"
    )

    # formatting and details
    plt.xlabel("Mean Augur Score")
    plt.ylabel("Cell Type")
    plt.yticks(y_axes_range, results["summary_metrics"].sort_values("mean_augur_score", axis=1).columns)

    return fig if return_figure else ax
