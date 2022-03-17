from typing import Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def important_features(results: dict, top_n=10, ax: Axes = None, return_figure: bool = False) -> Union[Figure, Axes]:
    """Plot a lollipop plot of the n features with largest feature importances.

    Args:
        results: results after running `predict()`
        top_n: n number feature importance values to plot. Default is 10.
        ax: optionally, axes used to draw plot
        return_figure: if `True` returns figure of the plot, default is `False`

    Returns:
        Axes of the plot.
    """
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
