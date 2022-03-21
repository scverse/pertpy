from typing import Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def scatterplot(results1, results2, top_n=None, ax: Axes = None, return_figure: bool = False) -> Union[Figure, Axes]:
    """Create scatterplot with two augur results.

    Args:
        results1: results after running `predict()`
        results2: results after running `predict()`
        top_n: optionally, the number of top prioritized cell types to label in the plot
        ax: optionally, axes used to draw plot
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
