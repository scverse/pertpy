from typing import Union

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_differential_prioritization(
    results, top_n=None, ax: Axes = None, return_figure: bool = False
) -> Union[Figure, Axes]:
    """Plot result of differential prioritization.

    Args:
        results: results after running differential prioritization
        top_n: optionally, the number of top prioritized cell types to label in the plot
        ax: optionally, axes used to draw plot
        return_figure: if `True` returns figure of the plot

    Returns:
        Axes of the plot.
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
