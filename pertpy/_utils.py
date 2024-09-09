from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure


def return_fig_or_show(
    show: bool,
    return_fig: bool = False,
) -> Figure | None:
    plt.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return plt.gcf()
    return None


def _doc_params(**kwds):  # pragma: no cover
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__.format_map(kwds))
        return obj

    return dec


doc_common_plot_args = """\
show: if `True`, shows the plot.
            return_fig: if `True`, returns figure of the plot.\
"""
