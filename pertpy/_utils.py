from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure


def savefig_or_show(
    writekey: str,
    show: bool,
    save: bool | str,
    return_fig: bool = False,
    dpi: int = 150,
    ext: str = "png",
) -> Figure | None:
    if isinstance(save, str):
        # check whether `save` contains a figure extension
        for try_ext in [".svg", ".pdf", ".png"]:
            if save.endswith(try_ext):
                ext = try_ext[1:]
                save = save.replace(try_ext, "")
                break
        # append extension
        writekey += f"_{save}"
        save = True

    if save:
        Path.mkdir(Path("figures"), exist_ok=True)
        plt.savefig(f"figures/{writekey}.{ext}", dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if save:
        plt.close()  # clear figure
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
            save: if `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`.pdf`, `.png`, `.svg`}.
            return_fig: if `True`, returns figure of the plot.\
"""
