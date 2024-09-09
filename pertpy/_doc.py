from textwrap import dedent


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
