import contextlib
from importlib import import_module
from importlib.util import find_spec

from ._base import LinearModelBase, MethodBase
from ._dge_comparison import DGEEVAL
from ._edger import EdgeR
from ._simple_tests import SimpleComparisonBase, TTest, WilcoxonTest


def __getattr__(name: str):
    deps = {
        "PyDESeq2": ["pydeseq2", "formulaic_contrasts", "formulaic"],
        "EdgeR": ["rpy2", "formulaic_contrasts", "formulaic"],
        "Statsmodels": ["formulaic_contrasts", "formulaic"],
    }

    if name in deps:
        for dep in deps[name]:
            if find_spec(dep) is None:
                raise ImportError(f"{dep} is required but not installed")

        module_map = {
            "PyDESeq2": "pertpy.tools._differential_gene_expression._pydeseq2",
            "EdgeR": "pertpy.tools._differential_gene_expression._edger",
            "Statsmodels": "pertpy.tools._differential_gene_expression._statsmodels",
        }

        module = import_module(module_map[name])
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_available_methods():
    methods = [WilcoxonTest, TTest]
    from importlib.util import find_spec

    for name in ["Statsmodels", "PyDESeq2", "EdgeR"]:
        with contextlib.suppress(ImportError):
            methods.append(__getattr__(name))

    return methods


AVAILABLE_METHODS = _get_available_methods()


AVAILABLE_METHODS = _get_available_methods()

__all__ = [
    "MethodBase",
    "LinearModelBase",
    "EdgeR",
    "PyDESeq2",
    "Statsmodels",
    "SimpleComparisonBase",
    "WilcoxonTest",
    "TTest",
]
