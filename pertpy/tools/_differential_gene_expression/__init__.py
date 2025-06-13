from ._base import LinearModelBase, MethodBase
from ._dge_comparison import DGEEVAL
from ._edger import EdgeR
from ._simple_tests import SimpleComparisonBase, TTest, WilcoxonTest
from ._statsmodels import Statsmodels


def __getattr__(name):
    if name == "PyDESeq2":
        from importlib.util import find_spec

        if find_spec("pydeseq2") is None:
            raise ImportError("pydeseq2 is required but not installed")
        from ._pydeseq2 import PyDESeq2

        return PyDESeq2
    elif name == "EdgeR":
        from importlib.util import find_spec

        if find_spec("rpy2") is None:
            raise ImportError("rpy2 is required but not installed")
        from ._edger import EdgeR

        return EdgeR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_available_methods():
    methods = [Statsmodels, WilcoxonTest, TTest]
    from importlib.util import find_spec

    if find_spec("pydeseq2") is not None:
        methods.append(__getattr__("PyDESeq2"))
    if find_spec("rpy2") is not None:
        methods.append(__getattr__("EdgeR"))
    return methods


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
