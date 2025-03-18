from ._base import LinearModelBase, MethodBase
from ._dge_comparison import DGEEVAL
from ._edger import EdgeR
from ._pydeseq2 import PyDESeq2
from ._simple_tests import PermutationTest, SimpleComparisonBase, TTest, WilcoxonTest
from ._statsmodels import Statsmodels

__all__ = [
    "MethodBase",
    "LinearModelBase",
    "EdgeR",
    "PyDESeq2",
    "Statsmodels",
    "SimpleComparisonBase",
    "WilcoxonTest",
    "TTest",
    "PermutationTest",
]

AVAILABLE_METHODS = [Statsmodels, EdgeR, PyDESeq2, WilcoxonTest, TTest, PermutationTest]
