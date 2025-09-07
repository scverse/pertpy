from importlib import import_module

from pertpy.tools._augur import Augur
from pertpy.tools._cinemaot import Cinemaot
from pertpy.tools._coda._sccoda import Sccoda
from pertpy.tools._dialogue import Dialogue
from pertpy.tools._distances._distance_tests import DistanceTest
from pertpy.tools._distances._distances import Distance
from pertpy.tools._enrichment import Enrichment
from pertpy.tools._milo import Milo
from pertpy.tools._mixscape import Mixscape
from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._comparison import PerturbationComparison
from pertpy.tools._perturbation_space._discriminator_classifiers import (
    LRClassifierSpace,
    MLPClassifierSpace,
)
from pertpy.tools._perturbation_space._simple import (
    CentroidSpace,
    DBSCANSpace,
    KMeansSpace,
    PseudobulkSpace,
)


def __getattr__(name: str):
    if name == "Tasccoda":
        try:
            for extra in ["toytree", "ete4"]:
                import_module(extra)
            module = import_module("pertpy.tools._coda._tasccoda")
            return module.Tasccoda
        except ImportError:
            raise ImportError(
                "Extra dependencies required: toytree, ete4. Please install with: pip install toytree ete4"
            ) from None
    elif name in ["EdgeR", "PyDESeq2", "Statsmodels", "TTest", "WilcoxonTest"]:
        module = import_module("pertpy.tools._differential_gene_expression")
        return getattr(module, name)
    elif name == "Scgen":
        try:
            module = import_module("pertpy.tools._scgen")
            return module.Scgen
        except ImportError:
            raise ImportError(
                "Scgen requires scvi-tools to be installed. Please install with: pip install scvi-tools"
            ) from None

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


__all__ = [
    "Augur",
    "Cinemaot",
    "Sccoda",
    "Tasccoda",
    "Dialogue",
    "EdgeR",
    "PyDESeq2",
    "WilcoxonTest",
    "TTest",
    "Statsmodels",
    "DistanceTest",
    "Distance",
    "Enrichment",
    "Milo",
    "Mixscape",
    "ClusteringSpace",
    "PerturbationComparison",
    "LRClassifierSpace",
    "MLPClassifierSpace",
    "CentroidSpace",
    "DBSCANSpace",
    "KMeansSpace",
    "PseudobulkSpace",
    "Scgen",
]
