from importlib import import_module


def lazy_import(module_path: str, class_name: str, extras: list[str]):
    try:
        for extra in extras:
            import_module(extra)
        module = import_module(module_path)
        return getattr(module, class_name)
    except ImportError:

        class Placeholder:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    f"Extra dependencies required: {', '.join(extras)}. "
                    f"Please install with: pip install {' '.join(extras)}"
                )

        return Placeholder


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
from pertpy.tools._scgen import Scgen

CODA_EXTRAS = ["toytree", "ete4"]  # also "pyqt6" but it cannot be imported
Tasccoda = lazy_import("pertpy.tools._coda._tasccoda", "Tasccoda", CODA_EXTRAS)

DE_EXTRAS = ["formulaic", "pydeseq2"]
EdgeR = lazy_import("pertpy.tools._differential_gene_expression", "EdgeR", DE_EXTRAS)  # edgeR will be imported via rpy2
PyDESeq2 = lazy_import("pertpy.tools._differential_gene_expression", "PyDESeq2", DE_EXTRAS)
Statsmodels = lazy_import("pertpy.tools._differential_gene_expression", "Statsmodels", DE_EXTRAS + ["statsmodels"])
TTest = lazy_import("pertpy.tools._differential_gene_expression", "TTest", DE_EXTRAS)
WilcoxonTest = lazy_import("pertpy.tools._differential_gene_expression", "WilcoxonTest", DE_EXTRAS)

from ._milo import group_nhoods

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
    "group_nhoods",
]
