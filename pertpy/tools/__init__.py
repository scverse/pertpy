from importlib import import_module


def lazy_import(import_path, extras):
    def _import():
        try:
            for extra in extras:
                import_module(extra)
        except ImportError as e:
            raise ImportError(
                f"Extra dependencies required: {', '.join(extras)}. "
                f"Please install with: pip install {' '.join(extras)}"
            ) from e

        module = import_module(import_path)
        return getattr(module, import_path.split(".")[-1])

    return _import


from pertpy.tools._augur import Augur
from pertpy.tools._cinemaot import Cinemaot
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

CODA_EXTRAS = ["toytree", "arviz", "ete3"]  # TODO also pyqt5 but can't be checked for as import
Sccoda = lazy_import("pertpy.tools._coda._sccoda", CODA_EXTRAS)
Tasccoda = lazy_import("pertpy.tools._coda._tasccoda", CODA_EXTRAS)

DE_EXTRAS = ["formulaic", "pydeseq2"]
EdgeR = lazy_import("pertpy.tools._differential_gene_expression.edger", DE_EXTRAS + ["edger"])
PyDESeq2 = lazy_import("pertpy.tools._differential_gene_expression.pydeseq2", DE_EXTRAS)
Statsmodels = lazy_import("pertpy.tools._differential_gene_expression.statsmodels", DE_EXTRAS + ["statsmodels"])
DGEEVAL = lazy_import("pertpy.tools._differential_gene_expression.dgeeval", DE_EXTRAS)
TTest = lazy_import("pertpy.tools._differential_gene_expression.ttest", DE_EXTRAS)
WilcoxonTest = lazy_import("pertpy.tools._differential_gene_expression.wilcoxon", DE_EXTRAS)

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
    "LRClassifierSpace",
    "MLPClassifierSpace",
    "CentroidSpace",
    "DBSCANSpace",
    "KMeansSpace",
    "PseudobulkSpace",
    "Scgen",
    "DGEEVAL",
]
