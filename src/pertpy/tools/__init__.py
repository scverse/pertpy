from importlib import import_module

from pertpy._jax import jax_import
from pertpy.tools._augur import Augur
from pertpy.tools._dialogue import Dialogue
from pertpy.tools._distances._distance_tests import DistanceTest
from pertpy.tools._distances._distances import Distance
from pertpy.tools._enrichment import Enrichment
from pertpy.tools._milo import Milo
from pertpy.tools._perturbation_efficacy._mixscale import Mixscale
from pertpy.tools._perturbation_efficacy._mixscape import Mixscape
from pertpy.tools._perturbation_evaluator import PerturbationEvaluator
from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._comparison import PerturbationComparison
from pertpy.tools._perturbation_space._discriminator_classifiers import LRClassifierSpace
from pertpy.tools._perturbation_space._simple import (
    CentroidSpace,
    DistanceSpace,
    EmbeddingSpace,
    HDBSCANSpace,
    KMeansSpace,
    PseudobulkSpace,
)

_JAX_BACKED = {
    "Cinemaot": ("pertpy.tools._cinemaot", "Cinemaot"),
    "Sccoda": ("pertpy.tools._coda._sccoda", "Sccoda"),
    "MLPClassifierSpace": ("pertpy.tools._perturbation_space._mlp_classifier", "MLPClassifierSpace"),
    "Scgen": ("pertpy.tools._scgen", "Scgen"),
}


def __getattr__(name: str):
    if name in _JAX_BACKED:
        module_path, attr = _JAX_BACKED[name]
        with jax_import(name):
            return getattr(import_module(module_path), attr)
    elif name == "Tasccoda":
        try:
            for extra in ["toytree", "ete4"]:
                import_module(extra)
        except ImportError:
            raise ImportError(
                "Extra dependencies required: toytree, ete4. Please install with: pip install toytree ete4"
            ) from None
        with jax_import(name):
            return import_module("pertpy.tools._coda._tasccoda").Tasccoda
    elif name in ["EdgeR", "PermutationTest", "PyDESeq2", "Statsmodels", "TTest", "WilcoxonTest"]:
        module = import_module("pertpy.tools._differential_gene_expression")
        return getattr(module, name)

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
    "PermutationTest",
    "Statsmodels",
    "DistanceTest",
    "Distance",
    "Enrichment",
    "Milo",
    "Mixscape",
    "Mixscale",
    "ClusteringSpace",
    "PerturbationComparison",
    "PerturbationEvaluator",
    "LRClassifierSpace",
    "MLPClassifierSpace",
    "CentroidSpace",
    "DistanceSpace",
    "EmbeddingSpace",
    "HDBSCANSpace",
    "KMeansSpace",
    "PseudobulkSpace",
    "Scgen",
]
