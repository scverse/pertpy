import subprocess
import sys

import pytest

import pertpy as pt
from pertpy._jax import jax_import

JAX_STACK = ("jax", "jaxlib", "numpyro", "ott", "flax", "optax")

JAX_BACKED_TOOLS = ("Sccoda", "Tasccoda", "Cinemaot", "MLPClassifierSpace", "Scgen")

JAX_FREE_TOOLS = (
    "Augur",
    "Dialogue",
    "Distance",
    "DistanceTest",
    "Enrichment",
    "LRClassifierSpace",
    "Milo",
    "Mixscape",
    "Mixscale",
    "PseudobulkSpace",
)


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_import_pertpy_does_not_import_jax():
    imported = _run(f"import sys, pertpy; print([m for m in {JAX_STACK} if m in sys.modules])")
    assert imported == "[]"


def test_import_pertpy_does_not_enable_x64():
    """``jax_enable_x64`` is process global, so importing pertpy must not change it for other libraries."""
    assert _run("import pertpy, jax; print(jax.config.jax_enable_x64)") == "False"


@pytest.mark.parametrize("name", JAX_FREE_TOOLS)
def test_jax_free_tools_do_not_import_jax(name):
    imported = _run(f"import sys, pertpy as pt; pt.tl.{name}; print([m for m in {JAX_STACK} if m in sys.modules])")
    assert imported == "[]"


def test_guide_assignment_does_not_import_jax():
    imported = _run(
        f"import sys, pertpy as pt; pt.pp.GuideAssignment().assign_by_threshold; "
        f"print([m for m in {JAX_STACK} if m in sys.modules])"
    )
    assert imported == "[]"


@pytest.mark.parametrize("name", JAX_BACKED_TOOLS)
def test_jax_backed_tools_are_accessible(name):
    assert isinstance(getattr(pt.tl, name), type)


def test_all_exports_are_resolvable():
    for name in pt.tl.__all__:
        assert getattr(pt.tl, name) is not None


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'DoesNotExist'"):
        _ = pt.tl.DoesNotExist


_BLOCK_JAX = f"""
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {JAX_STACK}:
            raise ModuleNotFoundError(f"No module named {{root!r}}", name=root)
        return None

sys.meta_path.insert(0, Blocker())
"""


def _run_without_jax(code: str) -> str:
    """Run ``code`` in a subprocess where the whole JAX stack is unimportable."""
    return _run(_BLOCK_JAX + code)


def test_pertpy_is_importable_without_jax():
    assert _run_without_jax("import pertpy; print('ok')") == "ok"


@pytest.mark.parametrize("name", JAX_BACKED_TOOLS)
def test_jax_backed_tools_report_the_extra_when_jax_is_missing(name):
    message = _run_without_jax(f"import pertpy as pt\ntry:\n    pt.tl.{name}\nexcept ImportError as e:\n    print(e)\n")
    assert name in message
    assert "pip install 'pertpy[jax]'" in message


@pytest.mark.parametrize("name", JAX_FREE_TOOLS)
def test_jax_free_tools_still_work_without_jax(name):
    assert _run_without_jax(f"import pertpy as pt; pt.tl.{name}; print('ok')") == "ok"


def test_wasserstein_metric_reports_the_extra_when_jax_is_missing():
    message = _run_without_jax(
        "import pertpy as pt\ntry:\n    pt.tl.Distance(metric='wasserstein')\nexcept ImportError as e:\n    print(e)\n"
    )
    assert "wasserstein" in message
    assert "pip install 'pertpy[jax]'" in message


def test_jax_free_distance_metric_works_without_jax():
    value = _run_without_jax(
        "import numpy as np, pertpy as pt\n"
        "rng = np.random.default_rng(0)\n"
        "X, Y = rng.normal(size=(20, 5)), rng.normal(size=(20, 5)) + 5.0\n"
        "print(round(float(pt.tl.Distance(metric='edistance')(X, Y)), 4))\n"
    )
    assert float(value) > 0


def test_assign_mixture_model_reports_the_extra_when_jax_is_missing():
    message = _run_without_jax(
        "import numpy as np, anndata as ad, pertpy as pt\n"
        "adata = ad.AnnData(np.random.default_rng(0).poisson(2.0, size=(60, 4)).astype(float))\n"
        "try:\n"
        "    pt.pp.GuideAssignment().assign_mixture_model(adata)\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    assert "assign_mixture_model" in message
    assert "pip install 'pertpy[jax]'" in message


def test_assign_by_threshold_works_without_jax():
    assert (
        _run_without_jax(
            "import numpy as np, anndata as ad, pertpy as pt\n"
            "adata = ad.AnnData(np.random.default_rng(0).poisson(2.0, size=(60, 4)).astype(float))\n"
            "pt.pp.GuideAssignment().assign_by_threshold(adata, assignment_threshold=3, output_layer='assigned')\n"
            "print('assigned' in adata.layers)\n"
        )
        == "True"
    )


def test_jax_import_reports_the_extra():
    with pytest.raises(ImportError, match=r"pertpy\[jax\]"), jax_import("Sccoda"):
        raise ModuleNotFoundError("No module named 'numpyro'", name="numpyro")


def test_jax_import_names_the_feature_and_module():
    with pytest.raises(ImportError, match="Sccoda requires .* 'numpyro' is not installed"), jax_import("Sccoda"):
        raise ModuleNotFoundError("No module named 'numpyro'", name="numpyro")


def test_jax_import_passes_through_unrelated_import_errors():
    with pytest.raises(ImportError, match="toytree"), jax_import("Tasccoda"):
        raise ModuleNotFoundError("No module named 'toytree'", name="toytree")
