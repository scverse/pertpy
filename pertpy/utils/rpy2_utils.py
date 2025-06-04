from functools import singledispatch
from typing import Sequence, Union, Any
from rpy2 import robjects as ro
from rpy2.robjects import vectors, methods
from rpy2.robjects.vectors import ListVector
from rpy2.robjects.methods import RS4
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter, numpy2ri, pandas2ri
from anndata2ri import scipy2ri
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from typing import Sequence, Union, Tuple
import types

def get_rpy2_objects():
    """
    Return commonly used rpy2 classes, functions, and modules for local use.
    
    Returns
    -------
    tuple : (ro, importr, STAP, localconverter, default_converter, pandas2ri, numpy2ri)
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr, STAP
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects import default_converter

    except ImportError as e:
        raise ImportError("rpy2 must be installed to use this function.") from e

    return ro, importr, STAP, localconverter, default_converter, pandas2ri, numpy2ri

ro, importr, STAP, localconverter, default_converter, pandas2ri, numpy2ri = get_rpy2_objects()

def lazy_import_r_packages(packages: Sequence[str] | str) -> tuple[Union[types.ModuleType, str], ...]:
    """
    Lazily import R packages using rpy2.robjects.packages.importr.

    For each package name:
    - Try to import it.
    - If successful, return the imported package object, prefixed with "r_" in variable naming convention.
    - If failed, return the package name string (as a marker of failure).

    Parameters
    ----------
    packages : list of str
        R package names to import.

    Returns
    -------
    tuple
        A tuple of imported modules or failed package names (as strings), in order.
    """
    ro, importr, _, *_ = get_rpy2_objects()

    if isinstance(packages, str):
        try:
            r_pkg = importr(packages)
        except Exception as e:
            raise ImportError(f"Failed to import required R package: {packages}")
        return r_pkg
        
    imported = []
    failures = []
    for pkg in packages:
        try:
            r_pkg = importr(pkg)
            imported.append(r_pkg)
        except Exception as e:
            imported.append(pkg)
            failures.append(pkg)

    if failures:
        raise ImportError(f"Failed to import required R packages: {', '.join(failures)}")
    
    return tuple(imported)


### 

# ----------------------
# Python -> R converters
# ----------------------
@singledispatch
def _py_to_r(obj) -> ro.RObject:
    """
    Fallback: convert with rpy2's generic converter.
    Lists/tuples of atomic types -> R vector; mixed lists -> R list.
    """
    with localconverter(default_converter):
        return ro.conversion.py2rpy(obj)

@_py_to_r.register(type(None))
def _(obj) -> ro.NULL:
    return ro.NULL

@_py_to_r.register(bool)
def _(obj: bool) -> vectors.BoolVector:
    return vectors.BoolVector([obj])

@_py_to_r.register(int)
def _(obj: int) -> vectors.IntVector:
    return vectors.IntVector([obj])

@_py_to_r.register(float)
def _(obj: float) -> vectors.FloatVector:
    return vectors.FloatVector([obj])

@_py_to_r.register(str)
def _(obj: str) -> vectors.StrVector:
    return vectors.StrVector([obj])

@_py_to_r.register(dict)
def _(obj: dict) -> ListVector:
    # Convert a Python dict to an R named list
    rdict = {str(k): _py_to_r(v) for k, v in obj.items()}
    return ListVector(rdict)

@_py_to_r.register(np.ndarray)
def _(obj: np.ndarray) -> ro.Matrix:
    with localconverter(default_converter + numpy2ri.converter):
        return numpy2ri.py2rpy(obj)

@_py_to_r.register(pd.DataFrame)
def _(obj: pd.DataFrame) -> ro.DataFrame:
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(obj)

@_py_to_r.register(pd.Series)
def _(obj: pd.Series) -> vectors.Vector:
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(obj)

@_py_to_r.register(csr_matrix)
def _(obj: csr_matrix) -> ro.Matrix:
    with localconverter(default_converter + scipy2ri.converter):
        return ro.conversion.py2rpy(obj)

@_py_to_r.register(csc_matrix)
def _(obj: csc_matrix) -> ro.Matrix:
    with localconverter(default_converter + scipy2ri.converter):
        return ro.conversion.py2rpy(obj)

# ----------------------
# R -> Python converters
# ----------------------
@singledispatch
def _r_to_py(obj: ro.RObject):
    """Fallback: generic R->Python via rpy2"""
    with localconverter(default_converter):
        return ro.conversion.rpy2py(obj)

@_r_to_py.register(type(ro.NULL))
def _(obj) -> None:
    return None

@_r_to_py.register(vectors.BoolVector)
def _(obj: vectors.BoolVector):
    py = [bool(x) for x in list(obj)]
    return py[0] if len(py) == 1 else py

@_r_to_py.register(vectors.IntVector)
def _(obj: vectors.IntVector):
    py = list(obj)
    return int(py[0]) if len(py) == 1 else py

@_r_to_py.register(vectors.FloatVector)
def _(obj: vectors.FloatVector):
    py = list(obj)
    return float(py[0]) if len(py) == 1 else py

@_r_to_py.register(vectors.StrVector)
def _(obj: vectors.StrVector):
    py = list(obj)
    return str(py[0]) if len(py) == 1 else py

@_r_to_py.register(ListVector)
def _(obj: ListVector):
    # Convert each element first
    items = [_r_to_py(obj[i]) for i in range(len(obj))]
    # Handle names (NULL means no names)
    names_r = obj.names
    if names_r == ro.NULL:
        return items
    names = list(names_r)
    return {names[i]: items[i] for i in range(len(items))}

@_r_to_py.register(vectors.Matrix)
def _(obj: vectors.Matrix):
    with localconverter(default_converter + numpy2ri.converter):
        return ro.conversion.rpy2py(obj)

@_r_to_py.register(ro.DataFrame)
def _(obj: ro.DataFrame) -> pd.DataFrame:
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(obj)

@_r_to_py.register(vectors.Vector)
def _(obj: vectors.Vector):
    return list(obj)

@_r_to_py.register(RS4)
def _(obj: RS4):
    # Sparse S4 matrix types go through scipy2ri
    if obj.rclass[0] in ("dgCMatrix", "dgRMatrix"):
        with localconverter(default_converter + scipy2ri.converter):
            return ro.conversion.rpy2py(obj)
    # Other S4 -> dict of slots
    return {name: _r_to_py(obj.slots[name]) for name in obj.slotnames()}


def _ad_to_rmat(adata, layer="X"):
    from rpy2 import robjects as ro
    from rpy2.robjects import baseenv
    from rpy2.robjects.packages import importr
    import numpy as np
    
    # 1) grab & transpose
    mat = adata.X if layer == "X" else adata.layers[layer]
    mat = mat.T

    # 2) convert to R
    rmat = _py_to_r(mat)

    # 3) build an *unnamed* R list(r_rownames, r_colnames)
    r_colnames = _py_to_r(np.asarray(adata.obs_names))
    r_rownames = _py_to_r(np.asarray(adata.var_names))
    dim_list   = ro.r.list(r_rownames, r_colnames)

    # 4) set dimnames via baseenv()
    assign_dim = baseenv["dimnames<-"]
    rmat = assign_dim(rmat, dim_list)

    return rmat

def _ad_to_dge(adata, layer="X") -> Any:
    edgeR = importr("edgeR")
    counts  = _ad_to_rmat(adata, layer)
    samples = _py_to_r(adata.obs)
    return edgeR.DGEList(counts=counts, samples=samples)


