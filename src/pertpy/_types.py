from typing import cast

import numpy as np
import pandas as pd
from scipy import sparse

CSBase = sparse.csr_matrix | sparse.csc_matrix | sparse.csr_array | sparse.csc_array
CSRBase = sparse.csr_matrix
CSCBase = sparse.csc_matrix
SpBase = sparse.spmatrix

RandomStateLike = int | np.random.Generator | np.random.RandomState | None


def cast_matrix(X: object) -> np.ndarray | CSBase:
    """Cast a matrix such as :attr:`~anndata.AnnData.X` to the types pertpy operates on.

    `AnnData` types its matrices as a union that also includes backed, dask and cupy containers,
    none of which the functions using this cast accept.
    Like every `cast`, this only informs the type checker and does not convert anything at runtime.
    """
    return cast("np.ndarray | CSBase", X)


def cast_dense(X: object) -> np.ndarray:
    """Cast a matrix that the surrounding code only supports as a dense array.

    Unlike :func:`fast_array_utils.conv.to_dense`, this does not densify anything, see :func:`cast_matrix`.
    """
    return cast("np.ndarray", X)


def cast_frame(df: object) -> pd.DataFrame:
    """Cast an annotation frame such as :attr:`~anndata.AnnData.obs` to a :class:`~pandas.DataFrame`.

    `AnnData` also types them as the `Dataset2D` of a lazily read object, which the functions using this cast
    do not accept, see :func:`cast_matrix`.
    """
    return cast("pd.DataFrame", df)
