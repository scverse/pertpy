from typing import cast

import numpy as np
import pandas as pd
from scipy import sparse

CSBase = sparse.csr_matrix | sparse.csc_matrix | sparse.csr_array | sparse.csc_array
CSRBase = sparse.csr_matrix
CSCBase = sparse.csc_matrix
SpBase = sparse.spmatrix

RandomStateLike = int | np.random.Generator | np.random.RandomState | None


def as_matrix(X: object) -> np.ndarray | CSBase:
    """Narrow a matrix such as :attr:`~anndata.AnnData.X` to the types pertpy operates on.

    This is a static narrowing without a runtime effect: `AnnData` types its matrices as a union that also
    includes backed, dask and cupy containers, none of which the functions using this helper accept.
    """
    return cast("np.ndarray | CSBase", X)


def as_dense(X: object) -> np.ndarray:
    """Narrow a matrix that the surrounding code only supports as a dense array.

    This is a static narrowing without a runtime effect, see :func:`as_matrix`.
    """
    return cast("np.ndarray", X)


def as_frame(df: object) -> pd.DataFrame:
    """Narrow an annotation frame such as :attr:`~anndata.AnnData.obs` to a :class:`~pandas.DataFrame`.

    This is a static narrowing without a runtime effect: `AnnData` also types them as the `Dataset2D` of a
    lazily read object, which the functions using this helper do not accept.
    """
    return cast("pd.DataFrame", df)
