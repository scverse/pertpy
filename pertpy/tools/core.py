import numpy as np
from scipy import sparse


def _is_raw_counts(X: np.ndarray | sparse.spmatrix) -> bool:
    """Check if data appears to be raw counts."""
    if sparse.issparse(X):
        sample = X[:1000, :1000] if X.shape[0] > 1000 else X
        data = sample.data
    else:
        sample = X[:1000, :1000] if X.shape[0] > 1000 else X
        data = sample.ravel()

    non_zero_data = data[data > 0]
    if len(non_zero_data) == 0:
        return True

    return np.all(data >= 0) and np.all(data == np.round(data))
