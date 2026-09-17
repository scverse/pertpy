import numpy as np

from pertpy._types import CSBase


def _is_raw_counts(X: np.ndarray | CSBase) -> bool:
    """Check if data appears to be raw counts."""
    sample = X[:1000, :1000] if X.shape[0] > 1000 else X
    data = sample.ravel() if isinstance(sample, np.ndarray) else sample.data

    non_zero_data = data[data > 0]
    if len(non_zero_data) == 0:
        return True

    return bool(np.all(data >= 0) and np.all(data == np.round(data)))
