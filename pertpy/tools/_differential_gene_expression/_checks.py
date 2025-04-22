import numpy as np
from scipy.sparse import issparse, spmatrix


def check_is_numeric_matrix(array: np.ndarray | spmatrix) -> None:
    """Check if a matrix is numeric and only contains finite/non-NA values.

    Args:
        array: Dense or sparse matrix to check.

    Raises:
        ValueError: If the matrix is not numeric or contains NaNs or infinite values.
    """
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Counts must be numeric.")
    if issparse(array):
        if np.any(~np.isfinite(array.data)):
            raise ValueError("Counts cannot contain negative, NaN or Inf values.")
    elif np.any(~np.isfinite(array)):
        raise ValueError("Counts cannot contain negative, NaN or Inf values.")


def check_is_integer_matrix(array: np.ndarray | spmatrix, tolerance: float = 1e-6) -> None:
    """Check if a matrix container integers, or floats that are close to integers.

    Args:
        array: Dense or sparse matrix to check.
        tolerance: Values must be this close to integers.

    Raises:
        ValueError: If the matrix contains values that are not close to integers.
    """
    if issparse(array):
        if not array.data.dtype.kind == "i" and not np.all(np.abs(array.data - np.round(array.data)) < tolerance):
            raise ValueError("Non-zero elements of the matrix must be close to integer values.")
    elif array.dtype.kind != "i" and not np.all(np.abs(array - np.round(array)) < tolerance):
        raise ValueError("Matrix must be a count matrix.")
    if (array < 0).sum() > 0:
        raise ValueError("Non-zero elements of the matrix must be positive.")
