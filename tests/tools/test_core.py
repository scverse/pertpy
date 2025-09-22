import numpy as np
import pytest
from pertpy.tools.core import _is_raw_counts
from scipy import sparse


@pytest.mark.parametrize(
    "data,expected",
    [
        # Dense arrays - positive cases
        (np.array([[1, 2, 3], [4, 0, 5]]), True),  # integers with zeros
        (np.array([[0, 0], [0, 0]]), True),  # all zeros
        (np.array([[1, 2], [3, 4]]), True),  # positive integers
        (np.array([[100, 200], [300, 400]]), True),  # larger integers
        # Dense arrays - negative cases
        (np.array([[1.5, 2.0], [3.0, 4.5]]), False),  # floats
        (np.array([[1, 2.1], [3, 4]]), False),  # mixed int/float
        (np.array([[-1, 2], [3, 4]]), False),  # negative values
        (np.log1p(np.array([[1, 2], [3, 4]])), False),  # log-transformed
        # Edge cases
        (np.array([[0]]), True),  # single zero
        (np.array([[1]]), True),  # single positive integer
        (np.array([[1.0]]), True),  # float that equals integer
    ],
)
def test_dense_arrays(data, expected):
    assert _is_raw_counts(data) == expected


@pytest.mark.parametrize("sparse_type", [sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix])
def test_sparse_arrays_positive(sparse_type):
    dense_data = np.array([[1, 0, 3], [0, 5, 0], [2, 0, 4]])
    sparse_data = sparse_type(dense_data)
    assert _is_raw_counts(sparse_data)


@pytest.mark.parametrize("sparse_type", [sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix])
def test_sparse_arrays_negative(sparse_type):
    dense_data = np.array([[1.5, 0, 3.2], [0, 5.7, 0]])
    sparse_data = sparse_type(dense_data)
    assert not _is_raw_counts(sparse_data)


def test_large_array_sampling():
    large_data = np.random.default_rng().integers(0, 100, size=(2000, 2000))
    assert _is_raw_counts(large_data)


def test_large_sparse_array_sampling():
    dense_data = np.random.default_rng().integers(0, 10, size=(2000, 2000))
    dense_data[dense_data < 7] = 0
    sparse_data = sparse.csr_matrix(dense_data)
    assert _is_raw_counts(sparse_data)


def test_empty_sparse_matrix():
    sparse_data = sparse.csr_matrix((100, 100))
    assert _is_raw_counts(sparse_data)
