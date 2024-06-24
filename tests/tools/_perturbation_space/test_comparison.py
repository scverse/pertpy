import pertpy as pt
import pytest


@pytest.fixture
def test_data(rng):
    X = rng.normal(size=(100, 10))
    Y = rng.normal(size=(100, 10))
    C = rng.normal(size=(100, 10))
    return X, Y, C


def test_compare_class(test_data):
    X, Y, C = test_data
    pt_comparison = pt.tl.PerturbationComparison()
    result = pt_comparison.compare_classification(X, Y, C)
    assert result <= 1


def test_compare_knn(test_data):
    X, Y, C = test_data
    pt_comparison = pt.tl.PerturbationComparison()
    result = pt_comparison.compare_knn(X, Y, C)
    assert isinstance(result, dict)
    assert "comp" in result
    assert isinstance(result["comp"], float)

    result_no_ctrl = pt_comparison.compare_knn(X, Y)
    assert isinstance(result_no_ctrl, dict)
