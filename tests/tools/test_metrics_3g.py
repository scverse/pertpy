import pertpy as pt
import numpy as np
import pytest
from pertpy.tools._metrics_3g import (
    compare_de,
    compare_class,
    compare_dist,
    compare_knn,
)


class TestMetrics3G:
    @pytest.fixture
    def test_data(self):
        rng = np.random.default_rng()
        X = rng.normal(size=(100, 10))
        Y = rng.normal(size=(100, 10))
        C = rng.normal(size=(100, 10))
        return X, Y, C

    def test_compare_de(self, test_data):
        X, Y, C = test_data
        result = compare_de(X, Y, C, shared_top=5)
        assert isinstance(result, dict)
        required_keys = {
            "shared_top_genes",
            "scores_corr",
            "pvals_adj_corr",
            "scores_ranks_corr",
        }
        assert all(key in result for key in required_keys)

    def test_compare_class(self, test_data):
        X, Y, C = test_data
        result = compare_class(X, Y, C)
        assert result <= 1

    def test_compare_knn(self, test_data):
        X, Y, C = test_data
        result = compare_knn(X, Y, C)
        assert isinstance(result, dict)
        assert "comp" in result
        assert isinstance(result["comp"], float)

        result_no_ctrl = compare_knn(X, Y)
        assert isinstance(result_no_ctrl, dict)

    def test_compare_dist(self, test_data):
        X, Y, C = test_data
        res_simple = compare_dist(X, Y, C, mode="simple")
        assert isinstance(res_simple, float)
        res_scaled = compare_dist(X, Y, C, mode="scaled")
        assert isinstance(res_scaled, float)
        with pytest.raises(ValueError):
            compare_dist(X, Y, C, mode="new_mode")
