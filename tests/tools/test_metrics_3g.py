import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from anndata import AnnData
from pertpy.tools._metrics_3g import (
    compare_class,
    compare_de,
    compare_knn,
)


@pytest.fixture
def test_data(rng):
    X = rng.normal(size=(100, 10))
    Y = rng.normal(size=(100, 10))
    C = rng.normal(size=(100, 10))
    return X, Y, C


@pytest.fixture
def compare_de_adata(rng):
    adata = AnnData(rng.normal(size=(100, 10)))
    genes = np.rec.fromarrays(
        [np.array([f"gene{i}" for i in range(10)])],
        names=["group1", "O"],
    )
    adata.uns["de_key1"] = {
        "names": genes,
        "scores": {"group1": rng.random(10)},
        "pvals_adj": {"group1": rng.random(10)},
    }
    adata.uns["de_key2"] = {
        "names": genes,
        "scores": {"group1": rng.random(10)},
        "pvals_adj": {"group1": rng.random(10)},
    }
    return adata


@pytest.fixture
def compare_de_dataframe(rng):
    df1 = pd.DataFrame(
        {
            "variable": ["gene" + str(i) for i in range(10)],
            "log_fc": rng.random(10),
            "adj_p_value": rng.random(10),
        }
    )
    df2 = pd.DataFrame(
        {
            "variable": ["gene" + str(i) for i in range(10)],
            "log_fc": rng.random(10),
            "adj_p_value": rng.random(10),
        }
    )
    return df1, df2


def test_error_both_keys_and_dfs(compare_de_adata, compare_de_dataframe):
    with pytest.raises(ValueError):
        compare_de(adata=compare_de_adata, de_key1="de_key1", de_df1=compare_de_dataframe[0])


def test_error_missing_adata():
    with pytest.raises(ValueError):
        compare_de(de_key1="de_key1", de_key2="de_key2")


def test_error_missing_df(compare_de_dataframe):
    with pytest.raises(ValueError):
        compare_de(de_df1=compare_de_dataframe[0])


def test_compare_de_key(compare_de_adata):
    results = compare_de(adata=compare_de_adata, de_key1="de_key1", de_key2="de_key2", shared_top=5)
    assert "shared_top_genes" in results
    assert "scores_corr" in results
    assert "pvals_adj_corr" in results
    assert "scores_ranks_corr" in results


def test_compare_de_df(compare_de_dataframe):
    results = compare_de(de_df1=compare_de_dataframe[0], de_df2=compare_de_dataframe[1], shared_top=5)
    assert "shared_top_genes" in results
    assert "scores_corr" in results
    assert "pvals_adj_corr" in results
    assert "scores_ranks_corr" in results


def test_compare_class(test_data):
    X, Y, C = test_data
    result = compare_class(X, Y, C)
    assert result <= 1


def test_compare_knn(test_data):
    X, Y, C = test_data
    result = compare_knn(X, Y, C)
    assert isinstance(result, dict)
    assert "comp" in result
    assert isinstance(result["comp"], float)

    result_no_ctrl = compare_knn(X, Y)
    assert isinstance(result_no_ctrl, dict)
