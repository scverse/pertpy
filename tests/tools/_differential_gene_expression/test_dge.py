import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from anndata import AnnData

pytest.skip("Disabled", allow_module_level=True)


@pytest.fixture
def adata(rng):
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
def dataframe(rng):
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


def test_error_both_keys_and_dfs(adata, dataframe):
    with pytest.raises(ValueError):
        pt_DGE = pt.tl.DGEEVAL()
        pt_DGE.compare(adata=adata, de_key1="de_key1", de_df1=dataframe[0])


def test_error_missing_adata():
    with pytest.raises(ValueError):
        pt_DGE = pt.tl.DGEEVAL()
        pt_DGE.compare(de_key1="de_key1", de_key2="de_key2")


def test_error_missing_df(dataframe):
    with pytest.raises(ValueError):
        pt_DGE = pt.tl.DGEEVAL()
        pt_DGE.compare(de_df1=dataframe[0])


def test_key(adata):
    pt_DGE = pt.tl.DGEEVAL()
    results = pt_DGE.compare(adata=adata, de_key1="de_key1", de_key2="de_key2", shared_top=5)
    assert "shared_top_genes" in results
    assert "scores_corr" in results
    assert "pvals_adj_corr" in results
    assert "scores_ranks_corr" in results


def test_df(dataframe):
    pt_DGE = pt.tl.DGEEVAL()
    results = pt_DGE.compare(de_df1=dataframe[0], de_df2=dataframe[1], shared_top=5)
    assert "shared_top_genes" in results
    assert "scores_corr" in results
    assert "pvals_adj_corr" in results
    assert "scores_ranks_corr" in results
