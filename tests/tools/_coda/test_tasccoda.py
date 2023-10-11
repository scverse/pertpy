from pathlib import Path

import numpy as np
import pytest
import scanpy as sc
from mudata import MuData

try:
    import ete3
except ImportError:
    pytest.skip("ete3 not available", allow_module_level=True)

import pertpy as pt

CWD = Path(__file__).parent.resolve()


class TesttascCODA:
    tasccoda = pt.tl.Tasccoda()

    @pytest.fixture
    def smillie_adata(self):
        smillie_adata = pt.dt.smillie()
        smillie_adata = sc.pp.subsample(smillie_adata, 0.1, copy=True)

        return smillie_adata

    def test_load(self, smillie_adata):
        mdata = self.tasccoda.load(
            smillie_adata,
            type="sample_level",
            levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            key_added="lineage",
            add_level_name=True,
        )
        assert isinstance(mdata, MuData)
        assert "rna" in mdata.mod
        assert "coda" in mdata.mod
        assert "lineage" in mdata["coda"].uns

    def test_prepare(self, smillie_adata):
        mdata = self.tasccoda.load(
            smillie_adata,
            type="sample_level",
            levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            key_added="lineage",
            add_level_name=True,
        )
        mdata = self.tasccoda.prepare(
            mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        )
        assert "scCODA_params" in mdata["coda"].uns
        assert "covariate_matrix" in mdata["coda"].obsm
        assert "sample_counts" in mdata["coda"].obsm
        assert isinstance(mdata["coda"].obsm["sample_counts"], np.ndarray)
        assert np.sum(mdata["coda"].obsm["covariate_matrix"]) == 8

    def test_run_nuts(self, smillie_adata):
        mdata = self.tasccoda.load(
            smillie_adata,
            type="sample_level",
            levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            key_added="lineage",
            add_level_name=True,
        )
        mdata = self.tasccoda.prepare(
            mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        )
        self.tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100)
        assert "effect_df_Health[T.Inflamed]" in mdata["coda"].varm
        assert "effect_df_Health[T.Non-inflamed]" in mdata["coda"].varm
        assert mdata["coda"].varm["effect_df_Health[T.Inflamed]"].shape == (51, 7)
        assert mdata["coda"].varm["effect_df_Health[T.Non-inflamed]"].shape == (51, 7)
