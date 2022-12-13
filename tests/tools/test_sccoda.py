from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from mudata import MuData
from sccoda.util.cell_composition_data import from_pandas

import pertpy as pt

CWD = Path(__file__).parent.resolve()


class TestscCODA:

    sccoda_model = pt.tl.SccodaModel2()
    tasccoda_model = pt.tl.TasccodaModel2()

    @pytest.fixture
    def adata(self):
        haber_data = pd.read_csv(f"{CWD}/haber_data.csv")
        adata = from_pandas(haber_data, covariate_columns=["Mouse"])
        adata.obs["Condition"] = adata.obs["Mouse"].str.replace(r"_[0-9]", "", regex=True)
        return adata

    def test_load(self, adata):
        mdata = self.sccoda_model.load(adata, type="sample_level")
        assert isinstance(mdata, MuData)
        assert "rna" in mdata.mod
        assert "coda" in mdata.mod

    def test_prepare(self, adata):
        mdata = self.sccoda_model.load(adata, type="sample_level")
        mdata = self.sccoda_model.prepare(mdata, formula="Condition", reference_cell_type="Endocrine")
        assert "scCODA_params" in mdata["coda"].uns
        assert "covariate_matrix" in mdata["coda"].obsm
        assert "sample_counts" in mdata["coda"].obsm
        assert isinstance(mdata["coda"].obsm["sample_counts"], np.ndarray)
        assert np.sum(mdata["coda"].obsm["covariate_matrix"]) == 6

    def test_go_nuts(self, adata):
        mdata = self.sccoda_model.load(adata, type="sample_level")
        mdata = self.sccoda_model.prepare(mdata, formula="Condition", reference_cell_type="Endocrine")
        self.sccoda_model.go_nuts(mdata, num_warmup=100, num_samples=1000)
        assert "effect_df_Condition[T.H.poly.Day10]" in mdata["coda"].varm
        assert "effect_df_Condition[T.H.poly.Day3]" in mdata["coda"].varm
        assert "effect_df_Condition[T.Salm]" in mdata["coda"].varm
        assert "intercept_df" in mdata["coda"].varm
        assert mdata["coda"].varm["effect_df_Condition[T.H.poly.Day10]"].shape == (9, 7)
        assert mdata["coda"].varm["effect_df_Condition[T.H.poly.Day3]"].shape == (9, 7)
        assert mdata["coda"].varm["effect_df_Condition[T.Salm]"].shape == (9, 7)
        assert mdata["coda"].varm["intercept_df"].shape == (9, 5)

    def test_credible_effects(self, adata):
        adata_salm = adata[adata.obs["Condition"].isin(["Control", "Salm"])]
        mdata = self.sccoda_model.load(adata_salm, type="sample_level")
        mdata = self.sccoda_model.prepare(mdata, formula="Condition", reference_cell_type="Goblet")
        self.sccoda_model.go_nuts(mdata)
        assert isinstance(self.sccoda_model.credible_effects(mdata), pd.Series)
        assert self.sccoda_model.credible_effects(mdata)["Condition[T.Salm]"]["Enterocyte"]

    def test_make_arviz(self, adata):
        adata_salm = adata[adata.obs["Condition"].isin(["Control", "Salm"])]
        mdata = self.sccoda_model.load(adata_salm, type="sample_level")
        mdata = self.sccoda_model.prepare(mdata, formula="Condition", reference_cell_type="Goblet")
        self.sccoda_model.go_nuts(mdata)
        arviz_data = self.sccoda_model.make_arviz(mdata, num_prior_samples=100)
        assert isinstance(arviz_data, az.InferenceData)

    @pytest.fixture
    def smillie_adata(self):
        smillie_adata = sc.read_h5ad(f"{CWD}/smillie_data.h5ad")
        return smillie_adata

    def test_load_tasccoda(self, smillie_adata):
        mdata = self.tasccoda_model.load(
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

    def test_prepare_tasccoda(self, smillie_adata):
        mdata = self.tasccoda_model.load(
            smillie_adata,
            type="sample_level",
            levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            key_added="lineage",
            add_level_name=True,
        )
        mdata = self.tasccoda_model.prepare(
            mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        )
        assert "scCODA_params" in mdata["coda"].uns
        assert "covariate_matrix" in mdata["coda"].obsm
        assert "sample_counts" in mdata["coda"].obsm
        assert isinstance(mdata["coda"].obsm["sample_counts"], np.ndarray)
        assert np.sum(mdata["coda"].obsm["covariate_matrix"]) == 85

    def test_go_nuts_tasccoda(self, smillie_adata):
        mdata = self.tasccoda_model.load(
            smillie_adata,
            type="sample_level",
            levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            key_added="lineage",
            add_level_name=True,
        )
        mdata = self.tasccoda_model.prepare(
            mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        )
        self.tasccoda_model.go_nuts(mdata, num_samples=1000, num_warmup=100)
        assert "effect_df_Health[T.Inflamed]" in mdata["coda"].varm
        assert "effect_df_Health[T.Non-inflamed]" in mdata["coda"].varm
        assert mdata["coda"].varm["effect_df_Health[T.Inflamed]"].shape == (51, 7)
        assert mdata["coda"].varm["effect_df_Health[T.Non-inflamed]"].shape == (51, 7)
