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

    sccoda = pt.tl.Sccoda()

    @pytest.fixture
    def adata(self):
        haber_data = pd.read_csv(f"{CWD}/haber_data.csv")
        adata = from_pandas(haber_data, covariate_columns=["Mouse"])
        adata.obs["Condition"] = adata.obs["Mouse"].str.replace(r"_[0-9]", "", regex=True)
        return adata

    def test_load(self, adata):
        mdata = self.sccoda.load(adata, type="sample_level")
        assert isinstance(mdata, MuData)
        assert "rna" in mdata.mod
        assert "coda" in mdata.mod

    def test_prepare(self, adata):
        mdata = self.sccoda.load(adata, type="sample_level")
        mdata = self.sccoda.prepare(mdata, formula="Condition", reference_cell_type="Endocrine")
        assert "scCODA_params" in mdata["coda"].uns
        assert "covariate_matrix" in mdata["coda"].obsm
        assert "sample_counts" in mdata["coda"].obsm
        assert isinstance(mdata["coda"].obsm["sample_counts"], np.ndarray)
        assert np.sum(mdata["coda"].obsm["covariate_matrix"]) == 6

    def test_run_nuts(self, adata):
        mdata = self.sccoda.load(adata, type="sample_level")
        mdata = self.sccoda.prepare(mdata, formula="Condition", reference_cell_type="Endocrine")
        self.sccoda.run_nuts(mdata, num_samples=1000, num_warmup=100)
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
        mdata = self.sccoda.load(adata_salm, type="sample_level")
        mdata = self.sccoda.prepare(mdata, formula="Condition", reference_cell_type="Goblet")
        self.sccoda.run_nuts(mdata)
        assert isinstance(self.sccoda.credible_effects(mdata), pd.Series)
        assert self.sccoda.credible_effects(mdata)["Condition[T.Salm]"]["Enterocyte"]

    def test_make_arviz(self, adata):
        adata_salm = adata[adata.obs["Condition"].isin(["Control", "Salm"])]
        mdata = self.sccoda.load(adata_salm, type="sample_level")
        mdata = self.sccoda.prepare(mdata, formula="Condition", reference_cell_type="Goblet")
        self.sccoda.run_nuts(mdata)
        arviz_data = self.sccoda.make_arviz(mdata, num_prior_samples=100)
        assert isinstance(arviz_data, az.InferenceData)
