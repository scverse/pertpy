from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from mudata import MuData

try:
    import ete3
except ImportError:
    pytest.skip("ete3 not available", allow_module_level=True)

import pertpy as pt

CWD = Path(__file__).parent.resolve()


class TestscCODA:
    sccoda = pt.tl.Sccoda()

    @pytest.fixture
    def adata(self):
        cells = pt.dt.haber_2017_regions()
        cells = sc.pp.subsample(cells, 0.1, copy=True)

        return cells

    def test_load(self, adata):
        mdata = self.sccoda.load(
            adata,
            type="cell_level",
            generate_sample_level=True,
            cell_type_identifier="cell_label",
            sample_identifier="batch",
            covariate_obs=["condition"],
        )
        assert isinstance(mdata, MuData)
        assert "rna" in mdata.mod
        assert "coda" in mdata.mod

    def test_prepare(self, adata):
        mdata = self.sccoda.load(
            adata,
            type="cell_level",
            generate_sample_level=True,
            cell_type_identifier="cell_label",
            sample_identifier="batch",
            covariate_obs=["condition"],
        )
        mdata = self.sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
        assert "scCODA_params" in mdata["coda"].uns
        assert "covariate_matrix" in mdata["coda"].obsm
        assert "sample_counts" in mdata["coda"].obsm
        assert isinstance(mdata["coda"].obsm["sample_counts"], np.ndarray)
        assert np.sum(mdata["coda"].obsm["covariate_matrix"]) == 6

    def test_run_nuts(self, adata):
        mdata = self.sccoda.load(
            adata,
            type="cell_level",
            generate_sample_level=True,
            cell_type_identifier="cell_label",
            sample_identifier="batch",
            covariate_obs=["condition"],
        )
        mdata = self.sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
        self.sccoda.run_nuts(mdata, num_samples=1000, num_warmup=100)
        assert "effect_df_condition[T.Hpoly.Day10]" in mdata["coda"].varm
        assert "effect_df_condition[T.Hpoly.Day3]" in mdata["coda"].varm
        assert "effect_df_condition[T.Salmonella]" in mdata["coda"].varm
        assert "intercept_df" in mdata["coda"].varm
        assert mdata["coda"].varm["effect_df_condition[T.Hpoly.Day10]"].shape == (8, 7)
        assert mdata["coda"].varm["effect_df_condition[T.Hpoly.Day3]"].shape == (8, 7)
        assert mdata["coda"].varm["effect_df_condition[T.Salmonella]"].shape == (8, 7)
        assert mdata["coda"].varm["intercept_df"].shape == (8, 5)

    def test_credible_effects(self, adata):
        adata_salm = adata[adata.obs["condition"].isin(["Control", "Salmonella"])]
        mdata = self.sccoda.load(
            adata_salm,
            type="cell_level",
            generate_sample_level=True,
            cell_type_identifier="cell_label",
            sample_identifier="batch",
            covariate_obs=["condition"],
        )
        mdata = self.sccoda.prepare(mdata, formula="condition", reference_cell_type="Goblet")
        self.sccoda.run_nuts(mdata)
        assert isinstance(self.sccoda.credible_effects(mdata), pd.Series)
        assert self.sccoda.credible_effects(mdata)["condition[T.Salmonella]"]["Enterocyte"]

    def test_make_arviz(self, adata):
        adata_salm = adata[adata.obs["condition"].isin(["Control", "Salmonella"])]
        mdata = self.sccoda.load(
            adata_salm,
            type="cell_level",
            generate_sample_level=True,
            cell_type_identifier="cell_label",
            sample_identifier="batch",
            covariate_obs=["condition"],
        )
        mdata = self.sccoda.prepare(mdata, formula="condition", reference_cell_type="Goblet")
        self.sccoda.run_nuts(mdata)
        arviz_data = self.sccoda.make_arviz(mdata, num_prior_samples=100)
        assert isinstance(arviz_data, az.InferenceData)
