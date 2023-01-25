from math import isclose
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc
from sklearn.ensemble import RandomForestRegressor

import pertpy as pt
from pertpy.tools._augurpy import Params

CWD = Path(__file__).parent.resolve()


class TestAugur:
    ag_rfc = pt.tl.Augurpy("random_forest_classifier", Params(random_state=42))
    ag_lrc = pt.tl.Augurpy("logistic_regression_classifier", Params(random_state=42))
    ag_rfr = pt.tl.Augurpy("random_forest_regressor", Params(random_state=42))

    def test_load(self):
        """Test if load function creates anndata objects."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        ag = pt.tl.Augurpy(estimator="random_forest_classifier")

        loaded_adata = ag.load(sc_sim_adata)
        loaded_df = ag.load(sc_sim_adata.to_df(), meta=sc_sim_adata.obs, cell_type_col="cell_type", label_col="label")

        assert loaded_adata.obs["y_"].equals(loaded_df.obs["y_"]) is True
        assert sc_sim_adata.to_df().equals(loaded_adata.to_df()) is True and sc_sim_adata.to_df().equals(
            loaded_df.to_df()
        )

    def test_random_forest_classifier(self):
        """Tests random forest for auc calculation."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_rfc.load(sc_sim_adata)
        sc.pp.highly_variable_genes(sc_sim_adata)
        adata, results = self.ag_rfc.predict(
            sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42, select_variance_features=False
        )

        assert results["CellTypeA"][2]["subsample_idx"] == 2
        assert "augur_score" in adata.obs.columns
        assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.634920, 0.933484, 0.902494])
        assert "feature_importances" in results.keys()
        assert len(set(results["summary_metrics"]["CellTypeA"])) == len(results["summary_metrics"]["CellTypeA"]) - 1

    def test_logistic_regression_classifier(self):
        """Tests logistic classifier for auc calculation."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_rfc.load(sc_sim_adata)
        sc.pp.highly_variable_genes(sc_sim_adata)
        adata, results = self.ag_lrc.predict(
            sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42, select_variance_features=False
        )

        assert "augur_score" in adata.obs.columns
        assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.691232, 0.955404, 0.972789])
        assert "feature_importances" in results.keys()

    def test_random_forest_regressor(self):
        """Tests random forest regressor for ccc calculation."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_rfc.load(sc_sim_adata)
        sc.pp.highly_variable_genes(sc_sim_adata)

        with pytest.raises(ValueError):
            self.ag_rfr.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42)

    def test_classifier(self):
        """Test run cross validation with classifier."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_rfc.load(sc_sim_adata)
        sc.pp.highly_variable_genes(sc_sim_adata)
        adata = sc.pp.subsample(sc_sim_adata, n_obs=100, random_state=42, copy=True)

        cv = self.ag_rfc.run_cross_validation(adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
        auc = 0.786412
        assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])

        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_lrc.load(sc_sim_adata)
        sc.pp.highly_variable_genes(sc_sim_adata)
        cv = self.ag_lrc.run_cross_validation(sc_sim_adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
        auc = 0.978673
        assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])

    def test_regressor(self):
        """Test run cross validation with regressor."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_rfc.load(sc_sim_adata)
        cv = self.ag_rfr.run_cross_validation(sc_sim_adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
        ccc = 0.168800
        r2 = 0.149887
        assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10**-5), isclose(cv["mean_r2"], r2, abs_tol=10**-5)])

    def test_subsample(self):
        """Test default, permute and velocity subsampling process."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_rfc.load(sc_sim_adata)
        sc.pp.highly_variable_genes(sc_sim_adata)
        categorical_subsample = self.ag_rfc.draw_subsample(
            adata=sc_sim_adata,
            augur_mode="default",
            subsample_size=20,
            feature_perc=0.3,
            categorical=True,
            random_state=42,
        )
        assert len(categorical_subsample.obs_names) == 40

        non_categorical_subsample = self.ag_rfc.draw_subsample(
            adata=sc_sim_adata,
            augur_mode="default",
            subsample_size=20,
            feature_perc=0.3,
            categorical=False,
            random_state=42,
        )
        assert len(non_categorical_subsample.obs_names) == 20

        permut_subsample = self.ag_rfc.draw_subsample(
            adata=sc_sim_adata,
            augur_mode="permute",
            subsample_size=20,
            feature_perc=0.3,
            categorical=True,
            random_state=42,
        )
        assert (sc_sim_adata.obs.loc[permut_subsample.obs.index, "y_"] != permut_subsample.obs["y_"]).any()

        velocity_subsample = self.ag_rfc.draw_subsample(
            adata=sc_sim_adata,
            augur_mode="velocity",
            subsample_size=20,
            feature_perc=0.3,
            categorical=True,
            random_state=42,
        )
        assert len(velocity_subsample.var_names) == 5505 and len(velocity_subsample.obs_names) == 40

    def test_multiclass(self):
        """Test multiclass evaluation."""
        pass

    def test_select_variance(self):
        """Test select variance implementation."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
        sc_sim_adata = self.ag_rfc.load(sc_sim_adata)
        sc.pp.highly_variable_genes(sc_sim_adata)
        adata = sc_sim_adata[sc_sim_adata.obs["cell_type"] == "CellTypeA"]
        ad = self.ag_rfc.select_variance(adata, var_quantile=0.5, span=0.3, filter_negative_residuals=False)

        assert 3672 == len(ad.var.index[ad.var["highly_variable"]])

    def test_creation(self):
        """Test output of create_estimator."""
        assert isinstance(self.ag_rfr.estimator, RandomForestRegressor)

    def test_params(self):
        """Test parameters."""
        rf_estimator = self.ag_rfr.create_estimator(
            "random_forest_classifier", Params(n_estimators=9, max_depth=10, penalty=13)
        )
        lr_estimator = self.ag_rfr.create_estimator("logistic_regression_classifier", Params(penalty="elasticnet"))
        assert rf_estimator.get_params()["n_estimators"] == 9
        assert rf_estimator.get_params()["max_depth"] == 10
        assert lr_estimator.get_params()["penalty"] == "elasticnet"

        with pytest.raises(TypeError):
            self.ag_rfr.create_estimator("random_forest_regressor", Params(unvalid=10))

    @pytest.mark.skip("Computationally expensive")
    def test_differential_prioritization(self):
        """Test differential prioritization run."""
        sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")

        ag = pt.tl.Augurpy("random_forest_classifier", Params(random_state=42))
        ag.load(sc_sim_adata)

        adata, results1 = ag.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=2)
        adata, results2 = ag.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42)

        a, permut1 = ag.predict(sc_sim_adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=2)
        a, permut2 = ag.predict(sc_sim_adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=42)
        delta = ag.predict_differential_prioritization(results1, results2, permut1, permut2)
        assert not np.isnan(delta["z"]).any()
