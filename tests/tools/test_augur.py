from math import isclose
from pathlib import Path

import numpy as np
import pertpy as pt
import pytest
import scanpy as sc
from pertpy.tools._augur import Params
from sklearn.ensemble import RandomForestRegressor

CWD = Path(__file__).parent.resolve()


class TestAugur:
    ag_rfc = pt.tl.Augur("random_forest_classifier", Params(random_state=42))
    ag_lrc = pt.tl.Augur("logistic_regression_classifier", Params(random_state=42))
    ag_rfr = pt.tl.Augur("random_forest_regressor", Params(random_state=42))

    @pytest.fixture
    def adata(self):
        adata = pt.dt.sc_sim_augur()
        adata = sc.pp.subsample(adata, n_obs=200, copy=True, random_state=10)

        return adata

    def test_load(self, adata):
        """Test if load function creates anndata objects."""
        ag = pt.tl.Augur(estimator="random_forest_classifier")

        loaded_adata = ag.load(adata)
        loaded_df = ag.load(adata.to_df(), meta=adata.obs, cell_type_col="cell_type", label_col="label")

        assert loaded_adata.obs["y_"].equals(loaded_df.obs["y_"]) is True
        assert adata.to_df().equals(loaded_adata.to_df()) is True and adata.to_df().equals(loaded_df.to_df())

    def test_random_forest_classifier(self, adata):
        """Tests random forest for auc calculation."""
        adata = self.ag_rfc.load(adata)
        sc.pp.highly_variable_genes(adata)
        h_adata, results = self.ag_rfc.predict(
            adata, n_threads=4, n_subsamples=3, random_state=42, select_variance_features=False
        )

        assert results["CellTypeA"][2]["subsample_idx"] == 2
        assert "augur_score" in h_adata.obs.columns
        assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.634920, 0.933484, 0.902494])
        assert "feature_importances" in results.keys()
        assert len(set(results["summary_metrics"]["CellTypeA"])) == len(results["summary_metrics"]["CellTypeA"]) - 1

    def test_logistic_regression_classifier(self, adata):
        """Tests logistic classifier for auc calculation."""
        adata = self.ag_rfc.load(adata)
        sc.pp.highly_variable_genes(adata)
        h_adata, results = self.ag_lrc.predict(
            adata, n_threads=4, n_subsamples=3, random_state=42, select_variance_features=False
        )

        assert "augur_score" in h_adata.obs.columns
        assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.691232, 0.955404, 0.972789])
        assert "feature_importances" in results.keys()

    def test_random_forest_regressor(self, adata):
        """Tests random forest regressor for ccc calculation."""
        adata = self.ag_rfc.load(adata)
        sc.pp.highly_variable_genes(adata)

        with pytest.raises(ValueError):
            self.ag_rfr.predict(adata, n_threads=4, n_subsamples=3, random_state=42)

    def test_classifier(self, adata):
        """Test run cross validation with classifier."""
        adata = self.ag_rfc.load(adata)
        sc.pp.highly_variable_genes(adata)
        adata_subsampled = sc.pp.subsample(adata, n_obs=100, random_state=42, copy=True)

        cv = self.ag_rfc.run_cross_validation(
            adata_subsampled, subsample_idx=1, folds=3, random_state=42, zero_division=0
        )
        auc = 0.786412
        assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])

        cv = self.ag_lrc.run_cross_validation(adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
        auc = 0.978673
        assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])

    def test_regressor(self, adata):
        """Test run cross validation with regressor."""
        adata = self.ag_rfc.load(adata)
        cv = self.ag_rfr.run_cross_validation(adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
        ccc = 0.168800
        r2 = 0.149887
        assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10**-5), isclose(cv["mean_r2"], r2, abs_tol=10**-5)])

    def test_subsample(self, adata):
        """Test default, permute and velocity subsampling process."""
        adata = self.ag_rfc.load(adata)
        sc.pp.highly_variable_genes(adata)
        categorical_subsample = self.ag_rfc.draw_subsample(
            adata=adata,
            augur_mode="default",
            subsample_size=20,
            feature_perc=0.3,
            categorical=True,
            random_state=42,
        )
        assert len(categorical_subsample.obs_names) == 40

        non_categorical_subsample = self.ag_rfc.draw_subsample(
            adata=adata,
            augur_mode="default",
            subsample_size=20,
            feature_perc=0.3,
            categorical=False,
            random_state=42,
        )
        assert len(non_categorical_subsample.obs_names) == 20

        permut_subsample = self.ag_rfc.draw_subsample(
            adata=adata,
            augur_mode="permute",
            subsample_size=20,
            feature_perc=0.3,
            categorical=True,
            random_state=42,
        )
        assert (adata.obs.loc[permut_subsample.obs.index, "y_"] != permut_subsample.obs["y_"]).any()

        velocity_subsample = self.ag_rfc.draw_subsample(
            adata=adata,
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

    def test_select_variance(self, adata):
        """Test select variance implementation."""
        adata = self.ag_rfc.load(adata)
        sc.pp.highly_variable_genes(adata)
        adata_cell_type = adata[adata.obs["cell_type"] == "CellTypeA"]
        ad = self.ag_rfc.select_variance(adata_cell_type, var_quantile=0.5, span=0.3, filter_negative_residuals=False)

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
    def test_differential_prioritization(self, adata):
        """Test differential prioritization run."""
        ag = pt.tl.Augur("random_forest_classifier", Params(random_state=42))
        ag.load(adata)

        adata, results1 = ag.predict(adata, n_threads=4, n_subsamples=3, random_state=2)
        adata, results2 = ag.predict(adata, n_threads=4, n_subsamples=3, random_state=42)

        a, permut1 = ag.predict(adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=2)
        a, permut2 = ag.predict(adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=42)
        delta = ag.predict_differential_prioritization(results1, results2, permut1, permut2)
        assert not np.isnan(delta["z"]).any()
