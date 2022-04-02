from math import isclose
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

import pertpy as pt
from pertpy.tools._augurpy import Params

CWD = Path(__file__).parent.resolve()

ag_rfc = pt.tl.Augurpy("random_forest_classifier", Params(random_state=42))
ag_lrc = pt.tl.Augurpy("logistic_regression_classifier", Params(random_state=42))
ag_rfr = pt.tl.Augurpy("random_forest_regressor", Params(random_state=42))


def test_random_forest_classifier():
    """Tests random forest for auc calculation."""
    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_rfc.load(sc_sim_adata)
    sc.pp.highly_variable_genes(sc_sim_adata)
    adata, results = ag_rfc.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42)

    assert results["CellTypeA"][2]["subsample_idx"] == 2
    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.521730, 0.783635, 0.868858])
    assert "feature_importances" in results.keys()
    assert len(set(results["summary_metrics"]["CellTypeA"])) == len(results["summary_metrics"]["CellTypeA"]) - 1


def test_logistic_regression_classifier():
    """Tests logistic classifier for auc calculation."""
    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_rfc.load(sc_sim_adata)
    sc.pp.highly_variable_genes(sc_sim_adata)
    adata, results = ag_lrc.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42)

    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.610733, 0.927815, 0.969765])
    assert "feature_importances" in results.keys()


def test_random_forest_regressor():
    """Tests random forest regressor for ccc calculation."""
    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_rfc.load(sc_sim_adata)
    sc.pp.highly_variable_genes(sc_sim_adata)

    with pytest.raises(ValueError):
        ag_rfr.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42)


# Test cross validation
def test_classifier():
    """Test run cross validation with classifier."""
    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_rfc.load(sc_sim_adata)
    sc.pp.highly_variable_genes(sc_sim_adata)
    adata = sc.pp.subsample(sc_sim_adata, n_obs=100, random_state=42, copy=True)

    cv = ag_rfc.run_cross_validation(adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
    auc = 0.766289
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])

    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_lrc.load(sc_sim_adata)
    sc.pp.highly_variable_genes(sc_sim_adata)
    cv = ag_lrc.run_cross_validation(sc_sim_adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
    auc = 0.991796
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])


def test_regressor():
    """Test run cross validation with regressor."""
    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_rfc.load(sc_sim_adata)
    cv = ag_rfr.run_cross_validation(sc_sim_adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
    ccc = 0.231356
    r2 = 0.206195
    assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10**-5), isclose(cv["mean_r2"], r2, abs_tol=10**-5)])


def test_subsample():
    """Test default, permute and velocity subsampling process."""
    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_rfc.load(sc_sim_adata)
    sc.pp.highly_variable_genes(sc_sim_adata)
    categorical_subsample = ag_rfc.draw_subsample(
        adata=sc_sim_adata, augur_mode="default", subsample_size=20, feature_perc=0.3, categorical=True, random_state=42
    )
    assert len(categorical_subsample.obs_names) == 40

    non_categorical_subsample = ag_rfc.draw_subsample(
        adata=sc_sim_adata,
        augur_mode="default",
        subsample_size=20,
        feature_perc=0.3,
        categorical=False,
        random_state=42,
    )
    assert len(non_categorical_subsample.obs_names) == 20

    permut_subsample = ag_rfc.draw_subsample(
        adata=sc_sim_adata, augur_mode="permute", subsample_size=20, feature_perc=0.3, categorical=True, random_state=42
    )
    assert (sc_sim_adata.obs.loc[permut_subsample.obs.index, "y_"] != permut_subsample.obs["y_"]).any()

    velocity_subsample = ag_rfc.draw_subsample(
        adata=sc_sim_adata,
        augur_mode="velocity",
        subsample_size=20,
        feature_perc=0.3,
        categorical=True,
        random_state=42,
    )
    assert len(velocity_subsample.var_names) == 5908 and len(velocity_subsample.obs_names) == 40


def test_multiclass():
    """Test multiclass evaluation."""
    pass


def test_select_variance():
    """Test select variance implementation."""
    sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
    sc_sim_adata = ag_rfc.load(sc_sim_adata)
    sc.pp.highly_variable_genes(sc_sim_adata)
    adata = sc_sim_adata[sc_sim_adata.obs["cell_type"] == "CellTypeA"]
    ad = ag_rfc.select_variance(adata, var_quantile=0.5, span=0.3, filter_negative_residuals=False)

    assert 4871 == len(ad.var.index[ad.var["highly_variable"]])
