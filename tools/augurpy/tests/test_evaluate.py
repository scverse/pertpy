from math import isclose
from pathlib import Path

import numpy as np
import pytest
import scanpy as sc

from augurpy.estimator import Params, create_estimator
from augurpy.evaluate import draw_subsample, predict, run_cross_validation, select_variance
from augurpy.read_load import load

CWD = Path(__file__).parent.resolve()

sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
sc_sim_adata = load(sc_sim_adata)
sc.pp.highly_variable_genes(sc_sim_adata)

bhattacher = sc.read_h5ad(f"{CWD}/bhattacher.h5ad")
bhattacher_adata = load(bhattacher)

# estimators
rf_classifier = create_estimator("random_forest_classifier", Params(random_state=42))
lr_classifier = create_estimator("logistic_regression_classifier", Params(random_state=42))
rf_regressor = create_estimator("random_forest_regressor", Params(random_state=42))


def test_random_forest_classifier():
    """Tests random forest for auc calculation."""
    adata, results = predict(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=rf_classifier, random_state=42)
    assert results["CellTypeA"][2]["subsample_idx"] == 2
    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.521730, 0.783635, 0.868858])
    assert "feature_importances" in results.keys()
    assert len(set(results["summary_metrics"]["CellTypeA"])) == len(results["summary_metrics"]["CellTypeA"]) - 1


def test_logistic_regression_classifier():
    """Tests logistic classifier for auc calculation."""
    adata, results = predict(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=lr_classifier, random_state=42)
    assert "augur_score" in adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.610733, 0.927815, 0.969765])
    assert "feature_importances" in results.keys()


def test_random_forest_regressor():
    """Tests random forest regressor for ccc calculation."""
    with pytest.raises(ValueError):
        predict(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=rf_regressor, random_state=42)


# Test cross validation
def test_classifier(adata=sc_sim_adata):
    """Test run cross validation with classifier."""
    adata = sc.pp.subsample(adata, n_obs=100, random_state=42, copy=True)

    cv = run_cross_validation(adata, rf_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.766289
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])

    cv = run_cross_validation(adata, lr_classifier, subsample_idx=1, folds=3, random_state=42)
    auc = 0.965745
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10 ** -5)])


def test_regressor(adata=sc_sim_adata):
    """Test run cross validation with regressor."""
    cv = run_cross_validation(adata, rf_regressor, subsample_idx=1, folds=3, random_state=42)
    ccc = 0.231356
    r2 = 0.206195
    assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10 ** -5), isclose(cv["mean_r2"], r2, abs_tol=10 ** -5)])


def test_subsample(adata=sc_sim_adata):
    """Test default, permute and velocity subsampling process."""
    categorical_subsample = draw_subsample(
        adata=adata, augur_mode="default", subsample_size=20, feature_perc=0.3, categorical=True, random_state=42
    )
    assert len(categorical_subsample.obs_names) == 40

    non_categorical_subsample = draw_subsample(
        adata=adata, augur_mode="default", subsample_size=20, feature_perc=0.3, categorical=False, random_state=42
    )
    assert len(non_categorical_subsample.obs_names) == 20

    permut_subsample = draw_subsample(
        adata=adata, augur_mode="permute", subsample_size=20, feature_perc=0.3, categorical=True, random_state=42
    )
    assert (sc_sim_adata.obs.loc[permut_subsample.obs.index, "y_"] != permut_subsample.obs["y_"]).any()

    velocity_subsample = draw_subsample(
        adata=adata, augur_mode="velocity", subsample_size=20, feature_perc=0.3, categorical=True, random_state=42
    )
    assert len(velocity_subsample.var_names) == 5908 and len(velocity_subsample.obs_names) == 40


def test_multiclass():
    """Test multiclass evaluation."""
    adata = bhattacher_adata[bhattacher_adata.obs["cell_type"] == "Astro"]
    rf_classifier = create_estimator("random_forest_classifier")

    a, bhattacher_results = predict(adata, n_threads=1, classifier=rf_classifier, random_state=None)

    # check that metric values are all different (except for augur and auc which are the same.)
    assert (
        len(set(bhattacher_results["summary_metrics"]["Astro"]))
        == len(bhattacher_results["summary_metrics"]["Astro"]) - 1
    )


def test_select_variance():
    """Test select variance implementation."""
    adata = bhattacher_adata[bhattacher_adata.obs["cell_type"] == "Astro"]
    ad = select_variance(adata, var_quantile=0.5, span=0.6, filter_negative_residuals=False)
    assert 7122 == len(ad.var.index[ad.var["highly_variable"]])
