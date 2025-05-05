from math import isclose
from pathlib import Path

import numpy as np
import pertpy as pt
import pytest
import scanpy as sc

CWD = Path(__file__).parent.resolve()


ag_rfc = pt.tl.Augur("random_forest_classifier", random_state=42)
ag_lrc = pt.tl.Augur("logistic_regression_classifier", random_state=42)
ag_rfr = pt.tl.Augur("random_forest_regressor", random_state=42)


@pytest.fixture
def adata():
    adata = pt.dt.sc_sim_augur()
    adata = sc.pp.subsample(adata, n_obs=200, copy=True, random_state=10)

    return adata


def test_load(adata):
    """Test if load function creates anndata objects."""
    ag = pt.tl.Augur(estimator="random_forest_classifier")

    loaded_adata = ag.load(adata)
    loaded_df = ag.load(adata.to_df(), meta=adata.obs, cell_type_col="cell_type", label_col="label")

    assert loaded_adata.obs["y_"].equals(loaded_df.obs["y_"]) is True
    assert adata.to_df().equals(loaded_adata.to_df()) is True and adata.to_df().equals(loaded_df.to_df())


def test_random_forest_classifier(adata):
    """Tests random forest for auc calculation."""
    adata = ag_rfc.load(adata)
    sc.pp.highly_variable_genes(adata)
    h_adata, results = ag_rfc.predict(
        adata, n_threads=4, n_subsamples=3, random_state=42, select_variance_features=False
    )

    assert results["CellTypeA"][2]["subsample_idx"] == 2
    assert "augur_score" in h_adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.634920, 0.933484, 0.902494])
    assert "feature_importances" in results
    assert len(set(results["summary_metrics"]["CellTypeA"])) == len(results["summary_metrics"]["CellTypeA"]) - 1


def test_logistic_regression_classifier(adata):
    """Tests logistic classifier for auc calculation."""
    adata = ag_rfc.load(adata)
    sc.pp.highly_variable_genes(adata)
    h_adata, results = ag_lrc.predict(
        adata, n_threads=4, n_subsamples=3, random_state=42, select_variance_features=False
    )

    assert "augur_score" in h_adata.obs.columns
    assert np.allclose(results["summary_metrics"].loc["mean_augur_score"].tolist(), [0.691232, 0.955404, 0.972789])
    assert "feature_importances" in results


def test_random_forest_regressor(adata):
    """Tests random forest regressor for ccc calculation."""
    adata = ag_rfc.load(adata)
    sc.pp.highly_variable_genes(adata)

    with pytest.raises(ValueError):
        ag_rfr.predict(adata, n_threads=4, n_subsamples=3, random_state=42)


def test_classifier(adata):
    """Test run cross validation with classifier."""
    adata = ag_rfc.load(adata)
    sc.pp.highly_variable_genes(adata)
    adata_subsampled = sc.pp.subsample(adata, n_obs=100, random_state=42, copy=True)

    cv = ag_rfc.run_cross_validation(adata_subsampled, subsample_idx=1, folds=3, random_state=42, zero_division=0)
    auc = 0.786412
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])

    cv = ag_lrc.run_cross_validation(adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
    auc = 0.978673
    assert any([isclose(cv["mean_auc"], auc, abs_tol=10**-3)])


def test_regressor(adata):
    """Test run cross validation with regressor."""
    adata = ag_rfc.load(adata)
    cv = ag_rfr.run_cross_validation(adata, subsample_idx=1, folds=3, random_state=42, zero_division=0)
    ccc = 0.168800
    r2 = 0.149887
    assert any([isclose(cv["mean_ccc"], ccc, abs_tol=10**-5), isclose(cv["mean_r2"], r2, abs_tol=10**-5)])


def test_subsample(adata):
    """Test default, permute and velocity subsampling process."""
    adata = ag_rfc.load(adata)
    sc.pp.highly_variable_genes(adata)
    categorical_subsample = ag_rfc.draw_subsample(
        adata=adata,
        augur_mode="default",
        subsample_size=20,
        feature_perc=0.3,
        categorical=True,
        random_state=42,
    )
    assert len(categorical_subsample.obs_names) == 40

    non_categorical_subsample = ag_rfc.draw_subsample(
        adata=adata,
        augur_mode="default",
        subsample_size=20,
        feature_perc=0.3,
        categorical=False,
        random_state=42,
    )
    assert len(non_categorical_subsample.obs_names) == 20

    permut_subsample = ag_rfc.draw_subsample(
        adata=adata,
        augur_mode="permute",
        subsample_size=20,
        feature_perc=0.3,
        categorical=True,
        random_state=42,
    )
    assert (adata.obs.loc[permut_subsample.obs.index, "y_"] != permut_subsample.obs["y_"]).any()

    velocity_subsample = ag_rfc.draw_subsample(
        adata=adata,
        augur_mode="velocity",
        subsample_size=20,
        feature_perc=0.3,
        categorical=True,
        random_state=42,
    )
    assert len(velocity_subsample.var_names) == 5505 and len(velocity_subsample.obs_names) == 40


def test_select_variance(adata):
    """Test select variance implementation."""
    adata = ag_rfc.load(adata)
    sc.pp.highly_variable_genes(adata)
    adata_cell_type = adata[adata.obs["cell_type"] == "CellTypeA"].copy()
    ad = ag_rfc.select_variance(adata_cell_type, var_quantile=0.5, span=0.3, filter_negative_residuals=False)

    assert len(ad.var.index[ad.var["highly_variable"]]) == 3672


@pytest.mark.slow
def test_differential_prioritization():
    """Test differential prioritization run."""
    # Requires the full dataset or it fails because of a lack of statistical power
    adata = pt.dt.sc_sim_augur()
    adata = sc.pp.subsample(adata, n_obs=500, copy=True, random_state=10)
    ag = pt.tl.Augur("logistic_regression_classifier", random_state=42)
    ag.load(adata)

    adata, results1 = ag.predict(adata, n_threads=4, n_subsamples=3, random_state=2)
    adata, results2 = ag.predict(adata, n_threads=4, n_subsamples=3, random_state=42)

    a, permut1 = ag.predict(adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=2)
    a, permut2 = ag.predict(adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=42)
    delta = ag.predict_differential_prioritization(results1, results2, permut1, permut2)
    assert not np.isnan(delta["z"]).any()
