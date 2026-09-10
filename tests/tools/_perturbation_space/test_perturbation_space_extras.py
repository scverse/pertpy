import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

import pertpy as pt


@pytest.fixture
def adata(rng):
    labels = np.array(["control", "A", "B"]).repeat(20)
    centers = {"control": 0.0, "A": 5.0, "B": -5.0}
    X = np.vstack([rng.normal(centers[label], 0.3, size=8) for label in labels])
    adata = AnnData(X, obs=pd.DataFrame({"perturbation": labels}))
    sc.pp.pca(adata, n_comps=5)
    return adata


def test_distance_space(adata):
    ds = pt.tl.DistanceSpace()
    ds_adata = ds.compute(adata, metric="euclidean", embedding_key="X_pca")

    assert ds_adata.shape == (3, 3)
    assert "distances" in ds_adata.obsp
    np.testing.assert_allclose(np.diag(ds_adata.obsp["distances"]), 0.0, atol=1e-6)
    # feeds directly into clustering
    clustered = pt.tl.KMeansSpace().compute(ds_adata, n_clusters=3, copy=True, random_state=0)
    assert clustered.obs["k-means"].nunique() == 3


def test_nearest_perturbations(adata):
    ds_adata = pt.tl.DistanceSpace().compute(adata, metric="euclidean", embedding_key="X_pca")
    neighbors = pt.tl.DistanceSpace().nearest_perturbations(ds_adata, "A", n_neighbors=2)
    assert list(neighbors.index) == ["control", "B"]  # control is closer to A than B


def test_embedding_space(adata):
    embedding = pd.DataFrame(np.eye(3), index=["A", "B", "control"])
    es_adata = pt.tl.EmbeddingSpace().compute(adata, embedding, target_col="perturbation")
    assert es_adata.shape == (3, 3)
    assert set(es_adata.obs_names) == {"A", "B", "control"}

    with pytest.raises(ValueError, match="No overlap"):
        pt.tl.EmbeddingSpace().compute(adata, pd.DataFrame(np.eye(2), index=["X", "Y"]))


def test_evaluate_combinations(rng):
    dim = 6
    eff = {"A": rng.normal(size=dim), "B": rng.normal(size=dim), "C": rng.normal(size=dim)}
    rows = {
        "control": np.zeros(dim),
        "A": eff["A"],
        "B": eff["B"],
        "C": eff["C"],
        "A+B": eff["A"] + eff["B"],  # perfectly additive
        "A+C": eff["A"] + eff["C"] + rng.normal(0, 3, dim),  # interaction
    }
    ps_adata = AnnData(X=np.vstack(list(rows.values())))
    ps_adata.obs_names = list(rows.keys())
    ps_adata.obs["perturbation"] = pd.Categorical(list(rows.keys()))

    result = pt.tl.PseudobulkSpace().evaluate_combinations(ps_adata, reference_key="control", metric="pearson")
    assert set(result.index) == {"A+B", "A+C"}
    assert result.loc["A+B", "distance"] < result.loc["A+C", "distance"]
    np.testing.assert_allclose(result.loc["A+B", "distance"], 0.0, atol=1e-6)


def test_dose_response():
    rng = np.random.default_rng(0)
    groups, doses = [], []
    for pert in ["control", "drug"]:
        for dose in [0.0] if pert == "control" else [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
            groups += [pert] * 15
            doses += [dose] * 15
    groups = np.array(groups)
    doses = np.array(doses, dtype=float)
    X = rng.normal(0, 0.3, (len(groups), 8))
    drug_doses = doses[groups == "drug"]
    X[groups == "drug"] += (5 * drug_doses**1.2 / (10**1.2 + drug_doses**1.2))[:, None]
    adata = AnnData(X, obs=pd.DataFrame({"perturbation": groups, "dose": doses}))
    sc.pp.pca(adata, n_comps=5)

    curves = pt.tl.PseudobulkSpace().dose_response(adata, dose_col="dose", metric="euclidean", embedding_key="X_pca")
    drug = curves[curves["perturbation"] == "drug"].sort_values("dose")
    assert drug["distance"].is_monotonic_increasing

    fits = pt.tl.PseudobulkSpace().fit_dose_response(curves)
    assert fits.loc[0, "ec50"] == pytest.approx(10, rel=0.2)
    assert fits.loc[0, "r_squared"] > 0.99
    assert fits.loc[0, "midpoint_in_range"]


@pytest.mark.parametrize(
    ("response_type", "e0", "emax", "midpoint", "midpoint_col"),
    [("effect", 0.1, 1.8, 3.0, "ec50"), ("inhibition", 1.0, 0.05, 8.0, "ic50")],
)
@pytest.mark.parametrize("response_scale", [1e-8, 1.0, 1e8])
def test_fit_dose_response(*, response_type, e0, emax, midpoint, midpoint_col, response_scale):
    e0, emax = e0 * response_scale, emax * response_scale
    doses = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    hill_coefficient = 1.4
    fraction = doses**hill_coefficient / (midpoint**hill_coefficient + doses**hill_coefficient)
    first = pd.DataFrame(
        {
            "compound": "drug_a",
            "concentration": doses,
            "response": e0 + (emax - e0) * fraction,
        }
    )
    second_midpoint = midpoint * 2
    second_fraction = doses**hill_coefficient / (second_midpoint**hill_coefficient + doses**hill_coefficient)
    second = first.assign(compound="drug_b", response=e0 + (emax - e0) * second_fraction)
    data = pd.concat([first, second], ignore_index=True)

    fits = (
        pt.tl.PseudobulkSpace()
        .fit_dose_response(
            data,
            perturbation_col="compound",
            dose_col="concentration",
            response_col="response",
            response_type=response_type,
        )
        .set_index("compound")
    )

    assert midpoint_col in fits
    assert f"{midpoint_col}_standard_error" in fits
    assert {"e0", "emax", "hill_coefficient", "r_squared", "midpoint_in_range"} <= set(fits)
    assert fits.loc["drug_a", "e0"] == pytest.approx(e0, rel=1e-6, abs=0)
    assert fits.loc["drug_a", "emax"] == pytest.approx(emax, rel=1e-6, abs=0)
    assert fits.loc["drug_a", "hill_coefficient"] == pytest.approx(hill_coefficient)
    assert fits.loc["drug_a", midpoint_col] == pytest.approx(midpoint)
    assert fits.loc["drug_b", midpoint_col] == pytest.approx(second_midpoint)
    assert fits.loc["drug_a", "r_squared"] == pytest.approx(1)
    assert np.isfinite(fits[f"{midpoint_col}_standard_error"]).all()


@pytest.mark.parametrize("response_scale", [1.0, 100.0])
def test_fit_dose_response_rank_deficient(response_scale):
    doses = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    responses = doses / (10 + doses) + np.random.default_rng(2026).normal(0, 0.025, len(doses))
    complete = pd.DataFrame({"perturbation": "complete", "dose": doses, "distance": response_scale * responses})
    limited = complete.loc[complete["dose"] <= 3].assign(perturbation="limited")

    with pytest.warns(UserWarning, match="Cannot estimate.*'limited'"):
        fits = pt.tl.PseudobulkSpace().fit_dose_response(pd.concat([complete, limited])).set_index("perturbation")

    assert fits.loc["complete", "ec50"] == pytest.approx(10, rel=0.1)
    assert np.isfinite(fits.loc["complete", "ec50_standard_error"])
    # A high R-squared and an in-range midpoint do not expose this poorly determined curve.
    assert fits.loc["limited", "r_squared"] > 0.98
    assert fits.loc["limited", "midpoint_in_range"]
    assert np.isfinite(fits.loc["limited", "ec50"])
    assert np.isnan(fits.loc["limited", "ec50_standard_error"])


@pytest.mark.parametrize("response_scale", [1e-8, 1.0, 1e8])
def test_fit_dose_response_standard_error(response_scale):
    doses = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    responses = doses / (10 + doses) + np.random.default_rng(2026).normal(0, 0.025, len(doses))
    responses *= response_scale
    data = pd.DataFrame({"perturbation": "drug", "dose": doses, "distance": responses})
    fit = pt.tl.PseudobulkSpace().fit_dose_response(data).iloc[0]

    # Independent derivatives with respect to EC50 itself, rather than the fitted log(EC50).
    midpoint, slope = fit["ec50"], fit["hill_coefficient"]
    fraction = doses**slope / (midpoint**slope + doses**slope)
    log_ratio = np.zeros_like(doses)
    np.log(doses / midpoint, out=log_ratio, where=doses > 0)
    sensitivity = (fit["emax"] - fit["e0"]) * fraction * (1 - fraction)
    jacobian = np.column_stack((1 - fraction, fraction, -slope * sensitivity / midpoint, sensitivity * log_ratio))
    residuals = responses - (fit["e0"] + (fit["emax"] - fit["e0"]) * fraction)
    variance = np.sum(residuals**2) / (len(doses) - 4)
    expected_error = np.sqrt(np.linalg.inv(jacobian.T @ jacobian)[2, 2] * variance)
    assert fit["ec50_standard_error"] == pytest.approx(expected_error, rel=1e-4)


def test_fit_dose_response_insufficient_dof():
    doses = np.array([0.0, 1.0, 10.0, 100.0])
    data = pd.DataFrame({"perturbation": "drug", "dose": doses, "distance": doses / (10 + doses)})
    with pytest.warns(UserWarning, match="Cannot estimate.*'drug'"):
        fit = pt.tl.PseudobulkSpace().fit_dose_response(data).iloc[0]
    assert fit["ec50"] == pytest.approx(10)
    assert np.isnan(fit["ec50_standard_error"])


@pytest.mark.parametrize(
    ("data", "kwargs", "match"),
    [
        (pd.DataFrame(), {}, "Columns"),
        (
            pd.DataFrame(
                {"perturbation": ["drug", "drug", None, "drug"], "dose": [0, 1, 2, 3], "distance": [0, 1, 2, 3]}
            ),
            {},
            "Perturbation labels",
        ),
        (
            pd.DataFrame({"perturbation": ["drug"] * 4, "dose": [-1, 1, 2, 3], "distance": [0, 1, 2, 3]}),
            {},
            "non-negative",
        ),
        (
            pd.DataFrame({"perturbation": ["drug"] * 3, "dose": [0, 1, 2], "distance": [0, 1, 2]}),
            {},
            "four distinct",
        ),
        (
            pd.DataFrame({"perturbation": ["drug"] * 4, "dose": [0, 1, 2, 3], "distance": [1, 1, 1, 1]}),
            {},
            "constant response",
        ),
        (
            pd.DataFrame({"perturbation": ["drug"] * 4, "dose": [0, 1, 2, 3], "distance": [0, 1, 2, 3]}),
            {"response_type": "unknown"},
            "response_type",
        ),
    ],
)
def test_fit_dose_response_validation(data, kwargs, match):
    with pytest.raises(ValueError, match=match):
        pt.tl.PseudobulkSpace().fit_dose_response(data, **kwargs)
