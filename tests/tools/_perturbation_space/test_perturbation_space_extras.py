import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

import pertpy as pt
from pertpy._types import cast_frame


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


@pytest.mark.parametrize("categorical_doses", [False, True])
def test_dose_response(rng, categorical_doses):
    groups, doses = [], []
    for pert in ["control", "drug"]:
        for dose in [0.0] if pert == "control" else [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
            groups += [pert] * 15
            doses += [dose] * 15
    groups = np.array(groups)
    doses = np.array(doses, dtype=float)
    X = np.tile(rng.normal(0, 0.3, (15, 8)), (len(groups) // 15, 1))
    drug_doses = doses[groups == "drug"]
    X[groups == "drug"] += (5 * drug_doses**1.2 / (10**1.2 + drug_doses**1.2))[:, None]
    adata = AnnData(X, obs=pd.DataFrame({"perturbation": groups, "dose": doses}))
    if categorical_doses:
        adata.obs["dose"] = pd.Categorical(adata.obs["dose"].astype(str))
    sc.pp.pca(adata, n_comps=5)

    ps = pt.tl.PseudobulkSpace()
    dose_adata = ps.dose_response(adata, dose_col="dose", metric="euclidean", embedding_key="X_pca")
    assert dose_adata.shape == (6, 5)
    assert dose_adata.obs["dose"].tolist() == [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    assert dose_adata.obs["distance"].is_monotonic_increasing

    ps.fit_dose_response(dose_adata)
    assert dose_adata.obs["hill_ec50"].iloc[0] == pytest.approx(10, rel=1e-4)
    assert dose_adata.obs["hill_ec50_in_range"].all()


def _assay(data: pd.DataFrame) -> AnnData:
    return AnnData(obs=data.reset_index(drop=True).rename(index=str))


def _fits(adata: AnnData) -> pd.DataFrame:
    return cast_frame(adata.obs).drop_duplicates("perturbation").set_index("perturbation")


@pytest.mark.parametrize(("e0", "emax"), [(0.1, 1.8), (1.0, 0.05)])
def test_fit_dose_response(e0, emax):
    doses = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    responses = e0 + (emax - e0) * doses**1.4 / (3.0**1.4 + doses**1.4)
    adata = _assay(pd.DataFrame({"perturbation": "drug", "dose": doses, "distance": responses}))
    pt.tl.PseudobulkSpace().fit_dose_response(adata)
    fit = _fits(adata).loc["drug"]

    assert fit["hill_e0"] == pytest.approx(e0)
    assert fit["hill_emax"] == pytest.approx(emax)
    assert fit["hill_slope"] == pytest.approx(1.4)
    assert fit["hill_ec50"] == pytest.approx(3.0)
    np.testing.assert_allclose(adata.obs["hill_fitted"], responses, rtol=1e-6)


def test_fit_dose_response_standard_error():
    doses = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    responses = doses / (10 + doses) + np.random.default_rng(2026).normal(0, 0.025, len(doses))
    complete = pd.DataFrame({"perturbation": "complete", "dose": doses, "distance": responses})
    limited = complete.loc[complete["dose"] <= 3].assign(perturbation="limited")
    adata = _assay(pd.concat([complete, limited]))

    with pytest.warns(UserWarning, match="Cannot estimate.*'limited'"):
        pt.tl.PseudobulkSpace().fit_dose_response(adata)
    fits = _fits(adata)
    fit = fits.loc["complete"]

    midpoint, slope = fit["hill_ec50"], fit["hill_slope"]
    fraction = doses**slope / (midpoint**slope + doses**slope)
    log_ratio = np.zeros_like(doses)
    np.log(doses / midpoint, out=log_ratio, where=doses > 0)
    sensitivity = (fit["hill_emax"] - fit["hill_e0"]) * fraction * (1 - fraction)
    jacobian = np.column_stack((1 - fraction, fraction, -slope * sensitivity / midpoint, sensitivity * log_ratio))
    residuals = responses - (fit["hill_e0"] + (fit["hill_emax"] - fit["hill_e0"]) * fraction)
    variance = np.sum(residuals**2) / (len(doses) - 4)
    assert fit["hill_ec50_se"] == pytest.approx(
        np.sqrt(np.linalg.inv(jacobian.T @ jacobian)[2, 2] * variance), rel=1e-4
    )
    assert np.isnan(fits.loc["limited", "hill_ec50_se"])


def test_fit_dose_response_unfittable():
    doses = np.array([0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0])
    good = pd.DataFrame({"perturbation": "good", "dose": doses, "distance": doses / (10 + doses)})
    adata = _assay(pd.concat([good, good.assign(perturbation="bad", distance=1.0)]))

    with pytest.warns(UserWarning, match="'bad'.*constant"):
        pt.tl.PseudobulkSpace().fit_dose_response(adata)
    fits = _fits(adata)
    assert fits.loc["good", "hill_ec50"] == pytest.approx(10)
    assert np.isnan(fits.loc["bad", "hill_ec50"])


def test_fit_dose_response_negative_dose():
    adata = _assay(pd.DataFrame({"perturbation": "drug", "dose": [-1, 1, 2, 3], "distance": [0, 1, 2, 3]}))
    with pytest.raises(ValueError, match="non-negative"):
        pt.tl.PseudobulkSpace().fit_dose_response(adata)
