import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import pertpy as pt


@pytest.fixture
def assay():
    return AnnData(
        obs=pd.DataFrame(
            {
                "dose_a": [0, 10, 10, 0, 0, 10, 10, 10],
                "dose_b": [0, 0, 0, 20, 20, 20, 20, 20],
                "inhibition": [0, 0.1, 0.3, 0.2, 0.4, 0.44, 0.6, 0.35],
            },
            index=["control", "a_1", "a_2", "b_1", "b_2", "additive", "synergistic", "antagonistic"],
        )
    )


@pytest.mark.parametrize("categorical_doses", [False, True])
def test_bliss_matches_known_scores(assay, categorical_doses):
    if categorical_doses:
        for col in ("dose_a", "dose_b"):
            assay.obs[col] = pd.Categorical(assay.obs[col].astype(str))
    original = assay.obs.copy()

    result = pt.tl.Synergy().score(assay, key_added="bliss")

    assert result is None
    combinations = ["additive", "synergistic", "antagonistic"]
    np.testing.assert_allclose(assay.obs.loc[combinations, "bliss_expected"], [0.44, 0.44, 0.44])
    np.testing.assert_allclose(assay.obs.loc[combinations, "bliss_score"], [0, 0.16, -0.09], atol=1e-15)
    assert assay.obs.loc[original.index[:5], ["bliss_expected", "bliss_score"]].isna().all().all()
    pd.testing.assert_frame_equal(assay.obs[original.columns], original)


@pytest.mark.parametrize("swap_drugs", [False, True])
def test_bliss_matches_doses_and_is_order_independent(assay, rng, swap_drugs):
    extra = pd.DataFrame(
        {"dose_a": [50, 0, 50, 10], "dose_b": [0, 60, 60, 60], "inhibition": [0.5, 0.6, 0.9, 0.84]},
        index=["a_high", "b_high", "both_high", "mixed_doses"],
    )
    obs = pd.concat([assay.obs, extra])
    shuffled = AnnData(obs=obs.iloc[rng.permutation(len(obs))].copy())
    columns = ("dose_b", "dose_a") if swap_drugs else ("dose_a", "dose_b")

    pt.tl.Synergy().score(shuffled, dose_cols=columns, key_added="bliss")

    np.testing.assert_allclose(shuffled.obs.loc[["both_high", "mixed_doses"], "bliss_expected"], [0.8, 0.68])
    np.testing.assert_allclose(shuffled.obs.loc[["both_high", "mixed_doses"], "bliss_score"], [0.1, 0.16])


@pytest.mark.parametrize(("column", "dose"), [("dose_a", 100), ("dose_b", 200), ("dose_a", 10 + 1e-14)])
def test_bliss_requires_exact_single_agent_doses(assay, column, dose):
    extra = pd.DataFrame({"dose_a": [10], "dose_b": [20], "inhibition": [0.5]}, index=["missing_reference"])
    extra[column] = dose
    adata = AnnData(obs=pd.concat([assay.obs, extra]))

    with pytest.warns(UserWarning, match="Cannot score 1 combination"):
        pt.tl.Synergy().score(adata, key_added="bliss")

    assert adata.obs.loc["missing_reference", ["bliss_expected", "bliss_score"]].isna().all()
    assert adata.obs.loc["synergistic", "bliss_score"] == pytest.approx(0.16)


def test_bliss_missing_responses_do_not_block_valid_combinations(assay):
    assay.obs.loc[["control", "a_1", "antagonistic"], "inhibition"] = np.nan
    with pytest.warns(UserWarning, match="Cannot score 1 combination") as warnings:
        pt.tl.Synergy().score(assay, key_added="bliss")
    assert len(warnings) == 1
    combinations = ["additive", "synergistic", "antagonistic"]
    np.testing.assert_allclose(assay.obs.loc[combinations, "bliss_expected"], 0.51)
    np.testing.assert_allclose(assay.obs.loc[combinations, "bliss_score"], [-0.07, 0.09, np.nan])


def test_bliss_custom_columns_and_overwrite(assay):
    assay.obs = assay.obs.rename(columns={"dose_a": "conc_a", "dose_b": "conc_b", "inhibition": "response"})
    scorer = pt.tl.Synergy()
    kwargs = {"dose_cols": ("conc_a", "conc_b"), "key_added": "interaction"}
    scorer.score(assay, response="response", **kwargs)
    assert "bliss_score" not in assay.obs
    assert assay.obs.loc["synergistic", "interaction_score"] == pytest.approx(0.16)

    assay.obs.loc["synergistic", "response"] = 0.4
    scorer.score(assay, response="response", **kwargs)
    assert assay.obs.loc["synergistic", "interaction_score"] == pytest.approx(-0.04)


def test_bliss_default_keys_keep_different_responses(assay):
    scorer = pt.tl.Synergy()
    scorer.score(assay)
    original = assay.obs["inhibition_bliss_score"].copy()
    assay.obs["other_inhibition"] = assay.obs["inhibition"] / 2
    scorer.score(assay, response="other_inhibition")
    pd.testing.assert_series_equal(assay.obs["inhibition_bliss_score"], original)
    assert assay.obs.loc["synergistic", "other_inhibition_bliss_score"] == pytest.approx(0.065)


@pytest.mark.parametrize(
    ("response", "dose_cols"),
    [("bliss_expected", ("dose_a", "dose_b")), ("inhibition", ("bliss_score", "dose_b"))],
)
def test_bliss_result_columns_cannot_overwrite_inputs(assay, response, dose_cols):
    assay.obs = assay.obs.rename(columns={"inhibition": response, "dose_a": dose_cols[0]})
    original = assay.obs.copy()
    with pytest.raises(ValueError, match="must not overwrite"):
        pt.tl.Synergy().score(assay, response=response, dose_cols=dose_cols, key_added="bliss")
    pd.testing.assert_frame_equal(assay.obs, original)


@pytest.mark.parametrize(
    ("inhibition_a", "inhibition_b", "observed", "expected", "score"),
    [(0, 0, 0, 0, 0), (1, 0, 1, 1, 0), (0, 1, 0.8, 1, -0.2), (1, 1, 1, 1, 0)],
)
def test_bliss_boundaries(inhibition_a, inhibition_b, observed, expected, score):
    adata = AnnData(
        obs=pd.DataFrame(
            {"dose_a": [1, 0, 1], "dose_b": [0, 1, 1], "inhibition": [inhibition_a, inhibition_b, observed]}
        )
    )
    pt.tl.Synergy().score(adata, key_added="bliss")
    assert adata.obs["bliss_expected"].iloc[-1] == pytest.approx(expected)
    assert adata.obs["bliss_score"].iloc[-1] == pytest.approx(score)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("dose_a", -1, "non-negative"),
        ("dose_b", np.inf, "finite"),
        ("dose_a", np.nan, "finite"),
        ("inhibition", np.inf, "between 0 and 1"),
        ("inhibition", -0.1, "between 0 and 1"),
        ("inhibition", 1.1, "between 0 and 1"),
    ],
)
def test_bliss_invalid_measurements_do_not_write_results(assay, column, value, message):
    assay.obs[column] = assay.obs[column].astype(float)
    assay.obs.loc["synergistic", column] = value
    with pytest.raises(ValueError, match=message):
        pt.tl.Synergy().score(assay, key_added="bliss")
    assert "bliss_score" not in assay.obs


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "loewe"}, "Only method='bliss'"),
        ({"dose_cols": ("dose_a",)}, "two distinct dose columns"),
        ({"dose_cols": ("dose_a", "dose_a")}, "distinct dose columns"),
        ({"response": "dose_a"}, "separate response column"),
    ],
)
def test_bliss_invalid_options(assay, kwargs, message):
    with pytest.raises(ValueError, match=message):
        pt.tl.Synergy().score(assay, **kwargs)


def test_bliss_requires_combinations(assay):
    singles = assay[:5].copy()
    with pytest.raises(ValueError, match="No combination measurements"):
        pt.tl.Synergy().score(singles)


def test_bliss_requires_named_columns(assay):
    with pytest.raises(KeyError, match="missing"):
        pt.tl.Synergy().score(assay, response="missing")
