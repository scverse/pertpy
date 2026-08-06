import numpy as np
import pandas as pd
import pytest

from pertpy.tools._milo_glmm import (
    between_within_df,
    fit_nb_glmm,
    has_separation,
    parse_random_effects,
    random_effect_matrices,
)


@pytest.mark.parametrize(
    "design,expected",
    [
        ("~ condition + (1 | donor)", ("~ condition", ["donor"])),
        ("~condition+(1|donor)+age", ("~condition+age", ["donor"])),
        ("~ (1 | donor)", ("~ 1", ["donor"])),
        ("~ condition + (1 | donor) + (1 | batch)", ("~ condition", ["donor", "batch"])),
        ("~ condition", ("~ condition", [])),
    ],
)
def test_parse_random_effects(design, expected):
    assert parse_random_effects(design) == expected


def test_parse_random_effects_rejects_invalid_syntax():
    with pytest.raises(ValueError, match="invalid formula for random effects"):
        parse_random_effects("~ condition + (donor | condition)")


def test_random_effect_matrices_are_indicators():
    obs = pd.DataFrame({"donor": ["A", "B", "A", "C"]})
    (name, Z), *rest = random_effect_matrices(obs, ["donor"])

    assert not rest
    assert name == "donor"
    assert Z.shape == (4, 3)
    assert np.array_equal(Z.sum(axis=1), np.ones(4))

    with pytest.raises(ValueError, match="not a column"):
        random_effect_matrices(obs, ["missing"])


def test_random_effect_matrices_rejects_single_level():
    obs = pd.DataFrame({"batch": ["A"] * 10})
    with pytest.raises(ValueError, match="single level"):
        random_effect_matrices(obs, ["batch"])


def test_fit_nb_glmm_survives_an_unfactorisable_variance(monkeypatch):
    """One pathological neighbourhood must not abort a run over thousands of them."""
    rng = np.random.default_rng(0)
    y = rng.poisson(30, 20).astype(float)
    condition = np.tile([0.0, 1.0], 10)
    X = np.column_stack([np.ones(20), condition])
    Z = [("donor", pd.get_dummies(pd.Categorical(np.repeat(np.arange(5), 4))).to_numpy(dtype=float))]

    def explode(*args, **kwargs):
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr("pertpy.tools._milo_glmm.cho_factor", explode)
    fit = fit_nb_glmm(y, X, Z, np.zeros(20))

    assert not fit.converged
    assert np.isnan(fit.beta).all()
    assert np.isnan(fit.se).all()


def test_has_separation():
    X = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])

    assert not has_separation(np.array([1.0, 2, 3, 4, 5, 6]), X)
    assert has_separation(np.array([0.0, 0, 0, 4, 5, 6]), X)
    assert has_separation(np.zeros(6), X)


def test_between_within_df():
    donor = np.repeat(np.arange(20), 8)
    Z = [("donor", pd.get_dummies(pd.Categorical(donor)).to_numpy(dtype=float))]
    intercept = np.ones(160)

    between = np.column_stack([intercept, np.repeat(np.tile([0.0, 1.0], 10), 8)])
    assert between_within_df(between, Z) == 18

    within = np.column_stack([intercept, np.tile([0.0, 1.0], 80)])
    assert between_within_df(within, Z) == 139


@pytest.fixture
def repeated_measures():
    """Counts from a negative binomial mixed model with a known effect and donor variance."""
    rng = np.random.default_rng(0)
    n_donor, per_donor = 30, 6
    donor = np.repeat(np.arange(n_donor), per_donor)
    condition = np.tile([0.0, 1.0], n_donor * per_donor // 2)
    u = rng.normal(0, np.sqrt(0.25), n_donor)
    mu = np.exp(3.0 + 1.2 * condition + u[donor])
    y = rng.negative_binomial(10, 10 / (10 + mu)).astype(float)

    X = np.column_stack([np.ones_like(condition), condition])
    Z = [("donor", pd.get_dummies(pd.Categorical(donor)).to_numpy(dtype=float))]
    return y, X, Z


def test_fit_nb_glmm_recovers_parameters(repeated_measures):
    y, X, Z = repeated_measures
    fit = fit_nb_glmm(y, X, Z, np.zeros(len(y)))

    assert fit.converged
    assert abs(fit.beta[1] - 1.2) < 2 * fit.se[1]
    assert abs(fit.beta[0] - 3.0) < 2 * fit.se[0]
    assert 0.1 < fit.sigma[0] < 0.6
    assert fit.dispersion == pytest.approx(0.1, abs=0.1)


def test_fit_nb_glmm_ignores_random_effect_when_absent(repeated_measures):
    """Without between-donor variance the variance component collapses towards zero."""
    _, X, Z = repeated_measures
    rng = np.random.default_rng(1)
    mu = np.exp(3.0 + 1.2 * X[:, 1])
    y = rng.negative_binomial(10, 10 / (10 + mu)).astype(float)

    fit = fit_nb_glmm(y, X, Z, np.zeros(len(y)))

    assert fit.converged
    assert fit.sigma[0] < 0.1


def test_fit_nb_glmm_two_random_effects():
    """Two crossed random intercepts each recover their own variance component."""
    rng = np.random.default_rng(5)
    n_donor, n_batch, per_donor = 20, 10, 10
    n = n_donor * per_donor
    donor = np.repeat(np.arange(n_donor), per_donor)
    batch = np.tile(np.arange(n_batch), n // n_batch)
    condition = np.tile([0.0, 1.0], n // 2)
    X = np.column_stack([np.ones(n), condition])
    Z = [
        ("donor", pd.get_dummies(pd.Categorical(donor)).to_numpy(dtype=float)),
        ("batch", pd.get_dummies(pd.Categorical(batch)).to_numpy(dtype=float)),
    ]

    donor_var, batch_var = 0.5, 0.8
    estimates = []
    for _ in range(10):
        u_donor = rng.normal(0, np.sqrt(donor_var), n_donor)
        u_batch = rng.normal(0, np.sqrt(batch_var), n_batch)
        mu = np.exp(3.0 + 1.0 * condition + u_donor[donor] + u_batch[batch])
        y = rng.negative_binomial(8, 8 / (8 + mu)).astype(float)
        fit = fit_nb_glmm(y, X, Z, np.zeros(n))
        assert fit.converged
        assert fit.sigma.shape == (2,)
        estimates.append(np.concatenate([fit.sigma, fit.beta[1:]]))

    mean = np.mean(estimates, axis=0)
    assert mean[0] == pytest.approx(donor_var, abs=0.3)
    assert mean[1] == pytest.approx(batch_var, abs=0.3)
    assert mean[2] == pytest.approx(1.0, abs=0.2)


def test_fit_nb_glmm_offset_shifts_intercept_only(repeated_measures):
    y, X, Z = repeated_measures
    offset = np.full(len(y), np.log(2.0))

    without = fit_nb_glmm(y, X, Z, np.zeros(len(y)))
    with_offset = fit_nb_glmm(y, X, Z, offset)

    assert with_offset.beta[1] == pytest.approx(without.beta[1], abs=1e-3)
    assert with_offset.beta[0] == pytest.approx(without.beta[0] - np.log(2.0), abs=1e-3)
