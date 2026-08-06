import numpy as np
import pandas as pd
import pytest

from pertpy.tools._milo_glmm import fit_nb_glmm, has_separation, parse_random_effects, random_effect_matrices


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


def test_has_separation():
    X = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])

    assert not has_separation(np.array([1.0, 2, 3, 4, 5, 6]), X)
    assert has_separation(np.array([0.0, 0, 0, 4, 5, 6]), X)
    assert has_separation(np.zeros(6), X)


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


def test_fit_nb_glmm_offset_shifts_intercept_only(repeated_measures):
    y, X, Z = repeated_measures
    offset = np.full(len(y), np.log(2.0))

    without = fit_nb_glmm(y, X, Z, np.zeros(len(y)))
    with_offset = fit_nb_glmm(y, X, Z, offset)

    assert with_offset.beta[1] == pytest.approx(without.beta[1], abs=1e-3)
    assert with_offset.beta[0] == pytest.approx(without.beta[0] - np.log(2.0), abs=1e-3)
