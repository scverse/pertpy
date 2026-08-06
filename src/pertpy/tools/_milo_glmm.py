from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

if TYPE_CHECKING:
    from collections.abc import Sequence

_RANDOM_EFFECT = re.compile(r"\(\s*1\s*\|\s*([^)]+?)\s*\)")


def parse_random_effects(design: str) -> tuple[str, list[str]]:
    """Split a formula into its fixed effects part and the variables entering as random intercepts.

    Random intercepts follow the ``(1 | variable)`` syntax of lme4 and R Milo.

    Returns:
        The formula with the random effect terms removed and the random intercept variables.
    """
    if re.search(r"\|", _RANDOM_EFFECT.sub("", design)):
        raise ValueError(f"{design!r} is an invalid formula for random effects. Use the '(1 | variable)' format.")

    random_effects = [match.group(1).strip() for match in _RANDOM_EFFECT.finditer(design)]
    fixed = _RANDOM_EFFECT.sub("", design)
    fixed = re.sub(r"\+\s*(?=\+|$)", "", fixed).strip().rstrip("+").strip()
    if fixed in {"", "~"}:
        fixed = "~ 1"
    return fixed, random_effects


def random_effect_matrices(obs: pd.DataFrame, random_effects: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    """Build one indicator matrix of shape samples x levels per random intercept variable."""
    matrices = []
    for variable in random_effects:
        if variable not in obs.columns:
            raise ValueError(f"Random effect variable {variable!r} is not a column of the sample metadata.")
        dummies = pd.get_dummies(obs[variable].astype("category"), drop_first=False)
        matrices.append((variable, dummies.to_numpy(dtype=float)))
    return matrices


class GLMMFit(NamedTuple):
    """Result of fitting a negative binomial GLMM to the counts of a single neighbourhood."""

    beta: np.ndarray
    se: np.ndarray
    sigma: np.ndarray
    dispersion: float
    loglik: float
    converged: bool


def _poisson_means(y: np.ndarray, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Fit a Poisson GLM with a log link by iteratively reweighted least squares."""
    mu = np.maximum(y.astype(float), 0.1)
    for _ in range(25):
        working = np.log(mu) - offset + (y - mu) / mu
        coef, *_ = np.linalg.lstsq(X * mu[:, None] ** 0.5, working * mu**0.5, rcond=None)
        new_mu = np.exp(np.clip(offset + X @ coef, -30, 30))
        if np.allclose(new_mu, mu, rtol=1e-6):
            return new_mu
        mu = new_mu
    return mu


def _dispersion_from_means(y: np.ndarray, mu: np.ndarray, df: int) -> float:
    """Solve for the dispersion at which the negative binomial Pearson statistic equals its degrees of freedom."""
    squared_error = (y - mu) ** 2

    def pearson(dispersion: float) -> float:
        return float(np.sum(squared_error / (mu + dispersion * mu**2)) - df)

    if pearson(0.0) <= 0:
        return 0.0
    upper = 1.0
    while pearson(upper) > 0 and upper < 1e6:
        upper *= 10
    return float(brentq(pearson, 0.0, upper)) if pearson(upper) <= 0 else 1e6


def has_separation(y: np.ndarray, X: np.ndarray) -> bool:
    """Check whether the counts are completely separated by a column of the model matrix.

    A neighbourhood in which every sample of one group has zero counts has no finite maximum likelihood estimate, so it is reported as not converged instead of as an arbitrarily large fold change.
    """
    if not np.any(y > 0):
        return True
    return any(
        not np.any(y[mask] > 0) or not np.any(y[~mask] > 0)
        for mask in (X[:, column] != 0 for column in range(X.shape[1]))
        if 0 < mask.sum() < len(y)
    )


def fit_nb_glmm(
    y: np.ndarray,
    X: np.ndarray,
    random_effects: Sequence[tuple[str, np.ndarray]],
    offset: np.ndarray,
    *,
    dispersion: float | None = None,
    reml: bool = True,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> GLMMFit:
    """Fit a negative binomial mixed model with random intercepts by pseudo-likelihood.

    Each iteration linearises the negative binomial likelihood into a working response, solves the resulting weighted mixed model for the fixed effects and the best linear unbiased predictors, and updates the variance components by Fisher scoring, as R Milo's pseudo-likelihood solver does.

    Args:
        y: Counts of one neighbourhood across samples.
        X: Fixed effects model matrix.
        random_effects: Indicator matrix per random intercept variable.
        offset: Log offset per sample.
        dispersion: Negative binomial dispersion. Estimated from the data if None.
        reml: Whether to estimate the variance components by restricted maximum likelihood rather than maximum likelihood.
        max_iter: Maximum number of pseudo-likelihood iterations.
        tol: Convergence tolerance on the fixed effects and variance components.
    """
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    zz = [Z @ Z.T for _, Z in random_effects]
    residual_df = max(n - p, 1)

    def pseudo_likelihood(dispersion: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, bool]:
        size = np.inf if dispersion <= 0 else 1.0 / dispersion
        beta, *_ = np.linalg.lstsq(X, np.log(y + 1) - offset, rcond=None)
        start = np.log(y + 1) - offset - X @ beta
        sigma = np.full(len(random_effects), max(float(start @ start) / residual_df, 1e-3))
        u = [np.zeros(Z.shape[1]) for _, Z in random_effects]
        converged = False

        for _ in range(max_iter):
            eta = offset + X @ beta + sum((Z @ u_k for (_, Z), u_k in zip(random_effects, u, strict=True)), np.zeros(n))
            mu = np.exp(np.clip(eta, -30, 30))
            weights = np.maximum(mu if np.isinf(size) else mu / (1.0 + mu / size), 1e-8)
            working = eta - offset + (y - mu) / mu

            V = sum((s * m for s, m in zip(sigma, zz, strict=True)), np.diag(1.0 / weights))
            V_inv = np.linalg.pinv(V)
            xtvx_inv = np.linalg.pinv(X.T @ V_inv @ X)
            new_beta = xtvx_inv @ X.T @ V_inv @ working
            projection = V_inv - V_inv @ X @ xtvx_inv @ X.T @ V_inv if reml else V_inv

            resid = working - X @ new_beta
            new_u = [s * (Z.T @ (V_inv @ resid)) for s, (_, Z) in zip(sigma, random_effects, strict=True)]

            projected = projection @ resid
            moments = [projection @ m for m in zz]
            score = np.array(
                [
                    -0.5 * np.trace(m) + 0.5 * float(projected @ zz_k @ projected)
                    for m, zz_k in zip(moments, zz, strict=True)
                ]
            )
            information = np.array([[0.5 * float(np.sum(a * b.T)) for b in moments] for a in moments])
            new_sigma = np.maximum(sigma + np.linalg.pinv(information) @ score, 1e-8)

            delta = max(np.max(np.abs(new_beta - beta)), np.max(np.abs(new_sigma - sigma)))
            beta, sigma, u = new_beta, new_sigma, new_u
            if delta < tol:
                converged = True
                break

        eta = offset + X @ beta + sum((Z @ u_k for (_, Z), u_k in zip(random_effects, u, strict=True)), np.zeros(n))
        return beta, sigma, u, np.exp(np.clip(eta, -30, 30)), converged

    if dispersion is None:
        # The fixed effects only estimate absorbs part of the random effect variance, so refine it once the
        # neighbourhood has been fitted with its random effects.
        dispersion = _dispersion_from_means(y, _poisson_means(y, X, offset), residual_df)
        _, _, _, fitted_mean, _ = pseudo_likelihood(dispersion)
        dispersion = _dispersion_from_means(y, fitted_mean, residual_df)

    beta, sigma, u, mu, converged = pseudo_likelihood(dispersion)

    size = np.inf if dispersion <= 0 else 1.0 / dispersion
    weights = np.maximum(mu if np.isinf(size) else mu / (1.0 + mu / size), 1e-8)
    working = np.log(mu) - offset + (y - mu) / mu
    V = sum((s * m for s, m in zip(sigma, zz, strict=True)), np.diag(1.0 / weights))
    V_inv = np.linalg.pinv(V)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(X.T @ V_inv @ X)), 0))

    resid = working - X @ beta
    sign, logdet = np.linalg.slogdet(V)
    loglik = -0.5 * (logdet + float(resid @ V_inv @ resid) + n * np.log(2 * np.pi)) if sign > 0 else np.nan
    if reml and sign > 0:
        loglik -= 0.5 * np.linalg.slogdet(X.T @ V_inv @ X)[1]

    return GLMMFit(
        beta=beta, se=se, sigma=sigma, dispersion=float(dispersion), loglik=float(loglik), converged=converged
    )


def fit_nb_glmm_nhoods(
    counts: np.ndarray,
    X: np.ndarray,
    random_effects: Sequence[tuple[str, np.ndarray]],
    offset: np.ndarray,
    *,
    reml: bool = True,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """Fit :func:`fit_nb_glmm` to every neighbourhood and assemble the results like R Milo does.

    The reported log fold change is the last column of the fixed effects model matrix, matching the coefficient that the edgeR solver tests.
    """
    library_size = counts.sum(axis=0)
    logcpm = np.log2(np.mean(counts / np.where(library_size > 0, library_size, 1), axis=1) * 1e6 + 1e-12)

    df = max(counts.shape[1] - X.shape[1], 1)
    records = []
    for nhood in range(counts.shape[0]):
        y = counts[nhood].astype(float)
        if has_separation(y, X):
            records.append(
                {
                    "logFC": np.nan,
                    "SE": np.nan,
                    "tvalue": np.nan,
                    "PValue": np.nan,
                    **{f"{name}_variance": np.nan for name, _ in random_effects},
                    "Dispersion": np.nan,
                    "Logliklihood": np.nan,
                    "Converged": False,
                }
            )
            continue

        fit = fit_nb_glmm(y, X, random_effects, offset, reml=reml, max_iter=max_iter, tol=tol)
        t_value = fit.beta[-1] / fit.se[-1] if fit.se[-1] > 0 else np.nan
        records.append(
            {
                "logFC": fit.beta[-1],
                "SE": fit.se[-1],
                "tvalue": t_value,
                "PValue": 2 * stats.t.sf(abs(t_value), df),
                **{f"{name}_variance": value for (name, _), value in zip(random_effects, fit.sigma, strict=True)},
                "Dispersion": fit.dispersion,
                "Logliklihood": fit.loglik,
                "Converged": fit.converged,
            }
        )

    res = pd.DataFrame.from_records(records)
    res.insert(1, "logCPM", logcpm)
    return res
