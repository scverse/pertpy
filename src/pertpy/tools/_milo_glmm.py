from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
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
        if dummies.shape[1] < 2:
            raise ValueError(
                f"Random effect variable {variable!r} has a single level, which cannot be told apart from the intercept."
            )
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


def _dispersion_from_means(y: np.ndarray, mu: np.ndarray, df: float) -> float:
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


def between_within_df(X: np.ndarray, random_effects: Sequence[tuple[str, np.ndarray]]) -> int:
    """Degrees of freedom for the t-test on the last fixed effect, following the between-within rule.

    A coefficient that is constant within every level of a random intercept is only informed by as many independent units as there are levels, so testing it against the number of samples treats repeated measurements as independent and is anti-conservative.
    """
    n, p = X.shape
    tested = X[:, -1]
    between = [
        Z.shape[1]
        for _, Z in random_effects
        if all(np.ptp(tested[mask]) == 0 for mask in (Z[:, level] != 0 for level in range(Z.shape[1])) if mask.any())
    ]
    if between:
        return max(min(between) - p, 1)
    return max(n - sum(Z.shape[1] for _, Z in random_effects) - p + 1, 1)


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
    matrices = [Z for _, Z in random_effects]
    return _fit_nb_glmm(
        y,
        X,
        matrices,
        [Z @ Z.T for Z in matrices],
        offset,
        dispersion=dispersion,
        reml=reml,
        max_iter=max_iter,
        tol=tol,
    )


def _fit_nb_glmm(
    y: np.ndarray,
    X: np.ndarray,
    matrices: Sequence[np.ndarray],
    zz: Sequence[np.ndarray],
    offset: np.ndarray,
    *,
    dispersion: float | None,
    reml: bool,
    max_iter: int,
    tol: float,
) -> GLMMFit:
    """Fit one neighbourhood, reporting a neighbourhood whose variance matrix cannot be factorised as not converged.

    One pathological neighbourhood should not abort a run over thousands of them.
    """
    try:
        return _fit_nb_glmm_core(
            y, X, matrices, zz, offset, dispersion=dispersion, reml=reml, max_iter=max_iter, tol=tol
        )
    except np.linalg.LinAlgError:
        return GLMMFit(
            beta=np.full(X.shape[1], np.nan),
            se=np.full(X.shape[1], np.nan),
            sigma=np.full(len(matrices), np.nan),
            dispersion=np.nan,
            loglik=np.nan,
            converged=False,
        )


def _fit_nb_glmm_core(
    y: np.ndarray,
    X: np.ndarray,
    matrices: Sequence[np.ndarray],
    zz: Sequence[np.ndarray],
    offset: np.ndarray,
    *,
    dispersion: float | None,
    reml: bool,
    max_iter: int,
    tol: float,
) -> GLMMFit:
    """Fit one neighbourhood, reusing the outer products of the random effect matrices across neighbourhoods."""
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    residual_df = max(n - p, 1)

    def pseudo_likelihood(
        dispersion: float, start_from: tuple | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, bool, float]:
        size = np.inf if dispersion <= 0 else 1.0 / dispersion
        if start_from is None:
            beta, *_ = np.linalg.lstsq(X, np.log(y + 1) - offset, rcond=None)
            start = np.log(y + 1) - offset - X @ beta
            sigma = np.full(len(matrices), max(float(start @ start) / residual_df, 1e-3))
            u = [np.zeros(Z.shape[1]) for Z in matrices]
        else:
            beta, sigma, u = start_from
        converged = False

        for _ in range(max_iter):
            eta = offset + X @ beta + sum((Z @ u_k for Z, u_k in zip(matrices, u, strict=True)), np.zeros(n))
            mu = np.exp(np.clip(eta, -30, 30))
            weights = np.maximum(mu if np.isinf(size) else mu / (1.0 + mu / size), 1e-8)
            working = eta - offset + (y - mu) / mu

            V = sum((s * m for s, m in zip(sigma, zz, strict=True)), np.diag(1.0 / weights))
            chol = cho_factor(V, lower=True)

            solved = cho_solve(chol, np.column_stack([X, *matrices]))
            v_inv_x = solved[:, :p]

            xtvx_inv = np.linalg.pinv(X.T @ v_inv_x)
            new_beta = xtvx_inv @ (v_inv_x.T @ working)

            resid = working - X @ new_beta
            v_inv_resid = cho_solve(chol, resid)
            new_u = [s * (Z.T @ v_inv_resid) for s, Z in zip(sigma, matrices, strict=True)]

            stacked = np.column_stack([v_inv_resid, solved[:, p:]])
            if reml:
                stacked = stacked - v_inv_x @ (xtvx_inv @ (X.T @ stacked))
            projected = stacked[:, 0]
            p_z = np.split(stacked[:, 1:], np.cumsum([Z.shape[1] for Z in matrices])[:-1], axis=1)

            score = np.array(
                [
                    -0.5 * float(np.sum(b * Z)) + 0.5 * float((Z.T @ projected) @ (Z.T @ projected))
                    for b, Z in zip(p_z, matrices, strict=True)
                ]
            )
            information = np.array(
                [
                    [
                        0.5 * float(np.sum((Z_k.T @ b_l) * (Z_l.T @ b_k).T))
                        for b_l, Z_l in zip(p_z, matrices, strict=True)
                    ]
                    for b_k, Z_k in zip(p_z, matrices, strict=True)
                ]
            )
            new_sigma = np.maximum(sigma + np.linalg.pinv(information) @ score, 1e-8)

            delta = max(np.max(np.abs(new_beta - beta)), np.max(np.abs(new_sigma - sigma)))
            beta, sigma, u = new_beta, new_sigma, new_u
            if delta < tol:
                converged = True
                break

        eta = offset + X @ beta + sum((Z @ u_k for Z, u_k in zip(matrices, u, strict=True)), np.zeros(n))
        fitted = float(sum(s * np.trace(Z.T @ b) for s, Z, b in zip(sigma, matrices, p_z, strict=True)))
        return beta, sigma, u, np.exp(np.clip(eta, -30, 30)), converged, fitted

    warm_start = None
    if dispersion is None:
        # The fixed effects only estimate absorbs part of the random effect variance, so refine it once the
        # neighbourhood has been fitted with its random effects. The refit also spends degrees of freedom on
        # the random effects, so the residuals are smaller than a fixed effects only fit would leave.
        dispersion = _dispersion_from_means(y, _poisson_means(y, X, offset), residual_df)
        beta, sigma, u, fitted_mean, _, fitted_df = pseudo_likelihood(dispersion)
        dispersion = _dispersion_from_means(y, fitted_mean, max(n - p - fitted_df, 1.0))
        warm_start = (beta, sigma, u)

    beta, sigma, u, mu, converged, _ = pseudo_likelihood(dispersion, warm_start)

    size = np.inf if dispersion <= 0 else 1.0 / dispersion
    weights = np.maximum(mu if np.isinf(size) else mu / (1.0 + mu / size), 1e-8)
    working = np.log(mu) - offset + (y - mu) / mu
    V = sum((s * m for s, m in zip(sigma, zz, strict=True)), np.diag(1.0 / weights))
    chol = cho_factor(V, lower=True)
    xtvx = X.T @ cho_solve(chol, X)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(xtvx)), 0))

    resid = working - X @ beta
    logdet = 2.0 * float(np.sum(np.log(np.abs(np.diag(chol[0])))))
    loglik = -0.5 * (logdet + float(resid @ cho_solve(chol, resid)) + n * np.log(2 * np.pi))
    if reml:
        loglik -= 0.5 * np.linalg.slogdet(xtvx)[1]

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
    Its p-value comes from a t-test whose degrees of freedom follow :func:`between_within_df`.
    """
    library_size = counts.sum(axis=0)
    logcpm = np.log2(np.mean(counts / np.where(library_size > 0, library_size, 1), axis=1) * 1e6 + 1e-12)

    df = between_within_df(X, random_effects)
    matrices = [Z for _, Z in random_effects]
    zz = [Z @ Z.T for Z in matrices]
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

        fit = _fit_nb_glmm(y, X, matrices, zz, offset, dispersion=None, reml=reml, max_iter=max_iter, tol=tol)
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
