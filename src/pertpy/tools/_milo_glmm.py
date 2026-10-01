from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from numba import get_num_threads, njit
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

_RANDOM_EFFECT = re.compile(r"\(\s*1\s*\|\s*([^)]+?)\s*\)")
_LOG_2PI = float(np.log(2 * np.pi))
_LOG_DISPERSION_BOUNDS = (float(np.log(1e-6)), float(np.log(1e3)))


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
    covariance: np.ndarray
    fitted: np.ndarray
    sigma: np.ndarray
    dispersion: float
    loglik: float
    converged: bool


@njit(cache=True)
def _cholesky(a: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a symmetric matrix, all NaN if it is not positive definite.

    The NaNs propagate through the fit of a neighbourhood, which is then reported as failed.
    """
    n = a.shape[0]
    lower = np.zeros_like(a)
    for j in range(n):
        pivot = a[j, j]
        for k in range(j):
            pivot -= lower[j, k] * lower[j, k]
        if not pivot > 0:
            lower[:] = np.nan
            return lower
        lower[j, j] = np.sqrt(pivot)
        for i in range(j + 1, n):
            value = a[i, j]
            for k in range(j):
                value -= lower[i, k] * lower[j, k]
            lower[i, j] = value / lower[j, j]
    return lower


@njit(cache=True)
def _cho_solve(lower: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``lower @ lower.T @ x = b`` for every column of ``b``."""
    n = lower.shape[0]
    x = b.copy()
    for column in range(x.shape[1]):
        for i in range(n):
            value = x[i, column]
            for k in range(i):
                value -= lower[i, k] * x[k, column]
            x[i, column] = value / lower[i, i]
        for i in range(n - 1, -1, -1):
            value = x[i, column]
            for k in range(i + 1, n):
                value -= lower[k, i] * x[k, column]
            x[i, column] = value / lower[i, i]
    return x


@njit(cache=True)
def _pinv(a: np.ndarray) -> np.ndarray:
    """:func:`numpy.linalg.pinv`, all NaN rather than an error if ``a`` is not finite."""
    if not np.isfinite(a).all():
        return np.full(a.shape, np.nan)
    return np.ascontiguousarray(np.linalg.pinv(a))


@njit(cache=True)
def _slogdet(a: np.ndarray) -> tuple[float, float]:
    """:func:`numpy.linalg.slogdet`, NaN rather than an error if ``a`` is not finite."""
    if not np.isfinite(a).all():
        return np.nan, np.nan
    return np.linalg.slogdet(a)


@njit(cache=True)
def _lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """:func:`numpy.linalg.lstsq` with ``rcond=None``, all NaN rather than an error if ``a`` or ``b`` is not finite."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.full(a.shape[1], np.nan)
    return np.linalg.lstsq(a, b, rcond=np.finfo(np.float64).eps * max(a.shape))[0]


@njit(cache=True)
def _poisson_means(y: np.ndarray, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Fit a Poisson GLM with a log link by iteratively reweighted least squares."""
    mu = np.maximum(y, 0.1)
    for _ in range(25):
        working = np.log(mu) - offset + (y - mu) / mu
        root = np.sqrt(mu)
        coef = _lstsq(X * root[:, None], working * root)
        new_mu = np.exp(np.clip(offset + X @ coef, -30.0, 30.0))
        if np.all(np.abs(new_mu - mu) <= 1e-8 + 1e-6 * np.abs(mu)):
            return new_mu
        mu = new_mu
    return mu


@njit(cache=True)
def _pearson_excess(dispersion: float, squared_error: np.ndarray, mu: np.ndarray, df: float) -> float:
    """Negative binomial Pearson statistic at ``dispersion`` in excess of ``df``."""
    return np.sum(squared_error / (mu + dispersion * mu**2)) - df


@njit(cache=True)
def _dispersion_from_means(y: np.ndarray, mu: np.ndarray, df: float) -> float:
    """Solve for the dispersion at which the negative binomial Pearson statistic equals its degrees of freedom.

    The root is bracketed in ``[0, upper]`` and found with the iterations of :func:`scipy.optimize.brentq`.
    """
    squared_error = (y - mu) ** 2
    xpre, fpre = 0.0, _pearson_excess(0.0, squared_error, mu, df)
    if fpre <= 0:
        return 0.0
    xcur, fcur = 1.0, _pearson_excess(1.0, squared_error, mu, df)
    while fcur > 0 and xcur < 1e6:
        xcur *= 10
        fcur = _pearson_excess(xcur, squared_error, mu, df)
    if fcur > 0:
        return 1e6

    xtol, rtol = 2e-12, 4 * np.finfo(np.float64).eps
    xblk = fblk = spre = scur = 0.0
    if fcur == 0:
        return xcur
    for _ in range(100):
        if fpre != 0 and fcur != 0 and (fpre < 0) != (fcur < 0):
            xblk, fblk = xpre, fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre, scur = scur, stry
            else:
                spre = scur = sbis
        else:
            spre = scur = sbis
        xpre, fpre = xcur, fcur
        xcur += scur if abs(scur) > delta else (delta if sbis > 0 else -delta)
        fcur = _pearson_excess(xcur, squared_error, mu, df)
    return xcur


@njit(cache=True)
def _negative_adjusted_loglik(
    log_dispersion: float, y: np.ndarray, mu: np.ndarray, design: np.ndarray, penalty: np.ndarray
) -> float:
    """Negative Cox-Reid adjusted profile log likelihood at ``log_dispersion``."""
    dispersion = np.exp(log_dispersion)
    size = 1.0 / dispersion
    loglik = 0.0
    for i in range(len(y)):
        loglik += (
            math.lgamma(y[i] + size)
            - math.lgamma(size)
            - math.lgamma(y[i] + 1)
            + size * np.log(size / (size + mu[i]))
            + y[i] * np.log(max(mu[i], 1e-12) / (size + mu[i]))
        )
    weights = mu / (1.0 + dispersion * mu)
    sign, logdet = _slogdet(design.T @ (design * weights[:, None]) + np.diag(penalty))
    return -(loglik - 0.5 * logdet) if sign > 0 else np.inf


@njit(cache=True)
def _adjusted_profile_dispersion(y: np.ndarray, mu: np.ndarray, design: np.ndarray, penalty: np.ndarray) -> float:
    """Dispersion maximising the Cox-Reid adjusted profile likelihood at the fitted means.

    Profiling out the coefficients biases the dispersion downwards by an amount that depends on how many of them there are, which is why a moment estimator drifts with the size of the design.
    Subtracting half the log determinant of the information of the coefficients removes that drift, and the information is the penalised one of the mixed model so that the random effects count for as much as they were shrunk towards zero.
    The log dispersion is searched with the iterations of ``scipy.optimize.minimize_scalar(method="bounded")``.
    """
    xatol, maxfun = 1e-5, 500
    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = _LOG_DISPERSION_BOUNDS
    fulc = a + golden_mean * (b - a)
    nfc = xf = fulc
    rat = e = 0.0
    fx = _negative_adjusted_loglik(xf, y, mu, design, penalty)
    num = 1
    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = True
        if abs(e) > tol1:
            golden = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if abs(p) < abs(0.5 * q * r) and p > q * (a - xf) and p < q * (b - xf):
                rat = (p + 0.0) / q
                x = xf + rat
                if (x - a) < tol2 or (b - x) < tol2:
                    rat = tol1 * (np.sign(xm - xf) + ((xm - xf) == 0))
            else:
                golden = True
        if golden:
            e = a - xf if xf >= xm else b - xf
            rat = golden_mean * e
        x = xf + (np.sign(rat) + (rat == 0)) * max(abs(rat), tol1)
        fu = _negative_adjusted_loglik(x, y, mu, design, penalty)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fnfc or nfc == xf:
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif fu <= ffulc or fulc in (xf, nfc):
                fulc, ffulc = x, fu
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            break
    return np.exp(xf)


def has_separation(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Check whether the counts are completely separated by a column of the model matrix.

    A neighbourhood in which every sample of one group has zero counts has no finite maximum likelihood estimate, so it is reported as not converged instead of as an arbitrarily large fold change.
    Every row of a two-dimensional ``y`` is checked separately.
    """
    positive = np.asarray(y) > 0
    separated = np.asarray(~positive.any(axis=-1))
    for column in X.T:
        mask = column != 0
        if 0 < mask.sum() < len(mask):
            separated |= ~positive[..., mask].any(axis=-1) | ~positive[..., ~mask].any(axis=-1)
    return separated


def between_within_df(
    X: np.ndarray, random_effects: Sequence[tuple[str, np.ndarray]], contrast: np.ndarray | None = None
) -> int:
    """Degrees of freedom for the t-test on ``contrast``, following the between-within rule.

    A contrast that is constant within every level of a random intercept is only informed by as many independent units as there are levels, so testing it against the number of samples treats repeated measurements as independent and is anti-conservative.
    """
    n, p = X.shape
    tested = X[:, -1] if contrast is None else X @ contrast
    between = [
        Z.shape[1]
        for _, Z in random_effects
        if all(np.ptp(tested[mask]) == 0 for mask in (Z[:, level] != 0 for level in range(Z.shape[1])) if mask.any())
    ]
    if between:
        return max(min(between) - p, 1)
    return max(n - sum(Z.shape[1] for _, Z in random_effects) - p + 1, 1)


@njit(cache=True)
def _random_effect_variance(sigma: np.ndarray, zz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Variance of the working response, the sum of the residual variance and the scaled random effect outer products."""
    V = np.diag(1.0 / weights)
    for k in range(len(sigma)):
        V += sigma[k] * zz[k]
    return V


@njit(cache=True)
def _pseudo_likelihood(  # noqa: PLR0917
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    bounds: np.ndarray,
    zz: np.ndarray,
    offset: np.ndarray,
    dispersion: float,
    beta: np.ndarray,
    sigma: np.ndarray,
    u: np.ndarray,
    reml: bool,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Pseudo-likelihood iterations at a fixed dispersion, starting from ``beta``, ``sigma`` and ``u``."""
    n, p = X.shape
    size = np.inf if dispersion <= 0 else 1.0 / dispersion
    design = np.hstack((X, Z))
    converged = False
    for _ in range(max_iter):
        eta = offset + X @ beta + Z @ u
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        working = eta - offset + (y - mu) / mu
        lower = _cholesky(_random_effect_variance(sigma, zz, np.maximum(mu / (1.0 + mu / size), 1e-8)))

        solved = _cho_solve(lower, design)
        v_inv_x = np.ascontiguousarray(solved[:, :p])
        xtvx_inv = _pinv(X.T @ v_inv_x)
        new_beta = xtvx_inv @ (v_inv_x.T @ working)

        v_inv_resid = _cho_solve(lower, (working - X @ new_beta)[:, None]).ravel()
        new_u = Z.T @ v_inv_resid
        stacked = np.empty((n, design.shape[1] - p + 1))
        stacked[:, 0] = v_inv_resid
        stacked[:, 1:] = solved[:, p:]
        if reml:
            stacked = stacked - v_inv_x @ (xtvx_inv @ (X.T @ stacked))
        z_stacked = Z.T @ stacked

        score = np.empty(len(sigma))
        information = np.empty((len(sigma), len(sigma)))
        for k in range(len(sigma)):
            lo, hi = bounds[k], bounds[k + 1]
            new_u[lo:hi] *= sigma[k]
            projected = z_stacked[lo:hi, 0]
            score[k] = -0.5 * np.sum(stacked[:, 1 + lo : 1 + hi] * Z[:, lo:hi]) + 0.5 * np.sum(projected * projected)
            for l in range(len(sigma)):
                lo_l, hi_l = bounds[l], bounds[l + 1]
                information[k, l] = 0.5 * np.sum(
                    z_stacked[lo:hi, 1 + lo_l : 1 + hi_l] * z_stacked[lo_l:hi_l, 1 + lo : 1 + hi].T
                )
        new_sigma = np.maximum(sigma + _pinv(information) @ score, 1e-8)

        delta = max(np.max(np.abs(new_beta - beta)), np.max(np.abs(new_sigma - sigma)))
        beta, sigma, u = new_beta, new_sigma, new_u
        if delta < tol:
            converged = True
            break

    return beta, sigma, u, np.exp(np.clip(offset + X @ beta + Z @ u, -30.0, 30.0)), converged


@njit(nogil=True, cache=True)
def _fit_nb_glmm_rows(  # noqa: PLR0917
    counts: np.ndarray,
    dispersion: np.ndarray,
    fit: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    bounds: np.ndarray,
    zz: np.ndarray,
    offset: np.ndarray,
    reml: bool,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit every row of ``counts`` for which ``fit`` is set, estimating its dispersion where ``dispersion`` is NaN.

    Rows that are not fit and rows whose variance matrix cannot be factorised are NaN and not converged, since one pathological neighbourhood should not abort a run over thousands of them.
    """
    n_rows, (n, p) = counts.shape[0], X.shape
    residual_df = max(n - p, 1)
    design = np.hstack((X, Z))
    beta_rows = np.full((n_rows, p), np.nan)
    covariance_rows = np.full((n_rows, p, p), np.nan)
    fitted_rows = np.full((n_rows, n), np.nan)
    sigma_rows = np.full((n_rows, len(bounds) - 1), np.nan)
    dispersion_rows = np.full(n_rows, np.nan)
    loglik_rows = np.full(n_rows, np.nan)
    converged_rows = np.zeros(n_rows, dtype=np.bool_)
    for row in range(n_rows):
        if not fit[row]:
            continue
        y = counts[row]
        start = np.log(y + 1) - offset
        beta = _lstsq(X, start)
        start = start - X @ beta
        sigma = np.full(len(bounds) - 1, max(np.sum(start * start) / residual_df, 1e-3))
        u = np.zeros(Z.shape[1])

        phi = dispersion[row]
        if np.isnan(phi):
            penalty = np.zeros(design.shape[1])
            phi = _dispersion_from_means(y, _poisson_means(y, X, offset), residual_df)
            for _ in range(4):
                beta, sigma, u, fitted_mean, _ = _pseudo_likelihood(
                    y, X, Z, bounds, zz, offset, phi, beta, sigma, u, reml, max_iter, tol
                )
                for k in range(len(sigma)):
                    penalty[p + bounds[k] : p + bounds[k + 1]] = 1.0 / max(sigma[k], 1e-8)
                refined = _adjusted_profile_dispersion(y, fitted_mean, design, penalty)
                if abs(np.log1p(refined) - np.log1p(phi)) < 1e-3:
                    phi = refined
                    break
                phi = refined

        beta, sigma, u, mu, converged = _pseudo_likelihood(
            y, X, Z, bounds, zz, offset, phi, beta, sigma, u, reml, max_iter, tol
        )
        size = np.inf if phi <= 0 else 1.0 / phi
        lower = _cholesky(_random_effect_variance(sigma, zz, np.maximum(mu / (1.0 + mu / size), 1e-8)))
        xtvx = X.T @ _cho_solve(lower, X)
        resid = np.log(mu) - offset + (y - mu) / mu - X @ beta
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(lower))))
        loglik = -0.5 * (logdet + np.sum(resid * _cho_solve(lower, resid[:, None]).ravel()) + n * _LOG_2PI)
        if reml:
            loglik -= 0.5 * _slogdet(xtvx)[1]
        if not np.isnan(loglik):
            beta_rows[row], covariance_rows[row], fitted_rows[row] = beta, _pinv(xtvx), mu
            sigma_rows[row], dispersion_rows[row], loglik_rows[row], converged_rows[row] = sigma, phi, loglik, converged
    return beta_rows, covariance_rows, fitted_rows, sigma_rows, dispersion_rows, loglik_rows, converged_rows


def _stack_random_effects(
    random_effects: Sequence[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated random effect matrices, the column bounds of every random effect and their outer products."""
    matrices = [np.asarray(Z, dtype=float) for _, Z in random_effects]
    bounds = np.cumsum([0, *(Z.shape[1] for Z in matrices)])
    return np.ascontiguousarray(np.hstack(matrices)), bounds, np.stack([Z @ Z.T for Z in matrices])


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
    beta, covariance, fitted, sigma, dispersion, loglik, converged = (
        result[0]
        for result in _fit_nb_glmm_rows(
            np.asarray(y, dtype=float)[None],
            np.array([np.nan if dispersion is None else dispersion], dtype=float),
            np.ones(1, dtype=bool),
            np.ascontiguousarray(X, dtype=float),
            *_stack_random_effects(random_effects),
            np.asarray(offset, dtype=float),
            reml,
            max_iter,
            tol,
        )
    )
    return GLMMFit(
        beta=beta,
        se=np.sqrt(np.maximum(np.diag(covariance), 0)),
        covariance=covariance,
        fitted=fitted,
        sigma=sigma,
        dispersion=float(dispersion),
        loglik=float(loglik),
        converged=bool(converged),
    )


def log_cpm(counts: np.ndarray) -> np.ndarray:
    """Log2 of the mean counts per million of every neighbourhood across samples."""
    library_size = counts.sum(axis=0)
    return np.log2(np.mean(counts / np.where(library_size > 0, library_size, 1), axis=1) * 1e6 + 1e-12)


def fit_nb_glmm_nhoods(
    counts: np.ndarray,
    X: np.ndarray,
    random_effects: Sequence[tuple[str, np.ndarray]],
    offset: np.ndarray,
    *,
    contrast: np.ndarray | None = None,
    reml: bool = True,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """Fit :func:`fit_nb_glmm` to every neighbourhood and assemble the results like R Milo does.

    The reported log fold change is ``contrast`` applied to the fixed effects, defaulting to the last column of the model matrix, which is the coefficient the edgeR solver tests.
    Its p-value comes from a t-test whose degrees of freedom follow :func:`between_within_df`.
    """
    weights = np.zeros(X.shape[1]) if contrast is None else np.asarray(contrast, dtype=float)
    if contrast is None:
        weights[-1] = 1.0
    y = np.ascontiguousarray(counts, dtype=float)
    fit = ~has_separation(counts, X)
    model = (
        np.ascontiguousarray(X, dtype=float),
        *_stack_random_effects(random_effects),
        np.asarray(offset, dtype=float),
    )
    with ThreadPoolExecutor(get_num_threads()) as pool:
        chunks = pool.map(
            lambda rows: _fit_nb_glmm_rows(y[rows], np.full(len(rows), np.nan), fit[rows], *model, reml, max_iter, tol),
            np.array_split(np.arange(len(y)), 4 * get_num_threads()),
        )
        beta, covariance, _, sigma, dispersion, loglik, converged = (
            np.concatenate(parts) for parts in zip(*chunks, strict=True)
        )
    estimate = beta @ weights
    standard_error = np.sqrt(np.maximum(weights @ covariance @ weights, 0))
    t_value = np.divide(estimate, standard_error, out=np.full(len(estimate), np.nan), where=standard_error > 0)
    return pd.DataFrame(
        {
            "logFC": estimate,
            "logCPM": log_cpm(counts),
            "SE": standard_error,
            "tvalue": t_value,
            "PValue": 2 * stats.t.sf(np.abs(t_value), between_within_df(X, random_effects, weights)),
            **{f"{name}_variance": sigma[:, k] for k, (name, _) in enumerate(random_effects)},
            "Dispersion": dispersion,
            "Logliklihood": loglik,
            "Converged": converged,
        }
    )
