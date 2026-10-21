"""Linear mixed models with one random intercept, shared by Milo, Dialogue and the differential expression interface.

Random intercepts are written as ``(1 | variable)`` in formulas, the syntax of lme4 and R Milo.
The numba kernels fit ``y ~ X + (1 | group)`` by REML and compute Satterthwaite degrees of freedom like ``lmerTest``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import njit, prange

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


def random_effect_codes(obs: pd.DataFrame, variable: str) -> tuple[np.ndarray, int]:
    """Integer level codes of a random intercept variable and its number of levels."""
    if variable not in obs.columns:
        raise ValueError(f"Random effect variable {variable!r} is not a column of the sample metadata.")
    levels = pd.Categorical(obs[variable]).remove_unused_categories()
    if (levels.codes < 0).any():
        raise ValueError(f"Random effect variable {variable!r} contains missing values.")
    if len(levels.categories) < 2:
        raise ValueError(
            f"Random effect variable {variable!r} has a single level, which cannot be told apart from the intercept."
        )
    return levels.codes.astype(np.int64), len(levels.categories)


def random_effect_matrices(obs: pd.DataFrame, random_effects: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    """Build one indicator matrix of shape samples x levels per random intercept variable."""
    matrices = []
    for variable in random_effects:
        codes, n_levels = random_effect_codes(obs, variable)
        matrices.append((variable, np.eye(n_levels)[codes]))
    return matrices


@njit(cache=True)
def _work_buffers(p: int, n_groups: int) -> tuple[np.ndarray, ...]:
    """Scratch arrays of one random-intercept fit: ``cross``, ``chol``, ``inv``, ``beta``, ``vector``, ``resid``, ``shrink`` and ``scaled``."""
    return (
        np.empty((p + 1, p + 1)),
        np.empty((p, p)),
        np.empty((p, p)),
        np.empty(p),
        np.empty(p),
        np.empty(n_groups),
        np.empty(n_groups),
        np.empty((n_groups, p)),
    )


@njit(cache=True, error_model="numpy")
def _cholesky(a: np.ndarray, p: int, chol: np.ndarray) -> bool:
    """Lower Cholesky factor of the leading ``p`` x ``p`` block of ``a`` into ``chol``, False if that block is singular to working precision."""
    for j in range(p):
        diagonal = a[j, j]
        for k in range(j):
            diagonal -= chol[j, k] ** 2
        if not diagonal > 1e-7 * a[j, j]:
            return False
        chol[j, j] = np.sqrt(diagonal)
        for i in range(j + 1, p):
            value = a[i, j]
            for k in range(j):
                value -= chol[i, k] * chol[j, k]
            chol[i, j] = value / chol[j, j]
    return True


@njit(cache=True, error_model="numpy")
def _cholesky_solve(chol: np.ndarray, p: int, x: np.ndarray) -> None:
    """Solve ``chol @ chol.T @ solution = x`` for the solution, in place."""
    for i in range(p):
        value = x[i]
        for k in range(i):
            value -= chol[i, k] * x[k]
        x[i] = value / chol[i, i]
    for i in range(p - 1, -1, -1):
        value = x[i]
        for k in range(i + 1, p):
            value -= chol[k, i] * x[k]
        x[i] = value / chol[i, i]


@njit(cache=True, error_model="numpy")
def _cholesky_inverse(chol: np.ndarray, p: int, inv: np.ndarray) -> None:
    """Inverse of the matrix factored by ``chol`` into ``inv``."""
    for j in range(p):
        for i in range(p):
            inv[i, j] = 1.0 if i == j else 0.0
        _cholesky_solve(chol, p, inv[:, j])


@njit(cache=True, error_model="numpy")
def _random_intercept_gls(
    within: np.ndarray, sums: np.ndarray, sizes: np.ndarray, gamma: float, work: tuple[np.ndarray, ...]
) -> float:
    """GLS fit of a random-intercept model, given the ratio ``gamma`` of the random-intercept to the residual variance.

    ``within`` holds the within-group cross-products and ``sums`` the per-group sums of the design columns followed by the response.
    Fills the generalized cross-products, their Cholesky factor, the fixed effects and the per-group residual sums divided by ``1 + n_g * gamma`` of ``work``.
    Returns the generalized residual sum of squares, NaN if the design is singular.
    """
    cross, chol, _, beta, _, resid, shrink, _ = work
    p = beta.shape[0]
    for g in range(sizes.shape[0]):
        shrink[g] = 1.0 / (1.0 + sizes[g] * gamma)
    for a in range(p + 1):
        for b in range(a, p + 1):
            value = within[a, b]
            for g in range(sizes.shape[0]):
                value += shrink[g] / sizes[g] * sums[g, a] * sums[g, b]
            cross[a, b] = cross[b, a] = value
    if not _cholesky(cross, p, chol):
        return np.nan
    for a in range(p):
        beta[a] = cross[a, p]
    _cholesky_solve(chol, p, beta)
    rss = float(cross[p, p])
    for a in range(p):
        rss -= cross[a, p] * beta[a]
    for g in range(sizes.shape[0]):
        value = sums[g, p]
        for a in range(p):
            value -= sums[g, a] * beta[a]
        resid[g] = value * shrink[g]
    return rss


@njit(cache=True, error_model="numpy")
def _random_intercept_slope(
    within: np.ndarray, sums: np.ndarray, sizes: np.ndarray, log_gamma: float, work: tuple[np.ndarray, ...]
) -> float:
    """Derivative of the REML profile log-likelihood with respect to ``log(gamma)``."""
    _, chol, inv, beta, _, resid, shrink, _ = work
    gamma = np.exp(log_gamma)
    p = beta.shape[0]
    rss = _random_intercept_gls(within, sums, sizes, gamma, work)
    if np.isnan(rss):
        return np.nan
    _cholesky_inverse(chol, p, inv)
    residual_term = 0.0
    size_term = 0.0
    quad_term = 0.0
    for g in range(sizes.shape[0]):
        residual_term += resid[g] ** 2
        size_term += sizes[g] * shrink[g]
        quad = 0.0
        for a in range(p):
            for b in range(p):
                quad += sums[g, a] * inv[a, b] * sums[g, b]
        quad_term += shrink[g] ** 2 * quad
    return 0.5 * gamma * ((sizes.sum() - p) * residual_term / rss - size_term + quad_term)


@njit(cache=True, error_model="numpy")
def _random_intercept_log_gamma(
    within: np.ndarray, sums: np.ndarray, sizes: np.ndarray, work: tuple[np.ndarray, ...]
) -> float:
    """Maximize the REML profile log-likelihood over ``log(gamma)`` by bracketing a root of its slope from 0, then bisection.

    Returns -inf if the maximum is on the boundary ``gamma = 0`` and NaN if the fit is degenerate.
    """
    a, fa = 0.0, _random_intercept_slope(within, sums, sizes, 0.0, work)
    b, fb = a, fa
    step = 2.0 if fa > 0 else -2.0
    while fb != 0.0 and (fb > 0) == (fa > 0):
        a, fa = b, fb
        b += step
        if b < -40.0:
            return -np.inf
        if b > 40.0 or not np.isfinite(fa):
            return np.nan
        fb = _random_intercept_slope(within, sums, sizes, b, work)
    if not np.isfinite(fb):
        return np.nan
    mid = 0.5 * (a + b)
    while fb != 0.0 and a != mid != b:
        f_mid = _random_intercept_slope(within, sums, sizes, mid, work)
        if (f_mid > 0) == (fa > 0):
            a, fa = mid, f_mid
        else:
            b, fb = mid, f_mid
        mid = 0.5 * (a + b)
    return b if fb == 0.0 else mid


@njit(cache=True, error_model="numpy")
def _satterthwaite_df(  # noqa: PLR0917
    sums: np.ndarray, sizes: np.ndarray, gamma: float, sigma2: float, contrast: np.ndarray, work: tuple[np.ndarray, ...]
) -> float:
    """Satterthwaite denominator degrees of freedom of a contrast of the fixed effects, as lmerTest computes them.

    Uses the gradient of the contrast's variance and the REML observed information, both with respect to ``(gamma, sigma2)``.
    Reads the inverse generalized cross-products and the shrunk residual sums of the fit from ``work``.
    """
    _, _, inv, beta, vector, resid, shrink, scaled = work
    p = beta.shape[0]
    n_groups = sizes.shape[0]
    for g in range(n_groups):
        shrink[g] = 1.0 / (1.0 + sizes[g] * gamma)
        for a in range(p):
            value = 0.0
            for b in range(p):
                value += sums[g, b] * inv[b, a]
            scaled[g, a] = value
    for a in range(p):
        vector[a] = 0.0
    size_term = 0.0
    pair_term = 0.0
    quad_term = 0.0
    residual_term = 0.0
    weighted_residual_term = 0.0
    for g in range(n_groups):
        quad = 0.0
        for a in range(p):
            quad += scaled[g, a] * sums[g, a]
            vector[a] += shrink[g] * resid[g] * sums[g, a]
        size_term += (sizes[g] * shrink[g]) ** 2
        quad_term += sizes[g] * shrink[g] ** 3 * quad
        residual_term += resid[g] ** 2
        weighted_residual_term += sizes[g] * shrink[g] * resid[g] ** 2
        for h in range(n_groups):
            pair = 0.0
            for a in range(p):
                pair += scaled[g, a] * sums[h, a]
            pair_term += (shrink[g] * shrink[h] * pair) ** 2
    vector_quad = 0.0
    for a in range(p):
        for b in range(p):
            vector_quad += vector[a] * inv[a, b] * vector[b]
    info_gg = 0.5 * (
        -size_term - pair_term + 2.0 * quad_term + (2.0 * weighted_residual_term - 2.0 * vector_quad) / sigma2
    )
    info_gs = residual_term / (2.0 * sigma2**2)
    info_ss = (sizes.sum() - p) / (2.0 * sigma2**2)
    grad_s = 0.0
    for a in range(p):
        vector[a] = 0.0
        for b in range(p):
            vector[a] += inv[a, b] * contrast[b]
        grad_s += contrast[a] * vector[a]
    grad_g = 0.0
    for g in range(n_groups):
        projection = 0.0
        for a in range(p):
            projection += sums[g, a] * vector[a]
        grad_g += shrink[g] ** 2 * projection**2
    grad_g *= sigma2
    det = info_gg * info_ss - info_gs**2
    denominator = (grad_g**2 * info_ss - 2.0 * grad_g * grad_s * info_gs + grad_s**2 * info_gg) / det
    return 2.0 * (sigma2 * grad_s) ** 2 / denominator


@njit(cache=True, error_model="numpy")
def _predictor_fit(  # noqa: PLR0917
    x: np.ndarray,
    centered: np.ndarray,
    within: np.ndarray,
    sums: np.ndarray,
    sizes: np.ndarray,
    groups: np.ndarray,
    cols: np.ndarray,
    unit: np.ndarray,
) -> tuple[float, float, float]:
    """Fit the model of :func:`_random_intercept_fits` for one predictor ``x`` and return its coefficient, standard error and Satterthwaite degrees of freedom."""
    n = x.shape[0]
    p = within.shape[0] - 1
    row_within = within.copy()
    row_sums = sums.copy()
    for i in range(n):
        row_sums[groups[i], 1] += x[i]
    for i in range(n):
        row_within[1, 1] += (x[i] - row_sums[groups[i], 1] / sizes[groups[i]]) ** 2
    for a in range(cols.shape[0]):
        value = 0.0
        for i in range(n):
            value += x[i] * centered[a, i]
        row_within[1, cols[a]] = row_within[cols[a], 1] = value
    work = _work_buffers(p, sizes.shape[0])
    log_gamma = _random_intercept_log_gamma(row_within, row_sums, sizes, work)
    if np.isnan(log_gamma):
        return np.nan, np.nan, np.nan
    gamma = np.exp(log_gamma)
    rss = _random_intercept_gls(row_within, row_sums, sizes, gamma, work)
    _, chol, inv, beta, _, _, _, _ = work
    _cholesky_inverse(chol, p, inv)
    sigma2 = rss / (n - p)
    df = n - p if gamma == 0.0 else _satterthwaite_df(row_sums, sizes, gamma, sigma2, unit, work)
    return beta[1], np.sqrt(sigma2 * inv[1, 1]), df


@njit(parallel=True, cache=True, error_model="numpy")
def _random_intercept_fits(
    x: np.ndarray, fixed: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """REML fits of ``fixed[:, -1] ~ fixed[:, 0] + x[i] + fixed[:, 1:-1] + (1 | groups)`` for every row ``x[i]``.

    Returns the coefficients of ``x[i]``, their standard errors and their Satterthwaite degrees of freedom, like lmerTest's ``summary``.
    """
    n, n_fixed = fixed.shape
    m = n_fixed + 1
    sizes = np.zeros(n_groups)
    fixed_sums = np.zeros((n_groups, n_fixed))
    for i in range(n):
        sizes[groups[i]] += 1.0
        for a in range(n_fixed):
            fixed_sums[groups[i], a] += fixed[i, a]
    centered = np.empty((n_fixed, n))
    for i in range(n):
        for a in range(n_fixed):
            centered[a, i] = fixed[i, a] - fixed_sums[groups[i], a] / sizes[groups[i]]
    cols = np.concatenate((np.zeros(1, dtype=np.int64), np.arange(2, m)))
    within = np.zeros((m, m))
    sums = np.zeros((n_groups, m))
    for a in range(n_fixed):
        for g in range(n_groups):
            sums[g, cols[a]] = fixed_sums[g, a]
        for b in range(n_fixed):
            value = 0.0
            for i in range(n):
                value += centered[a, i] * centered[b, i]
            within[cols[a], cols[b]] = value
    unit = np.zeros(m - 1)
    unit[1] = 1.0
    estimates = np.full(x.shape[0], np.nan)
    std_errors = np.full(x.shape[0], np.nan)
    dfs = np.full(x.shape[0], np.nan)
    for row in prange(x.shape[0]):
        estimates[row], std_errors[row], dfs[row] = _predictor_fit(
            x[row], centered, within, sums, sizes, groups, cols, unit
        )
    return estimates, std_errors, dfs


@njit(cache=True, error_model="numpy")
def _response_fit(  # noqa: PLR0917
    y: np.ndarray,
    centered: np.ndarray,
    within_design: np.ndarray,
    design_sums: np.ndarray,
    sizes: np.ndarray,
    groups: np.ndarray,
    coefficients: np.ndarray,
    inverse: np.ndarray,
    residual_sums: np.ndarray,
) -> tuple[float, float]:
    """Fit the model of :func:`_random_intercept_response_fits` for one response ``y``.

    Writes its fixed effects, inverse generalized cross-products and shrunk per-group residual sums into the given arrays.
    Returns the variance ratio and the residual variance, both NaN if the fit is degenerate.
    """
    n = y.shape[0]
    p = design_sums.shape[1]
    n_groups = sizes.shape[0]
    within = np.zeros((p + 1, p + 1))
    within[:p, :p] = within_design
    sums = np.zeros((n_groups, p + 1))
    sums[:, :p] = design_sums
    total = 0.0
    for i in range(n):
        sums[groups[i], p] += y[i]
        total += y[i] * y[i]
        for a in range(p):
            within[a, p] += y[i] * centered[i, a]
    for a in range(p):
        within[p, a] = within[a, p]
    for g in range(n_groups):
        total -= sums[g, p] ** 2 / sizes[g]
    within[p, p] = total
    work = _work_buffers(p, n_groups)
    log_gamma = _random_intercept_log_gamma(within, sums, sizes, work)
    if np.isnan(log_gamma):
        return np.nan, np.nan
    gamma = np.exp(log_gamma)
    rss = _random_intercept_gls(within, sums, sizes, gamma, work)
    _, chol, inv, beta, _, resid, _, _ = work
    _cholesky_inverse(chol, p, inv)
    coefficients[:] = beta
    inverse[:] = inv
    residual_sums[:] = resid
    return gamma, rss / (n - p)


@njit(parallel=True, cache=True, error_model="numpy")
def _random_intercept_response_fits(
    responses: np.ndarray, design: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """REML fits of ``responses[i] ~ design + (1 | groups)`` for every row of ``responses``.

    Returns the fixed effects, the inverse generalized cross-products of the design, the variance ratio, the residual variance and the shrunk per-group residual sums of every fit, followed by the per-group design sums and group sizes shared by all fits.
    Rows whose fit is degenerate hold NaN.
    """
    n_rows, n = responses.shape
    p = design.shape[1]
    sizes = np.zeros(n_groups)
    design_sums = np.zeros((n_groups, p))
    for i in range(n):
        sizes[groups[i]] += 1.0
        for a in range(p):
            design_sums[groups[i], a] += design[i, a]
    centered = np.empty((n, p))
    for i in range(n):
        for a in range(p):
            centered[i, a] = design[i, a] - design_sums[groups[i], a] / sizes[groups[i]]
    within_design = np.zeros((p, p))
    for i in range(n):
        for a in range(p):
            for b in range(p):
                within_design[a, b] += centered[i, a] * centered[i, b]
    coefficients = np.full((n_rows, p), np.nan)
    inverses = np.full((n_rows, p, p), np.nan)
    gammas = np.full(n_rows, np.nan)
    variances = np.full(n_rows, np.nan)
    residual_sums = np.full((n_rows, n_groups), np.nan)
    for row in prange(n_rows):
        gammas[row], variances[row] = _response_fit(
            responses[row],
            centered,
            within_design,
            design_sums,
            sizes,
            groups,
            coefficients[row],
            inverses[row],
            residual_sums[row],
        )
    return coefficients, inverses, gammas, variances, residual_sums, design_sums, sizes


@njit(cache=True, error_model="numpy")
def _contrast_fit(  # noqa: PLR0917
    contrast: np.ndarray,
    coefficients: np.ndarray,
    inverse: np.ndarray,
    gamma: float,
    variance: float,
    residual_sums: np.ndarray,
    design_sums: np.ndarray,
    sizes: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate, standard error and Satterthwaite degrees of freedom of ``contrast`` for one fit of :func:`_response_fit`."""
    p = coefficients.shape[0]
    work = _work_buffers(p, sizes.shape[0])
    _, _, inv, _, _, resid, _, _ = work
    inv[:] = inverse
    resid[:] = residual_sums
    effect = 0.0
    quad = 0.0
    for a in range(p):
        effect += contrast[a] * coefficients[a]
        for b in range(p):
            quad += contrast[a] * inv[a, b] * contrast[b]
    df = sizes.sum() - p if gamma == 0.0 else _satterthwaite_df(design_sums, sizes, gamma, variance, contrast, work)
    return effect, np.sqrt(variance * quad), df


@njit(parallel=True, cache=True, error_model="numpy")
def _random_intercept_contrast(  # noqa: PLR0917
    contrast: np.ndarray,
    coefficients: np.ndarray,
    inverses: np.ndarray,
    gammas: np.ndarray,
    variances: np.ndarray,
    residual_sums: np.ndarray,
    design_sums: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate, standard error and Satterthwaite degrees of freedom of one contrast for every fit of :func:`_random_intercept_response_fits`."""
    n_rows = coefficients.shape[0]
    effects = np.full(n_rows, np.nan)
    standard_errors = np.full(n_rows, np.nan)
    dfs = np.full(n_rows, np.nan)
    for row in prange(n_rows):
        if not np.isnan(gammas[row]):
            effects[row], standard_errors[row], dfs[row] = _contrast_fit(
                contrast,
                coefficients[row],
                inverses[row],
                gammas[row],
                variances[row],
                residual_sums[row],
                design_sums,
                sizes,
            )
    return effects, standard_errors, dfs
