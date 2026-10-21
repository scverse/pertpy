"""Poisson-Gaussian mixture model for gRNA assignment.

Implements the model of ``crispat.ga_poisson_gauss`` (Velten group, https://github.com/velten-group/crispat).
For every guide, the log2 of its non-zero UMI counts is modelled as a mixture of a continuous Poisson background and a Gaussian on-target component with priors ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)`` and ``lam ~ LogNormal(0, 1)``.
The MAP estimate is found by multi-start EM on the unique count values of every guide, run for all guides in parallel with JAX, where crispat runs one stochastic variational inference loop per guide.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from scipy.special import gammaln as _np_gammaln

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))
_SCALE_FLOOR = 1e-3
_WEIGHT_FLOOR = 1e-6
_INIT_QUANTILES = (0.5, 0.75, 0.9, 0.97, 0.99)
_PRIOR_INIT_SAMPLES = 15
_NEWTON_STEPS = 12
_CHUNK_ELEMENTS = 2**22


@dataclass
class PoissonGaussFit:
    """MAP estimates for the Poisson-Gaussian mixture, one entry per guide and ``nan`` where no admissible solution exists."""

    poisson_rate: np.ndarray
    gaussian_mean: np.ndarray
    gaussian_std: np.ndarray
    mix_probs: np.ndarray
    final_loss: np.ndarray


def _neg_log_joint(
    rate: jnp.ndarray,
    mu: jnp.ndarray,
    scale: jnp.ndarray,
    w_norm: jnp.ndarray,
    *,
    values: jnp.ndarray,
    weights: jnp.ndarray,
    log_gamma: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Negative log joint of one guide on weighted unique values, with the Gaussian responsibilities.

    Additive prior constants are dropped.
    """
    log_pois = values * jnp.log(rate) - rate - log_gamma
    log_norm = -_LOG_SQRT_2PI - jnp.log(scale) - 0.5 * jnp.square((values - mu) / scale)
    log_norm_weighted = jnp.log(w_norm) + log_norm
    log_mix = jnp.logaddexp(log_norm_weighted, jnp.log1p(-w_norm) + log_pois)
    log_prior = (
        -0.1 * jnp.log1p(-w_norm)
        - 0.9 * jnp.log(w_norm)
        - 0.5 * jnp.square((mu - 3.0) / 2.0)
        - jnp.log(scale)
        - 0.5 * jnp.square(jnp.log(scale) - 2.0)
        - jnp.log(rate)
        - 0.5 * jnp.square(jnp.log(rate))
    )
    return -(jnp.sum(weights * log_mix) + log_prior), jnp.exp(log_norm_weighted - log_mix)


def _solve_log_scale(sum_squares: jnp.ndarray, n_norm: jnp.ndarray) -> jnp.ndarray:
    """MAP log scale of the Gaussian given its responsibility mass: the root of a convex decreasing function."""
    log_floor = float(np.log(_SCALE_FLOOR))
    v0 = 0.5 * jnp.log(jnp.maximum(sum_squares, 1e-12) / jnp.maximum(n_norm, 1e-12))

    def step(_: int, v: jnp.ndarray) -> jnp.ndarray:
        e = sum_squares * jnp.exp(-2.0 * v)
        f = e - (v - 2.0) - n_norm - 1.0
        return jnp.clip(v - f / (-2.0 * e - 1.0), log_floor, 10.0)

    return jax.lax.fori_loop(0, _NEWTON_STEPS, step, jnp.clip(v0, log_floor, 10.0))


def _solve_log_rate(sum_values: jnp.ndarray, n_pois: jnp.ndarray) -> jnp.ndarray:
    """MAP log rate of the Poisson given its responsibility mass: the root of a concave decreasing function."""
    l0 = jnp.log(jnp.maximum(sum_values, 1e-12) / jnp.maximum(n_pois, 1e-12))

    def step(_: int, log_rate: jnp.ndarray) -> jnp.ndarray:
        e = n_pois * jnp.exp(log_rate)
        g = sum_values - 1.0 - log_rate - e
        return jnp.clip(log_rate - g / (-1.0 - e), -20.0, 10.0)

    return jax.lax.fori_loop(0, _NEWTON_STEPS, step, jnp.clip(l0, -20.0, 10.0))


def _em_step(
    params: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    values: jnp.ndarray,
    weights: jnp.ndarray,
    log_gamma: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """One MAP-EM step; returns the updated parameters and the loss of the incoming ones."""
    rate, mu, scale, w_norm = params
    loss, resp = _neg_log_joint(rate, mu, scale, w_norm, values=values, weights=weights, log_gamma=log_gamma)
    w_resp = weights * resp
    w_rest = weights - w_resp
    n_norm = w_resp.sum()
    n_pois = w_rest.sum()
    w_new = jnp.clip((n_norm - 0.9) / (n_norm + n_pois - 1.0), _WEIGHT_FLOOR, 1.0 - _WEIGHT_FLOOR)
    precision = 1.0 / jnp.square(scale)
    mu_new = ((w_resp * values).sum() * precision + 0.75) / (n_norm * precision + 0.25)
    sum_squares = (w_resp * jnp.square(values - mu_new)).sum()
    scale_new = jnp.maximum(jnp.exp(_solve_log_scale(sum_squares, n_norm)), _SCALE_FLOOR)
    rate_new = jnp.exp(_solve_log_rate((w_rest * values).sum(), n_pois))
    return (rate_new, mu_new, scale_new, w_new), loss


def _fit_one(
    init: jnp.ndarray,
    values: jnp.ndarray,
    weights: jnp.ndarray,
    log_gamma: jnp.ndarray,
    *,
    n_iter: int,
    tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run EM from one initialisation until the loss stops changing."""
    params = (init[0], init[1], init[2], init[3])
    big = jnp.asarray(1e30, dtype=values.dtype)

    def cond(carry):
        _, previous, current, i = carry
        return (i < n_iter) & (jnp.abs(previous - current) > tol * jnp.abs(current))

    def body(carry):
        params, _, current, i = carry
        new_params, loss = _em_step(params, values, weights, log_gamma)
        return new_params, current, loss, i + 1

    params, _, _, _ = jax.lax.while_loop(cond, body, (params, jnp.asarray(jnp.inf, dtype=values.dtype), big, 0))
    loss, _ = _neg_log_joint(*params, values=values, weights=weights, log_gamma=log_gamma)
    return (*params, loss)


def _prior_median_inits(key: jax.Array, n_guides: int, n_starts: int, dtype: jnp.dtype) -> jnp.ndarray:
    """Initialisations as in crispat: prior medians for the continuous parameters and equal weights."""
    k_rate, k_mu, k_scale = jax.random.split(key, 3)
    shape = (n_starts, _PRIOR_INIT_SAMPLES, n_guides)
    rate = jnp.exp(jnp.median(jax.random.normal(k_rate, shape, dtype=dtype), axis=1))
    mu = jnp.median(3.0 + 2.0 * jax.random.normal(k_mu, shape, dtype=dtype), axis=1)
    scale = jnp.exp(jnp.median(2.0 + jax.random.normal(k_scale, shape, dtype=dtype), axis=1))
    w_norm = jnp.full((n_starts, n_guides), 0.5, dtype=dtype)
    return jnp.stack([rate, mu, scale, w_norm], axis=-1)


def _data_inits(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Initialisations that split every guide at a quantile, Poisson below and Gaussian above."""
    total = jnp.maximum(weights.sum(axis=1), 1.0)
    cumulative = jnp.cumsum(weights, axis=1) / total[:, None]
    inits = []
    for q in _INIT_QUANTILES:
        upper = (cumulative > q) & (weights > 0)
        w_up = jnp.where(upper, weights, 0.0)
        w_low = weights - w_up
        n_up = w_up.sum(axis=1)
        n_low = w_low.sum(axis=1)
        rate = jnp.maximum((w_low * values).sum(axis=1) / jnp.maximum(n_low, 1.0), 0.05)
        mu = jnp.where(n_up > 0, (w_up * values).sum(axis=1) / jnp.maximum(n_up, 1.0), values.max(axis=1))
        variance = (w_up * jnp.square(values - mu[:, None])).sum(axis=1) / jnp.maximum(n_up, 1.0)
        scale = jnp.maximum(jnp.sqrt(variance), 0.5)
        w_norm = jnp.clip(n_up / total, 0.01, 0.9)
        inits.append(jnp.stack([rate, mu, scale, w_norm], axis=-1))
    return jnp.stack(inits)


@partial(jax.jit, static_argnames=("n_init_seeds", "n_iter", "tol"))
def _fit_chunk(
    values: jnp.ndarray,
    weights: jnp.ndarray,
    key: jax.Array,
    *,
    n_init_seeds: int,
    n_iter: int,
    tol: float,
) -> tuple[jnp.ndarray, ...]:
    """Fit all starts of a chunk of guides and keep the admissible solution with the lowest loss per guide.

    Label-switched optima (Gaussian below the Poisson) and solutions pinned at the weight or scale floor are inadmissible.
    """
    log_gamma = gammaln(values + 1.0)
    inits = jnp.concatenate(
        [_prior_median_inits(key, values.shape[0], n_init_seeds, values.dtype), _data_inits(values, weights)]
    )
    fit_one = partial(_fit_one, n_iter=n_iter, tol=tol)
    run = jax.vmap(jax.vmap(fit_one, in_axes=(0, 0, 0, 0)), in_axes=(0, None, None, None))
    rate, mu, scale, w_norm, loss = run(inits, values, weights, log_gamma)
    admissible = jnp.isfinite(loss) & (mu > rate) & (w_norm > 2 * _WEIGHT_FLOOR) & (scale > 2 * _SCALE_FLOOR)
    best = jnp.argmin(jnp.where(admissible, loss, jnp.inf), axis=0)[None]
    rate, mu, scale, w_norm, loss, ok = (
        jnp.take_along_axis(a, best, axis=0)[0] for a in (rate, mu, scale, w_norm, loss, admissible)
    )
    return rate, mu, scale, w_norm, loss, ok


def fit_poisson_gauss_mixture(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_iter: int = 1000,
    n_init_seeds: int = 10,
    seed: int = 2024,
    tol: float = 1e-7,
) -> PoissonGaussFit:
    """Fit Poisson-Gaussian mixtures for many guides in parallel by multi-start MAP-EM.

    Guides are grouped by their number of distinct values and fit in chunks of bounded size, so peak memory does not grow with the number of guides.

    Args:
        values: Sorted log2-transformed distinct non-zero counts per guide of shape ``[n_guides, n_values]``, zero padded.
        weights: Number of cells with each value, zero for padding.
        n_iter: Maximum EM iterations per start.
        n_init_seeds: Number of prior-median initialisations per guide, used in addition to quantile-split initialisations.
        seed: Random seed for the prior-median initialisations.
        tol: Relative change of the loss below which EM stops.

    Returns:
        MAP estimates for all guides.
    """
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    n_guides = values.shape[0]
    n_starts = n_init_seeds + len(_INIT_QUANTILES)
    n_values = (weights > 0).sum(axis=1)
    bucket = np.ceil(np.log2(np.maximum(n_values, 2))).astype(int)
    key = jax.random.PRNGKey(seed)

    out = {name: np.full(n_guides, np.nan) for name in ("rate", "mu", "scale", "w_norm", "loss")}
    for chunk_id, b in enumerate(np.unique(bucket)):
        guides = np.flatnonzero(bucket == b)
        width = min(2**b, values.shape[1])
        rows = 2 ** int(np.ceil(np.log2(min(guides.size, max(_CHUNK_ELEMENTS // (n_starts * width), 1)))))
        for part, chunk in enumerate(np.array_split(guides, int(np.ceil(guides.size / rows)))):
            chunk_values = np.zeros((rows, width), dtype=np.float32)
            chunk_weights = np.zeros((rows, width), dtype=np.float32)
            chunk_values[: chunk.size] = values[chunk, :width]
            chunk_weights[: chunk.size] = weights[chunk, :width]
            chunk_key = jax.random.fold_in(key, chunk_id * 2**16 + part)
            rate, mu, scale, w_norm, loss, ok = (
                np.asarray(a)[: chunk.size]
                for a in _fit_chunk(
                    chunk_values, chunk_weights, chunk_key, n_init_seeds=n_init_seeds, n_iter=n_iter, tol=tol
                )
            )
            fitted = chunk[ok]
            for name, estimate in zip(out, (rate, mu, scale, w_norm, loss), strict=True):
                out[name][fitted] = estimate[ok]

    return PoissonGaussFit(
        poisson_rate=out["rate"],
        gaussian_mean=out["mu"],
        gaussian_std=out["scale"],
        mix_probs=np.stack([1.0 - out["w_norm"], out["w_norm"]], axis=-1),
        final_loss=out["loss"],
    )


def compute_count_thresholds(fit: PoissonGaussFit, max_counts: np.ndarray) -> np.ndarray:
    """Per-guide assignment threshold matching crispat.

    For each guide, scans integer UMI counts ``t = 1, 2, ..., max_count`` and returns the smallest ``t`` for which ``P(Normal | log2(t)) > 0.5``.
    Returns ``np.nan`` if no such ``t`` exists within ``max_count`` or the guide has no fit.
    """
    max_counts = np.asarray(max_counts, dtype=np.int64)
    n_guides = len(fit.poisson_rate)
    thresholds = np.full(n_guides, np.nan, dtype=np.float64)
    for g, max_c in enumerate(max_counts):
        if max_c < 1 or np.isnan(fit.gaussian_mean[g]):
            continue
        counts = np.arange(1, int(max_c) + 1, dtype=np.float64)
        log_counts = np.log2(counts)
        mu = float(fit.gaussian_mean[g])
        scale = float(fit.gaussian_std[g])
        rate = float(fit.poisson_rate[g])
        w_pois = float(fit.mix_probs[g, 0])
        w_norm = float(fit.mix_probs[g, 1])
        log_norm = -_LOG_SQRT_2PI - np.log(scale) - 0.5 * ((log_counts - mu) / scale) ** 2
        log_pois = log_counts * np.log(rate) - rate - _np_gammaln(log_counts + 1.0)
        log_num = np.log(w_norm) + log_norm
        log_den = np.logaddexp(log_num, np.log(w_pois) + log_pois)
        prob_normal = np.exp(log_num - log_den)
        above = np.where(prob_normal > 0.5)[0]
        if above.size > 0:
            thresholds[g] = float(counts[above[0]])
    return thresholds
