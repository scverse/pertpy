"""Poisson-Gaussian mixture model for gRNA assignment.

Reimplements the model of ``crispat.ga_poisson_gauss`` (Velten group) in JAX/optax.
The priors, log-density, and per-guide thresholding rule match crispat so that output is directly comparable, while MAP estimation runs for every guide in parallel via ``jax.vmap`` and ``jax.lax.scan``.

Reference: https://github.com/velten-group/crispat
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import gammaln, logsumexp
from scipy.special import gammaln as _np_gammaln

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


@dataclass
class PoissonGaussFit:
    """MAP estimates for the Poisson-Gaussian mixture, one entry per guide."""

    poisson_rate: np.ndarray
    gaussian_mean: np.ndarray
    gaussian_std: np.ndarray
    mix_probs: np.ndarray
    final_loss: np.ndarray


def _log_poisson_pmf(x: jnp.ndarray, rate: jnp.ndarray) -> jnp.ndarray:
    """Continuous Poisson log-density used by crispat: ``log(lam^x exp(-lam) / Gamma(x+1))``."""
    return x * jnp.log(rate) - rate - gammaln(x + 1.0)


def _log_normal_pdf(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return -_LOG_SQRT_2PI - jnp.log(std) - 0.5 * jnp.square((x - mean) / std)


def _unpack(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map unconstrained parameters to ``(rate, mu, scale, w_pois, w_norm)``."""
    log_rate = params[..., 0]
    mu = params[..., 1]
    log_scale = params[..., 2]
    logit_w_norm = params[..., 3]
    rate = jnp.exp(log_rate)
    scale = jnp.exp(log_scale)
    w_norm = jax.nn.sigmoid(logit_w_norm)
    w_pois = 1.0 - w_norm
    return rate, mu, scale, w_pois, w_norm


def _neg_log_joint_one(params: jnp.ndarray, data: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Negative log joint for one guide.

    ``data`` and ``mask`` have shape ``[N]``.
    Priors match crispat: ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)``, ``lam ~ LogNormal(0, 1)``.
    """
    rate, mu, scale, w_pois, w_norm = _unpack(params)

    log_pois = _log_poisson_pmf(data, rate)
    log_norm = _log_normal_pdf(data, mu, scale)
    log_mix = logsumexp(
        jnp.stack([jnp.log(w_pois) + log_pois, jnp.log(w_norm) + log_norm], axis=-1),
        axis=-1,
    )
    log_lik = jnp.sum(jnp.where(mask, log_mix, 0.0))

    # Log priors in the constrained space; additive constants dropped.
    log_prior_w = (0.9 - 1.0) * jnp.log(w_pois) + (0.1 - 1.0) * jnp.log(w_norm)
    log_prior_mu = -0.5 * jnp.square((mu - 3.0) / 2.0)
    log_prior_scale = -jnp.log(scale) - 0.5 * jnp.square(jnp.log(scale) - 2.0)
    log_prior_rate = -jnp.log(rate) - 0.5 * jnp.square(jnp.log(rate))

    return -(log_lik + log_prior_w + log_prior_mu + log_prior_scale + log_prior_rate)


_neg_log_joint_batched = jax.vmap(_neg_log_joint_one, in_axes=(0, 0, 0))


def _total_loss(params: jnp.ndarray, data: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(_neg_log_joint_batched(params, data, mask))


def _sample_init_params(key: jax.Array, n_guides: int) -> jnp.ndarray:
    """Initial unconstrained params for one restart.

    Continuous params start at their prior medians (Pyro's ``init_to_median`` default for Normal/LogNormal), with a small per-seed jitter for diversity across restarts.
    The Dirichlet weight init uses the prior *mean* ``w_norm = 0.1`` instead of the median ~0.01: at the median, the sigmoid is saturated and the gradient vanishes, leaving borderline guides stuck at init.
    Using the mean yields the same MAP for clearly-fittable guides while letting borderline ones escape the local mode.
    """
    n_inner = 10
    k_lam, k_mu, k_scale, k_w = jax.random.split(key, 4)
    log_rate_samples = jax.random.normal(k_lam, (n_inner, n_guides))
    mu_samples = 3.0 + 2.0 * jax.random.normal(k_mu, (n_inner, n_guides))
    log_scale_samples = 2.0 + jax.random.normal(k_scale, (n_inner, n_guides))

    log_rate = jnp.median(log_rate_samples, axis=0)
    mu = jnp.median(mu_samples, axis=0)
    log_scale = jnp.median(log_scale_samples, axis=0)
    # Prior mean for the Normal weight; jitter slightly per seed in [0.02, 0.5].
    w_norm = jnp.clip(0.1 * jnp.exp(0.5 * jax.random.normal(k_w, (n_guides,))), 0.02, 0.5)
    logit_w_norm = jnp.log(w_norm) - jnp.log1p(-w_norm)
    return jnp.stack([log_rate, mu, log_scale, logit_w_norm], axis=-1)


def fit_poisson_gauss_mixture(
    data: np.ndarray | jnp.ndarray,
    mask: np.ndarray | jnp.ndarray,
    *,
    n_iter: int = 500,
    learning_rate: float = 0.01,
    n_init_seeds: int = 10,
    seed: int = 2024,
) -> PoissonGaussFit:
    """Fit Poisson-Gaussian mixtures for many guides in parallel via MAP/SVI.

    Args:
        data: Log2-transformed gRNA counts of shape ``[G, N_max]``; padding entries are ignored via ``mask``.
        mask: Boolean array of shape ``[G, N_max]`` marking valid (non-padded) cells.
        n_iter: Optimizer steps; crispat uses 500 with early stopping but we run a fixed budget so the JAX scan can compile.
        learning_rate: Adam learning rate (matches crispat's default of 0.01).
        n_init_seeds: Number of prior-sampled inits; the best per guide is kept (matches crispat's 10-seed multi-start).
        seed: Top-level RNG seed.

    Returns:
        MAP estimates for all guides.
    """
    data_arr = jnp.asarray(data, dtype=jnp.float32)
    mask_arr = jnp.asarray(mask, dtype=jnp.bool_)
    n_guides = data_arr.shape[0]
    key = jax.random.PRNGKey(seed)

    # Multi-seed init: pick the candidate with lowest negative-log-joint per guide.
    init_keys = jax.random.split(key, n_init_seeds)
    candidate_params = jax.vmap(lambda k: _sample_init_params(k, n_guides))(init_keys)  # [S, G, 4]
    candidate_losses = jax.vmap(lambda p: _neg_log_joint_batched(p, data_arr, mask_arr))(candidate_params)  # [S, G]
    best_seed_per_guide = jnp.argmin(candidate_losses, axis=0)  # [G]
    init_params = candidate_params[best_seed_per_guide, jnp.arange(n_guides)]  # [G, 4]

    optimizer = optax.adam(learning_rate=learning_rate, b1=0.8, b2=0.99)
    opt_state = optimizer.init(init_params)

    grad_fn = jax.grad(_total_loss)

    def step(carry, _):
        params, opt_state, best_loss, best_params = carry
        grads = grad_fn(params, data_arr, mask_arr)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_losses = _neg_log_joint_batched(new_params, data_arr, mask_arr)
        improved = new_losses < best_loss
        best_loss = jnp.where(improved, new_losses, best_loss)
        best_params = jnp.where(improved[:, None], new_params, best_params)
        return (new_params, opt_state, best_loss, best_params), None

    init_carry = (
        init_params,
        opt_state,
        _neg_log_joint_batched(init_params, data_arr, mask_arr),
        init_params,
    )
    (_final_params, _opt_state, best_loss, best_params), _ = jax.lax.scan(step, init_carry, None, length=n_iter)

    rate, mu, scale, w_pois, w_norm = _unpack(best_params)
    mix_probs = jnp.stack([w_pois, w_norm], axis=-1)
    return PoissonGaussFit(
        poisson_rate=np.asarray(rate),
        gaussian_mean=np.asarray(mu),
        gaussian_std=np.asarray(scale),
        mix_probs=np.asarray(mix_probs),
        final_loss=np.asarray(best_loss),
    )


def compute_count_thresholds(fit: PoissonGaussFit, max_counts: np.ndarray) -> np.ndarray:
    """Per-guide assignment threshold matching crispat.

    For each guide, scans integer UMI counts ``t = 1, 2, ..., max_count`` and returns the smallest ``t`` for which ``P(Normal | log2(t)) > 0.5``.
    Returns ``np.nan`` if no such ``t`` exists within ``max_count``.
    """
    max_counts = np.asarray(max_counts, dtype=np.int64)
    n_guides = len(fit.poisson_rate)
    thresholds = np.full(n_guides, np.nan, dtype=np.float64)
    for g, max_c in enumerate(max_counts):
        if max_c < 1:
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
