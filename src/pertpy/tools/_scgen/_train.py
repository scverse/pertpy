"""JAX training loop for scGen, replacing the scvi-tools ``JaxTrainingPlan`` it used before 1.5.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

if TYPE_CHECKING:
    from ._scgenvae import JaxSCGENVAE

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_EPS = 0.01


class TrainStateWithState(train_state.TrainState):
    """Train state that also carries the mutable Flax collections, such as ``batch_stats``."""

    state: Any


def get_max_epochs_heuristic(n_obs: int) -> int:
    """Number of epochs scvi-tools used when ``max_epochs`` is not given."""
    return int(min(round((20000 / n_obs) * 400), 400))


def build_optimizer(
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    eps: float = DEFAULT_EPS,
    max_norm: float | None = None,
) -> optax.GradientTransformation:
    """Adam with decoupled weight decay, as the scvi-tools ``JaxTrainingPlan`` used."""
    clip_by = optax.clip_by_global_norm(max_norm) if max_norm else optax.identity()
    return optax.chain(
        clip_by,
        optax.add_decayed_weights(weight_decay=weight_decay),
        optax.adam(lr, eps=eps),
    )


def _split_indices(
    n_obs: int,
    train_size: float,
    validation_size: float | None,
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_size <= 1.0:
        raise ValueError(f"train_size must be in (0, 1], got {train_size}.")
    n_train = int(round(n_obs * train_size))
    if validation_size is None:
        n_val = n_obs - n_train
    else:
        n_val = int(round(n_obs * validation_size))
        if n_train + n_val > n_obs:
            raise ValueError("train_size + validation_size must not exceed 1.")

    indices = rng.permutation(n_obs) if shuffle else np.arange(n_obs)
    return indices[:n_train], indices[n_train : n_train + n_val]


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    order = rng.permutation(indices) if rng is not None else indices
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


@jax.jit
def _train_step(state: TrainStateWithState, x: jnp.ndarray, rngs: dict[str, jnp.ndarray]):
    def loss_fn(params):
        variables = {"params": params, **state.state}
        outputs, new_model_state = state.apply_fn(variables, x, rngs=rngs, mutable=list(state.state.keys()))
        loss_output = outputs[2]
        return loss_output.loss, (loss_output, new_model_state)

    (_loss, (loss_output, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads, state=new_model_state)
    return new_state, loss_output.loss


def _make_eval_step(eval_apply_fn):
    @jax.jit
    def _eval_step(params, model_state, x: jnp.ndarray, rngs: dict[str, jnp.ndarray]):
        variables = {"params": params, **model_state}
        outputs = eval_apply_fn(variables, x, rngs=rngs)
        return outputs[2].loss

    return _eval_step


def train_module(
    module: JaxSCGENVAE,
    x: np.ndarray,
    *,
    max_epochs: int | None = None,
    batch_size: int = 128,
    train_size: float = 0.9,
    validation_size: float | None = None,
    shuffle_set_split: bool = True,
    early_stopping: bool = False,
    early_stopping_patience: int = 45,
    early_stopping_min_delta: float = 0.0,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    eps: float = DEFAULT_EPS,
    max_norm: float | None = None,
    seed: int = 0,
) -> tuple[TrainStateWithState, dict[str, list[float]]]:
    """Fit ``module`` on the dense ``[cells, genes]`` matrix ``x``.

    The keyword arguments are documented on :meth:`~pertpy.tools.Scgen.train`.

    Returns:
        The final train state and a history with ``train_loss`` and, when a validation set exists,
        ``validation_loss`` per epoch.
    """
    x = np.asarray(x, dtype=np.float32)
    n_obs = x.shape[0]
    if max_epochs is None:
        max_epochs = get_max_epochs_heuristic(n_obs)

    np_rng = np.random.default_rng(seed)
    train_idx, val_idx = _split_indices(n_obs, train_size, validation_size, shuffle_set_split, np_rng)
    if len(train_idx) == 0:
        raise ValueError("The training set is empty. Increase train_size.")

    key = jax.random.PRNGKey(seed)
    key, *init_keys = jax.random.split(key, len(module.required_rngs) + 1)
    variables = module.init(dict(zip(module.required_rngs, init_keys, strict=True)), jnp.asarray(x[:1]))
    params = variables["params"]
    model_state = {k: v for k, v in variables.items() if k != "params"}

    state = TrainStateWithState.create(
        apply_fn=module.apply,
        params=params,
        tx=build_optimizer(lr=lr, weight_decay=weight_decay, eps=eps, max_norm=max_norm),
        state=model_state,
    )
    eval_step = _make_eval_step(module.clone(training=False).apply)

    history: dict[str, list[float]] = {"train_loss": []}
    if len(val_idx) > 0:
        history["validation_loss"] = []

    best_loss = np.inf
    epochs_without_improvement = 0
    for _epoch in range(max_epochs):
        epoch_loss, epoch_cells = 0.0, 0
        for batch_idx in _batches(train_idx, batch_size, np_rng):
            key, dropout_key, z_key = jax.random.split(key, 3)
            state, loss = _train_step(state, jnp.asarray(x[batch_idx]), {"dropout": dropout_key, "z": z_key})
            epoch_loss += float(loss) * len(batch_idx)
            epoch_cells += len(batch_idx)
        history["train_loss"].append(epoch_loss / epoch_cells)

        if len(val_idx) == 0:
            continue

        val_loss, val_cells = 0.0, 0
        for batch_idx in _batches(val_idx, batch_size, None):
            key, z_key = jax.random.split(key)
            loss = eval_step(state.params, state.state, jnp.asarray(x[batch_idx]), {"z": z_key})
            val_loss += float(loss) * len(batch_idx)
            val_cells += len(batch_idx)
        val_loss /= val_cells
        history["validation_loss"].append(val_loss)

        if early_stopping:
            if val_loss < best_loss - early_stopping_min_delta:
                best_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    break

    return state, history
