"""JAX backend for scGen, vendored from scvi-tools 1.4.0.

scvi-tools 1.5.0 removed its JAX backend entirely (scverse/scvi-tools#3786), migrating to
PyTorch and MLX.
pertpy's scGen is a JAX/Flax implementation built on top of that backend, so the pieces it
relied on (``JaxBaseModuleClass``, ``flax_configure``, ``JaxTrainingPlan``, ``JaxModuleInit`` and
``JaxTrainingMixin``) are vendored here to keep scGen working and numerically identical without
pinning scvi-tools below 1.5.0.

Everything that still ships in scvi-tools (``TrainingPlan``, ``TrainRunner``, ``DataSplitter``,
``LossOutput`` etc.) is imported rather than copied, so only the removed JAX-specific code lives
here.
"""

# This module is vendored, largely verbatim, from scvi-tools 1.4.0, so it is exempt from type
# checking rather than annotated to pertpy's standards.
# mypy: ignore-errors

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import field
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Literal

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training import train_state
from lightning.pytorch.callbacks import Callback
from scvi import REGISTRY_KEYS, settings
from scvi.dataloaders import AnnDataLoader, DataSplitter
from scvi.model._utils import get_max_epochs_heuristic, parse_device_args
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils import dependencies
from scvi.utils._docstrings import devices_dsp

if TYPE_CHECKING:
    from numpyro.distributions import Distribution
    from scvi._types import LossRecord, Tensor

logger = logging.getLogger(__name__)

JaxOptimizerCreator = Callable[[], optax.GradientTransformation]


@dependencies("jax")
def device_selecting_PRNGKey(use_cpu: bool = True) -> Callable:
    """Returns a PRNGKey that is either on CPU or GPU."""
    # if key is generated on CPU, model params will be on CPU
    import jax
    from jax import random

    if use_cpu is True:

        def key(i: int):
            return jax.device_put(random.PRNGKey(i), jax.devices("cpu")[0])
    else:
        # dummy function
        def key(i: int):
            return random.PRNGKey(i)

    return key


def _parse_jax_device(
    accelerator: str,
    devices: int | list[int] | str,
    validate_single_device: bool,
):
    """Resolve a JAX device from Lightning-style accelerator/devices arguments.

    scvi-tools 1.5.0 removed the ``return_device="jax"`` branch of
    :func:`scvi.model._utils.parse_device_args` together with its JAX backend, so the accelerator
    parsing is reused with ``return_device="torch"`` and only the JAX device lookup is vendored,
    matching scvi-tools 1.4.0 behaviour.
    """
    import jax

    _accelerator, _devices, _ = parse_device_args(
        accelerator,
        devices,
        return_device="torch",
        validate_single_device=validate_single_device,
    )

    if isinstance(_devices, list):
        device_idx = _devices[0]
    elif isinstance(_devices, str) and "," in _devices:
        device_idx = _devices.split(",")[0]
    else:
        device_idx = _devices

    if _accelerator == "cpu":
        return jax.devices("cpu")[0]
    if _accelerator == "mps":
        return jax.devices("METAL")[device_idx]  # MPS-JAX
    return jax.devices(_accelerator)[device_idx]


def flax_configure(cls: nn.Module) -> Callable:
    """Decorator to raise an error if a boolean `training` param is missing in the call."""
    original_init = cls.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        self.configure()
        original_init(self, *args, **kwargs)
        if not isinstance(self.training, bool):
            raise ValueError("Custom sublclasses must have a training parameter.")

    cls.__init__ = init
    return cls


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param


@flax.struct.dataclass
class LossOutput:
    """Loss signature for Jax models.

    This class provides an organized way to record the model loss, as well as
    the components of the ELBO. This may also be used in MLE, MAP, EM methods.
    The loss is used for backpropagation during inference. The other parameters
    are used for logging/early stopping during inference.

    scvi-tools 1.5.0 kept only the PyTorch :class:`~scvi.module.base.LossOutput` (a plain
    dataclass) and dropped this JAX-registered ``flax.struct`` variant together with its JAX
    backend.
    The ``flax.struct`` registration is what makes the loss a valid JAX pytree so it can be
    returned from a jitted training step, so it is vendored here.

    Parameters
    ----------
    loss
        Tensor with loss for minibatch. Should be one dimensional with one value.
        Note that loss should be in an array/tensor and not a float.
    reconstruction_loss
        Reconstruction loss for each observation in the minibatch. If a tensor, converted to
        a dictionary with key "reconstruction_loss" and value as tensor.
    kl_local
        KL divergence associated with each observation in the minibatch. If a tensor, converted to
        a dictionary with key "kl_local" and value as tensor.
    kl_global
        Global KL divergence term. Should be one dimensional with one value. If a tensor, converted
        to a dictionary with key "kl_global" and value as tensor.
    classification_loss
        Classification loss.
    logits
        Logits for classification.
    true_labels
        True labels for classification.
    extra_metrics
        Additional metrics can be passed as arrays/tensors or dictionaries of
        arrays/tensors.
    n_obs_minibatch
        Number of observations in the minibatch. If None, will be inferred from
        the shape of the reconstruction_loss tensor.
    """

    loss: LossRecord
    reconstruction_loss: LossRecord | None = None
    kl_local: LossRecord | None = None
    kl_global: LossRecord | None = None
    classification_loss: LossRecord | None = None
    logits: Tensor | None = None
    true_labels: Tensor | None = None
    extra_metrics: dict[str, Tensor] | None = field(default_factory=dict)
    n_obs_minibatch: int | None = None
    reconstruction_loss_sum: Tensor = field(default=None)
    kl_local_sum: Tensor = field(default=None)
    kl_global_sum: Tensor = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "loss", self.dict_sum(self.loss))

        if self.n_obs_minibatch is None and self.reconstruction_loss is None:
            raise ValueError("Must provide either n_obs_minibatch or reconstruction_loss")

        default = 0 * self.loss
        if self.reconstruction_loss is None:
            object.__setattr__(self, "reconstruction_loss", default)
        if self.kl_local is None:
            object.__setattr__(self, "kl_local", default)
        if self.kl_global is None:
            object.__setattr__(self, "kl_global", default)

        object.__setattr__(self, "reconstruction_loss", self._as_dict("reconstruction_loss"))
        object.__setattr__(self, "kl_local", self._as_dict("kl_local"))
        object.__setattr__(self, "kl_global", self._as_dict("kl_global"))
        object.__setattr__(
            self,
            "reconstruction_loss_sum",
            self.dict_sum(self.reconstruction_loss).sum(),
        )
        object.__setattr__(self, "kl_local_sum", self.dict_sum(self.kl_local).sum())
        object.__setattr__(self, "kl_global_sum", self.dict_sum(self.kl_global))

        if self.reconstruction_loss is not None and self.n_obs_minibatch is None:
            rec_loss = self.reconstruction_loss
            object.__setattr__(self, "n_obs_minibatch", list(rec_loss.values())[0].shape[0])

        if self.classification_loss is not None and (self.logits is None or self.true_labels is None):
            raise ValueError("Must provide `logits` and `true_labels` if `classification_loss` is provided.")

    @staticmethod
    def dict_sum(dictionary: dict[str, Tensor] | Tensor):
        """Sum over elements of a dictionary."""
        if isinstance(dictionary, dict):
            return sum(dictionary.values())
        else:
            return dictionary

    @property
    def extra_metrics_keys(self) -> Iterable[str]:
        """Keys for extra metrics."""
        return self.extra_metrics.keys()

    def _as_dict(self, attr_name: str):
        attr = getattr(self, attr_name)
        if isinstance(attr, dict):
            return attr
        else:
            return {attr_name: attr}


class TrainStateWithState(train_state.TrainState):
    """TrainState with state attribute."""

    state: dict[str, Any]


class JaxBaseModuleClass(flax.linen.Module):
    """Abstract class for Jax-based scvi-tools modules.

    The :class:`~pertpy.tools._scgen._jax.JaxBaseModuleClass` provides an interface for Jax-backed
    modules consistent with the :class:`~scvi.module.base.BaseModuleClass`.

    Any subclass must have a `training` parameter in its constructor, as well as
    use the `@flax_configure` decorator.

    Children of :class:`~pertpy.tools._scgen._jax.JaxBaseModuleClass` should
    use the instance attribute ``self.training`` to appropriately modify
    the behavior of the model whether it is in training or evaluation mode.
    """

    def configure(self) -> None:
        """Add necessary attrs."""
        self.training: bool | None = None
        self.train_state: TrainStateWithState | None = None
        self.seed = settings.seed if settings.seed is not None else 0
        self.seed_rng = device_selecting_PRNGKey()(self.seed)
        self._set_rngs()

    @abstractmethod
    def setup(self):
        """Flax setup method.

        With scvi-tools we prefer to use the setup parameterization of
        flax.linen Modules. This lends the interface to be more like
        PyTorch. More about this can be found here:

        https://flax.readthedocs.io/en/latest/design_notes/setup_or_nncompact.html
        """

    @property
    @abstractmethod
    def required_rngs(self):
        """Returns a tuple of rng sequence names required for this Flax module."""
        return ("params",)

    def __call__(
        self,
        tensors: dict[str, jnp.ndarray],
        get_inference_input_kwargs: dict | None = None,
        get_generative_input_kwargs: dict | None = None,
        inference_kwargs: dict | None = None,
        generative_kwargs: dict | None = None,
        loss_kwargs: dict | None = None,
        compute_loss=True,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, LossOutput]:
        """Forward pass through the network.

        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for ``_get_inference_input()``
        get_generative_input_kwargs
            Keyword args for ``_get_generative_input()``
        inference_kwargs
            Keyword args for ``inference()``
        generative_kwargs
            Keyword args for ``generative()``
        loss_kwargs
            Keyword args for ``loss()``
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        return _generic_forward(
            self,
            tensors,
            inference_kwargs,
            generative_kwargs,
            loss_kwargs,
            get_inference_input_kwargs,
            get_generative_input_kwargs,
            compute_loss,
        )

    @abstractmethod
    def _get_inference_input(self, tensors: dict[str, jnp.ndarray], **kwargs):
        """Parse tensors dictionary for inference related values."""

    @abstractmethod
    def _get_generative_input(
        self,
        tensors: dict[str, jnp.ndarray],
        inference_outputs: dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Parse tensors dictionary for generative related values."""

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> dict[str, jnp.ndarray | Distribution]:
        """Run the recognition model.

        In the case of variational inference, this function will perform steps related to
        computing variational distribution parameters. In a VAE, this will involve running
        data through encoder networks.

        This function should return a dictionary with str keys and :class:`~jnp.ndarray` values
        """

    @abstractmethod
    def generative(self, *args, **kwargs) -> dict[str, jnp.ndarray | Distribution]:
        """Run the generative model.

        This function should return the parameters associated with the likelihood of the data.
        This is typically written as :math:`p(x|z)`.

        This function should return a dictionary with str keys and :class:`~jnp.ndarray` values
        """

    @abstractmethod
    def loss(self, *args, **kwargs) -> LossOutput:
        """Compute the loss for a minibatch of data.

        This function uses the outputs of the inference and generative functions to compute
        a loss. This many optionally include other penalty terms, which should be computed here

        This function should return an object of type :class:`~scvi.module.base.LossOutput`.
        """

    @property
    def device(self):
        devices = self.seed_rng.devices()
        if len(devices) > 1:
            raise RuntimeError("Module rng on multiple devices.")
        return next(iter(devices))

    def train(self):
        """Switch to train mode. Emulates Pytorch's interface."""
        self.training = True

    def eval(self):
        """Switch to evaluation mode. Emulates Pytorch's interface."""
        self.training = False

    @property
    def rngs(self) -> dict[str, jnp.ndarray]:
        """Dictionary of RNGs mapping required RNG name to RNG values.

        Calls ``self._split_rngs()`` resulting in newly generated RNGs on
        every reference to ``self.rngs``.
        """
        return self._split_rngs()

    def _set_rngs(self):
        """Creates RNGs split off of the seed RNG for each RNG required by the module."""
        from jax import random

        required_rngs = self.required_rngs
        rng_keys = random.split(self.seed_rng, num=len(required_rngs) + 1)
        self.seed_rng, module_rngs = rng_keys[0], rng_keys[1:]
        self._rngs = {k: module_rngs[i] for i, k in enumerate(required_rngs)}

    def _split_rngs(self):
        """Regenerates the current set of RNGs and returns newly split RNGs.

        Importantly, this method does not reuse RNGs in future references to ``self.rngs``.
        """
        from jax import random

        new_rngs = {}
        ret_rngs = {}
        for k, v in self._rngs.items():
            new_rngs[k], ret_rngs[k] = random.split(v)
        self._rngs = new_rngs
        return ret_rngs

    @property
    def params(self) -> dict[str, Any]:
        self._check_train_state_is_not_none()
        return self.train_state.params

    @property
    def state(self) -> dict[str, Any]:
        self._check_train_state_is_not_none()
        return self.train_state.state

    def state_dict(self) -> dict[str, Any]:
        """Returns a serialized version of the train state as a dictionary."""
        self._check_train_state_is_not_none()
        return flax.serialization.to_state_dict(self.train_state)

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load a state dictionary into a train state."""
        if self.train_state is None:
            raise RuntimeError("Train state is not set. Train for one iteration prior to loading state dict.")
        self.train_state = flax.serialization.from_state_dict(self.train_state, state_dict)

    def to(self, device: jax.Device):
        """Move module to device."""
        import jax

        if device is not self.device:
            if self.train_state is not None:
                self.train_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, device), self.train_state)

            self.seed_rng = jax.device_put(self.seed_rng, device)
            self._rngs = jax.device_put(self._rngs, device)

    def _check_train_state_is_not_none(self):
        if self.train_state is None:
            raise RuntimeError("Train state is not set. Module has not been trained.")

    def as_bound(self) -> JaxBaseModuleClass:
        """Module bound with parameters learned from training."""
        return self.bind(
            {"params": self.params, **self.state},
            rngs=self.rngs,
        )

    def get_jit_inference_fn(
        self,
        get_inference_input_kwargs: dict[str, Any] | None = None,
        inference_kwargs: dict[str, Any] | None = None,
    ) -> Callable[[dict[str, jnp.ndarray], dict[str, jnp.ndarray]], dict[str, jnp.ndarray]]:
        """Create a method to run inference using the bound module.

        Parameters
        ----------
        get_inference_input_kwargs
            Keyword arguments to pass to subclass `_get_inference_input`
        inference_kwargs
            Keyword arguments  for subclass `inference` method

        Returns:
        -------
        A callable taking rngs and array_dict as input and returning the output
        of the `inference` method. This callable runs `_get_inference_input`.
        """
        import jax

        vars_in = {"params": self.params, **self.state}
        get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
        inference_kwargs = _get_dict_if_none(inference_kwargs)

        @jax.jit
        def _run_inference(rngs, array_dict):
            module = self.clone()
            inference_input = module._get_inference_input(array_dict)
            out = module.apply(
                vars_in,
                rngs=rngs,
                method=module.inference,
                **inference_input,
                **inference_kwargs,
            )
            return out

        return _run_inference

    @staticmethod
    def on_load(model, **kwargs):
        """Callback function run in :meth:`~scvi.model.base.BaseModelClass.load`.

        Run one training step prior to loading state dict in order to initialize params.
        """
        old_history = model.history_.copy()
        model.train(max_steps=1)
        model.history_ = old_history

    @staticmethod
    def as_numpy_array(x: jnp.ndarray):
        """Converts a jax device array to a numpy array."""
        import jax

        return np.array(jax.device_get(x))


def _generic_forward(
    module,
    tensors,
    inference_kwargs,
    generative_kwargs,
    loss_kwargs,
    get_inference_input_kwargs,
    get_generative_input_kwargs,
    compute_loss,
):
    """Core of the forward call shared by PyTorch- and Jax-based modules."""
    inference_kwargs = _get_dict_if_none(inference_kwargs)
    generative_kwargs = _get_dict_if_none(generative_kwargs)
    loss_kwargs = _get_dict_if_none(loss_kwargs)
    get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
    get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)
    if not ("latent_qzm" in tensors and "latent_qzv" in tensors):
        # Remove full_forward_pass if not minified model
        get_inference_input_kwargs.pop("full_forward_pass", None)

    inference_inputs = module._get_inference_input(tensors, **get_inference_input_kwargs)
    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)
    generative_inputs = module._get_generative_input(tensors, inference_outputs, **get_generative_input_kwargs)
    generative_outputs = module.generative(**generative_inputs, **generative_kwargs)
    if compute_loss:
        losses = module.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
        return inference_outputs, generative_outputs, losses
    else:
        return inference_outputs, generative_outputs


class JaxTrainingPlan(TrainingPlan):
    """Lightning module task to train Jax scvi-tools modules.

    Parameters
    ----------
    module
        An instance of :class:`~pertpy.tools._scgen._jax.JaxBaseModuleClass`.
    optimizer
        One of "Adam", "AdamW", or "Custom", which requires a custom
        optimizer creator callable to be passed via `optimizer_creator`.
    optimizer_creator
        A callable returning a :class:`~optax.GradientTransformation`.
        This allows using any optax optimizer with custom hyperparameters.
    lr
        Learning rate used for optimization, when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    max_norm
        Max global norm of gradients for gradient clipping.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from
        `min_kl_weight` to `max_kl_weight`. Only activated when `n_epochs_kl_warmup` is
        set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from `min_kl_weight` to
        `max_kl_weight`. Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        module: JaxBaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: JaxOptimizerCreator | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        eps: float = 0.01,
        max_norm: float | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 400,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            **loss_kwargs,
        )
        self.max_norm = max_norm
        self.automatic_optimization = False
        self._dummy_param = torch.nn.Parameter(torch.Tensor([0.0]))

    def get_optimizer_creator(self) -> JaxOptimizerCreator:
        """Get optimizer creator for the model."""
        clip_by = optax.clip_by_global_norm(self.max_norm) if self.max_norm else optax.identity()
        if self.optimizer_name == "Adam":
            # Replicates PyTorch Adam defaults
            optim = optax.chain(
                clip_by,
                optax.add_decayed_weights(weight_decay=self.weight_decay),
                optax.adam(self.lr, eps=self.eps),
            )
        elif self.optimizer_name == "AdamW":
            optim = optax.chain(
                clip_by,
                optax.clip_by_global_norm(self.max_norm),
                optax.adamw(self.lr, eps=self.eps, weight_decay=self.weight_decay),
            )
        elif self.optimizer_name == "Custom":
            optim = self.optimizer_creator
        else:
            raise ValueError("Optimizer not understood.")

        return lambda: optim

    def set_train_state(self, params, state=None):
        """Set the state of the module."""
        if self.module.train_state is not None:
            return
        optimizer = self.get_optimizer_creator()()
        train_state = TrainStateWithState.create(
            apply_fn=self.module.apply,
            params=params,
            tx=optimizer,
            state=state,
        )
        self.module.train_state = train_state

    @staticmethod
    @jax.jit
    def jit_training_step(
        state: TrainStateWithState,
        batch: dict[str, np.ndarray],
        rngs: dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Jit training step."""

        def loss_fn(params):
            # state can't be passed here
            vars_in = {"params": params, **state.state}
            outputs, new_model_state = state.apply_fn(
                vars_in, batch, rngs=rngs, mutable=list(state.state.keys()), **kwargs
            )
            loss_output = outputs[2]
            loss = loss_output.loss
            return loss, (loss_output, new_model_state)

        (loss, (loss_output, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads, state=new_model_state)
        return new_state, loss, loss_output

    def training_step(self, batch, batch_idx):
        """Training step for Jax."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        self.module.train()
        self.module.train_state, _, loss_output = self.jit_training_step(
            self.module.train_state,
            batch,
            self.module.rngs,
            loss_kwargs=self.loss_kwargs,
        )
        loss_output = jax.tree_util.tree_map(
            lambda x: torch.tensor(jax.device_get(x)),
            loss_output,
        )
        # TODO: Better way to get batch size
        self.log(
            "train_loss",
            loss_output.loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
            prog_bar=True,
        )
        self.compute_and_log_metrics(loss_output, self.train_metrics, "train")
        # Update the dummy optimizer to update the global step
        _opt = self.optimizers()
        _opt.step()

    @partial(jax.jit, static_argnums=(0,))
    def jit_validation_step(
        self,
        state: TrainStateWithState,
        batch: dict[str, np.ndarray],
        rngs: dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Jit validation step."""
        vars_in = {"params": state.params, **state.state}
        outputs = self.module.apply(vars_in, batch, rngs=rngs, **kwargs)
        loss_output = outputs[2]

        return loss_output

    def validation_step(self, batch, batch_idx):
        """Validation step for Jax."""
        self.module.eval()
        loss_output = self.jit_validation_step(
            self.module.train_state,
            batch,
            self.module.rngs,
            loss_kwargs=self.loss_kwargs,
        )
        loss_output = jax.tree_util.tree_map(
            lambda x: torch.tensor(jax.device_get(x)),
            loss_output,
        )
        self.log(
            "validation_loss",
            loss_output.loss,
            on_epoch=True,
            batch_size=loss_output.n_obs_minibatch,
        )
        self.compute_and_log_metrics(loss_output, self.val_metrics, "validation")

    @staticmethod
    def transfer_batch_to_device(batch, device, dataloader_idx):
        """Bypass Pytorch Lightning device management."""
        return batch

    def configure_optimizers(self):
        """Shim optimizer for PyTorch Lightning.

        PyTorch Lightning wants to take steps on an optimizer
        returned by this function in order to increment the global
        step count. See PyTorch Lighinting optimizer manual loop.

        Here we provide a shim optimizer that we can take steps on
        at minimal computational cost in order to keep Lightning happy :).
        """
        return torch.optim.Adam([self._dummy_param])

    def optimizer_step(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class JaxModuleInit(Callback):
    """A callback to initialize the Jax-based module."""

    def __init__(self, dataloader: AnnDataLoader = None) -> None:
        super().__init__()
        self.dataloader = dataloader

    @dependencies("flax")
    def on_train_start(self, trainer, pl_module):
        import flax

        module = pl_module.module
        dl = trainer.datamodule.train_dataloader() if self.dataloader is None else self.dataloader
        module_init = module.init(module.rngs, next(iter(dl)))
        state, params = flax.core.pop(module_init, "params")
        pl_module.set_train_state(params, state)


class JaxTrainingMixin:
    """General purpose train method for Jax-backed modules."""

    _data_splitter_cls = DataSplitter
    _training_plan_cls = JaxTrainingPlan
    _train_runner_cls = TrainRunner

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        lr
            Learning rate to use during training.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for ``JaxTrainingPlan``. Keyword arguments
            passed to `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        device = _parse_jax_device(accelerator, devices, validate_single_device=True)
        try:
            self.module.to(device)
            logger.info(
                f"Jax module moved to {device}.Note: Pytorch lightning will show GPU is not being used for the Trainer."
            )
        except RuntimeError:
            logger.debug("No GPU available to Jax.")

        datasplitter_kwargs = datasplitter_kwargs or {}

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            iter_ndarray=True,
            **datasplitter_kwargs,
        )
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        self.training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        if "callbacks" not in trainer_kwargs:
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(JaxModuleInit())

        # Ignore Pytorch Lightning warnings for Jax workarounds.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"pytorch_lightning.*")
            runner = self._train_runner_cls(
                self,
                training_plan=self.training_plan,
                data_splitter=data_splitter,
                max_epochs=max_epochs,
                accelerator="cpu",
                devices="auto",
                **trainer_kwargs,
            )
            runner()

        self.is_trained_ = True
        self.module.eval()
