from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax.numpy as jnp
import numpyro.distributions as dist

from ._base_components import FlaxDecoder, FlaxEncoder


class LossOutput(NamedTuple):
    """``reconstruction_loss`` and ``kl_local`` are per cell, ``loss`` is the scalar to differentiate."""

    loss: jnp.ndarray
    reconstruction_loss: jnp.ndarray
    kl_local: jnp.ndarray
    n_obs_minibatch: int


class JaxSCGENVAE(nn.Module):
    """Gaussian VAE used by scGen.

    ``training`` is a module attribute so that ``setup`` can pass it to the encoder and decoder.
    Use ``module.clone(training=False)`` for evaluation.
    """

    n_input: int
    n_hidden: int = 800
    n_latent: int = 10
    n_layers: int = 2
    dropout_rate: float = 0.1
    log_variational: bool = False
    latent_distribution: str = "normal"
    use_batch_norm: str = "both"
    use_layer_norm: str = "none"
    kl_weight: float = 0.00005
    training: bool = True

    def setup(self):
        use_batch_norm_encoder = self.use_batch_norm in ("encoder", "both")
        use_layer_norm_encoder = self.use_layer_norm in ("encoder", "both")
        use_batch_norm_decoder = self.use_batch_norm in ("decoder", "both")
        use_layer_norm_decoder = self.use_layer_norm in ("decoder", "both")

        self.encoder = FlaxEncoder(
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            latent_distribution=self.latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            activation_fn=nn.activation.leaky_relu,
            training=self.training,
        )

        self.decoder = FlaxDecoder(
            n_output=self.n_input,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=nn.activation.leaky_relu,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )

    @property
    def required_rngs(self) -> tuple[str, ...]:
        return ("params", "dropout", "z")

    def inference(self, x: jnp.ndarray, n_samples: int = 1) -> dict:
        mean, var = self.encoder(x)
        stddev = jnp.sqrt(var)

        qz = dist.Normal(mean, stddev)
        z_rng = self.make_rng("z")
        sample_shape = () if n_samples == 1 else (n_samples,)
        z = qz.rsample(z_rng, sample_shape=sample_shape)

        return {"qz": qz, "z": z}

    def generative(self, z: jnp.ndarray) -> dict:
        px = self.decoder(z)
        return {"px": px}

    def __call__(self, x: jnp.ndarray, n_samples: int = 1, compute_loss: bool = True):
        inference_outputs = self.inference(x, n_samples=n_samples)
        generative_outputs = self.generative(inference_outputs["z"])
        if not compute_loss:
            return inference_outputs, generative_outputs
        return inference_outputs, generative_outputs, self.loss(x, inference_outputs, generative_outputs)

    def loss(self, x: jnp.ndarray, inference_outputs: dict, generative_outputs: dict) -> LossOutput:
        px = generative_outputs["px"]
        qz = inference_outputs["qz"]

        kl_divergence_z = dist.kl_divergence(qz, dist.Normal(0, 1)).sum(-1)
        reconst_loss = self.get_reconstruction_loss(px, x)

        weighted_kl_local = self.kl_weight * kl_divergence_z

        loss = jnp.mean(0.5 * reconst_loss + 0.5 * weighted_kl_local)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_divergence_z,
            n_obs_minibatch=x.shape[0],
        )

    def get_reconstruction_loss(self, x: jnp.ndarray, px: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((x - px) ** 2, axis=-1)
