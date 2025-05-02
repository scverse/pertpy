from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpyro.distributions as dist
from scvi import REGISTRY_KEYS
from scvi.module.base import JaxBaseModuleClass, LossOutput, flax_configure

from ._base_components import FlaxDecoder, FlaxEncoder


@flax_configure
class JaxSCGENVAE(JaxBaseModuleClass):
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
            activation_fn=nn.activation.leaky_relu,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )

    @property
    def required_rngs(self):
        return ("params", "dropout", "z")

    def _get_inference_input(self, tensors: dict[str, jnp.ndarray]):
        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = {"x": x}
        return input_dict

    def inference(self, x: jnp.ndarray, n_samples: int = 1) -> dict:
        mean, var = self.encoder(x)
        stddev = jnp.sqrt(var)

        qz = dist.Normal(mean, stddev)
        z_rng = self.make_rng("z")
        sample_shape = () if n_samples == 1 else (n_samples,)
        z = qz.rsample(z_rng, sample_shape=sample_shape)

        return {"qz": qz, "z": z}

    def _get_generative_input(
        self,
        tensors: dict[str, jnp.ndarray],
        inference_outputs: dict[str, jnp.ndarray],
    ):
        # x = tensors[REGISTRY_KEYS.X_KEY]
        z = inference_outputs["z"]
        # batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        input_dict = {
            # x=x,
            "z": z,
            # batch_index=batch_index,
        }
        return input_dict

    # def generative(self, x, z, batch_index) -> dict:
    def generative(self, z) -> dict:
        px = self.decoder(z)
        return {"px": px}

    def loss(self, tensors, inference_outputs, generative_outputs):
        x = tensors[REGISTRY_KEYS.X_KEY]
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

    def sample(
        self,
        tensors,
        n_samples=1,
    ):
        inference_kwargs = {"n_samples": n_samples}
        (
            inference_outputs,
            generative_outputs,
        ) = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )
        px = dist.Normal(generative_outputs["px"], 1).sample()
        return px

    def get_reconstruction_loss(self, x, px):
        return jnp.sum((x - px) ** 2)
