from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import linen as nn

if TYPE_CHECKING:
    import jaxlib


class FlaxEncoder(nn.Module):
    n_latent: int = 10
    n_layers: int = 2
    n_hidden: int = 800
    dropout_rate: float = 0.1
    latent_distribution: str = "normal"
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    activation_fn: jaxlib.xla_extension.CompiledFunction = nn.activation.leaky_relu  # type: ignore
    training: bool | None = None
    var_activation: jaxlib.xla_extension.CompiledFunction = jnp.exp  # type: ignore
    # var_eps: float=1e-4,

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool | None = None) -> tuple[float, float]:
        """Forward pass.

        Args:
            x: The input data matrix.
            training: Whether to use running training average.

        Returns:
            Mean and variance.
        """
        training = nn.merge_param("training", self.training, training)
        for _ in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(
                    momentum=0.99,
                    epsilon=0.001,
                    use_running_average=not training,
                )(x)
            x = self.activation_fn(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(x)  # type: ignore
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x) if self.dropout_rate > 0 else x

        mean_x = nn.Dense(self.n_latent)(x)
        logvar_x = nn.Dense(self.n_latent)(x)

        return mean_x, self.var_activation(logvar_x)


class FlaxDecoder(nn.Module):
    n_output: int
    n_layers: int = 1
    n_hidden: int = 128
    dropout_rate: float = 0.2
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    activation_fn: nn.activation = nn.activation.leaky_relu  # type: ignore
    training: bool | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool | None = None) -> jnp.ndarray:  # type: ignore
        """Forward pass.

        Args:
            x: Input data.
            training: Whether to use running training average.

        Returns:
            Decoded data.
        """
        training = nn.merge_param("training", self.training, training)

        for _ in range(self.n_layers):
            x = nn.Dense(self.n_hidden)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(
                    momentum=0.99,
                    epsilon=0.001,
                    use_running_average=not training,
                )(x)
            x = self.activation_fn(x)  # type: ignore
            if self.use_layer_norm:
                x = nn.LayerNorm(x)  # type: ignore
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x) if self.dropout_rate > 0 else x

        x = nn.Dense(self.n_output)(x)  # type: ignore

        return x
