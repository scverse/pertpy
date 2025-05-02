from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from numpyro import factor, plate, sample
from numpyro.distributions import Dirichlet, Exponential, HalfNormal, Normal, Poisson
from numpyro.infer import MCMC, NUTS

ParamsDict = Mapping[str, jnp.ndarray]


class MixtureModel(ABC):
    """Abstract base class for 2-component mixture models.

    Args:
        num_warmup: Number of warmup steps for MCMC sampling.
        num_samples: Number of samples to draw after warmup.
        fraction_positive_expected: Prior belief about fraction of positive components.
        poisson_rate_prior: Rate parameter for exponential prior on Poisson component.
        gaussian_mean_prior: Mean and standard deviation for Gaussian prior on positive component mean.
        gaussian_std_prior: Scale parameter for half-normal prior on positive component std.
    """

    def __init__(
        self,
        num_warmup: int = 50,
        num_samples: int = 100,
        fraction_positive_expected: float = 0.15,
        poisson_rate_prior: float = 0.2,
        gaussian_mean_prior: tuple[float, float] = (3, 2),
        gaussian_std_prior: float = 1,
    ) -> None:
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.fraction_positive_expected = fraction_positive_expected
        self.poisson_rate_prior = poisson_rate_prior
        self.gaussian_mean_prior = gaussian_mean_prior
        self.gaussian_std_prior = gaussian_std_prior

    @abstractmethod
    def initialize_params(self) -> ParamsDict:
        """Initialize model parameters via sampling from priors.

        Returns:
            Dictionary of sampled parameter values.
        """

    @abstractmethod
    def log_likelihood(self, data: jnp.ndarray, params: ParamsDict) -> jnp.ndarray:
        """Calculate log likelihood of data under current parameters.

        Args:
            data: Input data array.
            params: Current parameter values.

        Returns:
            Log likelihood values for each datapoint.
        """

    def fit_model(self, data: jnp.ndarray, seed: int = 0) -> MCMC:
        """Fit the mixture model using MCMC.

        Args:
            data: Input data to fit.
            seed: Random seed for reproducibility.

        Returns:
            Fitted MCMC object containing samples.
        """
        nuts_kernel = NUTS(self.mixture_model)
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, progress_bar=False)
        mcmc.run(PRNGKey(seed), data=data)
        return mcmc

    def run_model(self, data: jnp.ndarray, seed: int = 0) -> np.ndarray:
        """Run model fitting and assign components.

        Args:
            data: Input data array.
            seed: Random seed.

        Returns:
            Array of "Positive"/"Negative" assignments for each datapoint.
        """
        self.mcmc = self.fit_model(data, seed)
        self.samples = self.mcmc.get_samples()
        self.assignments = self.assignment(self.samples, data)
        return self.assignments

    def mixture_model(self, data: jnp.ndarray) -> None:
        """Define mixture model structure for NumPyro.

        Args:
            data: Input data array.
        """
        params = self.initialize_params()

        with plate("data", data.shape[0]):
            log_likelihoods = self.log_likelihood(data, params)
            log_mixture_likelihood = logsumexp(log_likelihoods, axis=-1)
            sample("obs", Normal(log_mixture_likelihood, 1.0), obs=data)

    def assignment(self, samples: ParamsDict, data: jnp.ndarray) -> np.ndarray:
        """Assign data points to mixture components.

        Args:
            samples: MCMC samples of parameters.
            data: Input data array.

        Returns:
            Array of component assignments.
        """
        params = {key: samples[key].mean(axis=0) for key in samples}
        self.params = params

        log_likelihoods = self.log_likelihood(data, params)
        guide_assignments = jnp.argmax(log_likelihoods, axis=-1)

        assignments = ["Negative" if assign == 0 else "Positive" for assign in guide_assignments]
        return np.array(assignments)


class PoissonGaussMixture(MixtureModel):
    """Mixture model combining Poisson and Gaussian distributions."""

    def log_likelihood(self, data: np.ndarray, params: ParamsDict) -> jnp.ndarray:
        """Calculate component-wise log likelihoods.

        Args:
            data: Input data array.
            params: Current parameter values.

        Returns:
            Log likelihood values for each component.
        """
        poisson_rate = params["poisson_rate"]
        gaussian_mean = params["gaussian_mean"]
        gaussian_std = params["gaussian_std"]
        mix_probs = params["mix_probs"]

        # We penalize the model for positioning the Poisson component to the right of the Gaussian component
        # by imposing a soft constraint to penalize the Poisson rate being larger than the Gaussian mean
        # Heuristic regularization term to prevent flipping of the components
        factor("separation_penalty", +10 * jnp.heaviside(-poisson_rate + gaussian_mean, 0))

        log_likelihoods = jnp.stack(
            [
                # Poisson component
                jnp.log(mix_probs[0]) + Poisson(poisson_rate).log_prob(data),
                # Gaussian component
                jnp.log(mix_probs[1]) + Normal(gaussian_mean, gaussian_std).log_prob(data),
            ],
            axis=-1,
        )

        return log_likelihoods

    def initialize_params(self) -> ParamsDict:
        """Initialize model parameters via prior sampling.

        Returns:
            Dictionary of sampled parameter values.
        """
        params = {}
        params["poisson_rate"] = sample("poisson_rate", Exponential(self.poisson_rate_prior))
        params["gaussian_mean"] = sample("gaussian_mean", Normal(*self.gaussian_mean_prior))
        params["gaussian_std"] = sample("gaussian_std", HalfNormal(self.gaussian_std_prior))
        params["mix_probs"] = sample(
            "mix_probs",
            Dirichlet(jnp.array([1 - self.fraction_positive_expected, self.fraction_positive_expected])),
        )
        return params
