from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.pyplot import Figure


class MixtureModel(ABC):
    """Template class for 2-component mixture models for guide assignment.

    It handles:
    - Fitting the model to the data
    - Running the model on the data and assigning each data point to a component

    The user needs to implement the following methods:
    - initialize: Initialize the model parameters
    - log_likelihood: Calculate the log-likelihood of the data under the model
    """

    def __init__(self, num_warmup=50, num_samples=100):
        self.num_warmup = num_warmup
        self.num_samples = num_samples

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def log_likelihood(self, data, **params):
        pass

    def fit_model(self, data):
        nuts_kernel = NUTS(self.mixture_model)
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)
        mcmc.run(random.PRNGKey(0), data=data)
        return mcmc

    def run_model(self, data):
        self.mcmc = self.fit_model(data)
        self.samples = self.mcmc.get_samples()
        self.assignments = self.assignment(self.samples, data)
        return self.assignments

    def mixture_model(self, data=None):
        # Note: numpyro does not natively support discrete latent variables.
        # Hence here we manually marginalize out the discrete latent variable

        params = self.initialize()

        with numpyro.plate("data", data.shape[0]):
            # Calculate log-likelihood for each component
            log_likelihoods = self.log_likelihood(data, **params)

            # Use logsumexp for numerical stability
            log_mixture_likelihood = jax.scipy.special.logsumexp(log_likelihoods, axis=-1)

            # Sample the data from the mixture distribution
            numpyro.sample("obs", dist.Normal(log_mixture_likelihood, 1.0), obs=data)

    def assignment(self, samples, data):
        params = {key: samples[key].mean(axis=0) for key in samples.keys()}
        self.params = params

        log_likelihoods = self.log_likelihood(data, **params)
        guide_assignments = jnp.argmax(log_likelihoods, axis=-1)

        assignments = ["Negative" if assign == 0 else "Positive" for assign in guide_assignments]
        return np.array(assignments)


class Poisson_Gauss_Mixture(MixtureModel):
    def log_likelihood(self, data, poisson_rate, gaussian_mean, gaussian_std, mix_probs):
        # Defines how to calculate the log-likelihood of the data under the model

        # Heuristic regularization term to prevent flipping of the components
        regularization = jnp.log(poisson_rate < gaussian_mean) + jnp.log(3 * poisson_rate < gaussian_std)

        log_likelihoods = jnp.stack(
            [
                jnp.log(mix_probs[0]) + dist.Poisson(poisson_rate).log_prob(data) + regularization,  # Poisson component
                jnp.log(mix_probs[1])
                + dist.Normal(gaussian_mean, gaussian_std).log_prob(data)
                + regularization,  # Gaussian component
            ],
            axis=-1,
        )

        return log_likelihoods

    def initialize(self):
        # Initialize the parameters of the model
        params = {}
        params["poisson_rate"] = numpyro.sample("poisson_rate", dist.Exponential(0.5))
        params["gaussian_mean"] = numpyro.sample("gaussian_mean", dist.Normal(4, 6))
        params["gaussian_std"] = numpyro.sample("gaussian_std", dist.HalfNormal(6.0))
        params["mix_probs"] = numpyro.sample("mix_probs", dist.Dirichlet(jnp.array([0.95, 0.05])))
        return params
