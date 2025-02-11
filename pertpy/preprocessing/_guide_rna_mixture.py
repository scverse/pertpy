from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS


class MixtureModel(ABC):
    """Template class for 2-component mixture models for guide assignment.

    It handles:
    - Fitting the model to the data
    - Running the model on the data and assigning each data point to a component

    The user needs to implement the following methods:
    - initialize: Initialize the model parameters
    - log_likelihood: Calculate the log-likelihood of the data under the model

    This class has the following parameters:
    - num_warmup: Number of warmup or "burn-in" steps in MCMC
    - num_samples: Number of samples in MCMC. Recommended to be at least 100
    - fraction_positive_expected: Expected fraction of gRNA positive data points
    - poisson_rate_prior: Prior for the Poisson rate of the negative component
    - gaussian_mean_prior: Prior for the Gaussian mean of the positive component
    - gaussian_std_prior: Prior for the Gaussian standard deviation of the positive component
    """

    def __init__(
        self,
        num_warmup=50,
        num_samples=100,
        fraction_positive_expected=0.15,
        poisson_rate_prior=0.2,
        gaussian_mean_prior=(3, 2),
        gaussian_std_prior=1,
    ):
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.fraction_positive_expected = fraction_positive_expected
        self.poisson_rate_prior = poisson_rate_prior
        self.gaussian_mean_prior = gaussian_mean_prior
        self.gaussian_std_prior = gaussian_std_prior

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def log_likelihood(self, data, **params):
        pass

    def fit_model(self, data, seed=0):
        nuts_kernel = NUTS(self.mixture_model)
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, progress_bar=False)
        mcmc.run(random.PRNGKey(seed), data=data)
        return mcmc

    def run_model(self, data, seed=0):
        # Runs MCMS on the model and returns the assignments of the data points
        with numpyro.plate("data", data.shape[0]):
            self.mcmc = self.fit_model(data, seed)
            self.samples = self.mcmc.get_samples()
        self.assignments = self.assignment(self.samples, data)
        return self.assignments

    def mixture_model(self, data=None):
        # Note: numpyro does not natively support discrete latent variables.
        # Hence here we manually marginalize out the discrete latent variable,
        # which requires us to use a log-likelihood formulation.

        params = self.initialize()

        with numpyro.plate("data", data.shape[0]):
            # Calculate log-likelihood for each component
            log_likelihoods = self.log_likelihood(data, **params)

            # Use logsumexp for numerical stability
            log_mixture_likelihood = jax.scipy.special.logsumexp(log_likelihoods, axis=-1)

            # Sample the data from the mixture distribution
            numpyro.sample("obs", dist.Normal(log_mixture_likelihood, 1.0), obs=data)

    def assignment(self, samples, data):
        # Assigns each data point to a component based on the highest log-likelihood
        params = {key: samples[key].mean(axis=0) for key in samples.keys()}
        self.params = params

        log_likelihoods = self.log_likelihood(data, **params)
        guide_assignments = jnp.argmax(log_likelihoods, axis=-1)

        assignments = ["Negative" if assign == 0 else "Positive" for assign in guide_assignments]
        return np.array(assignments)


class Poisson_Gauss_Mixture(MixtureModel):
    def log_likelihood(self, data, poisson_rate, gaussian_mean, gaussian_std, mix_probs):
        # Defines how to calculate the log-likelihood of the data under the model

        # We penalize the model for positioning the Poisson component to the right of the Gaussian component
        # by imposing a soft constraint to penalize the Poisson rate being larger than the Gaussian mean
        # Heuristic regularization term to prevent flipping of the components
        numpyro.factor("separation_penalty", +10 * jnp.heaviside(-poisson_rate + gaussian_mean, 0))

        log_likelihoods = jnp.stack(
            [
                # Poisson component
                jnp.log(mix_probs[0]) + dist.Poisson(poisson_rate).log_prob(data),
                # Gaussian component
                jnp.log(mix_probs[1]) + dist.Normal(gaussian_mean, gaussian_std).log_prob(data),
            ],
            axis=-1,
        )

        return log_likelihoods

    def initialize(self):
        # Initialize the parameters of the model
        params = {}
        params["poisson_rate"] = numpyro.sample("poisson_rate", dist.Exponential(self.poisson_rate_prior))
        params["gaussian_mean"] = numpyro.sample("gaussian_mean", dist.Normal(*self.gaussian_mean_prior))
        params["gaussian_std"] = numpyro.sample("gaussian_std", dist.HalfNormal(self.gaussian_std_prior))
        params["mix_probs"] = numpyro.sample(
            "mix_probs",
            dist.Dirichlet(jnp.array([1 - self.fraction_positive_expected, self.fraction_positive_expected])),
        )
        return params
