from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from numpyro import factor, plate, sample
from numpyro.distributions import Categorical, Dirichlet, Exponential, HalfNormal, Normal, Poisson
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
        batch_size: Number of genes to fit simultaneously in batched mode.
    """

    def __init__(
        self,
        num_warmup: int = 50,
        num_samples: int = 100,
        fraction_positive_expected: float = 0.15,
        poisson_rate_prior: float = 0.2,
        gaussian_mean_prior: tuple[float, float] = (3, 2),
        gaussian_std_prior: float = 1,
        batch_size: int = 1,
    ) -> None:
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.fraction_positive_expected = fraction_positive_expected
        self.poisson_rate_prior = poisson_rate_prior
        self.gaussian_mean_prior = gaussian_mean_prior
        self.gaussian_std_prior = gaussian_std_prior
        self.batch_size = batch_size
        self._cached_mcmc: MCMC | None = None
        self._cached_data_size: int | None = None

    @abstractmethod
    def initialize_params(self) -> ParamsDict:
        """Initialize model parameters via sampling from priors.

        Returns:
            Dictionary of sampled parameter values.
        """

    @abstractmethod
    def initialize_params_batched(self) -> ParamsDict:
        """Initialize batched model parameters via sampling from priors.

        Returns:
            Dictionary of sampled parameter values with batch dimension.
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

    @abstractmethod
    def log_likelihood_batched(self, data: jnp.ndarray, params: ParamsDict, mask: jnp.ndarray) -> jnp.ndarray:
        """Calculate log likelihood for batched data.

        Args:
            data: Input data array [batch_size, max_cells].
            params: Current parameter values with batch dimension.
            mask: Boolean mask [batch_size, max_cells] for valid data.

        Returns:
            Log likelihood values for each datapoint [batch_size, max_cells, 2].
        """

    def fit_model(self, data: jnp.ndarray, seed: int = 0) -> MCMC:
        """Fit the mixture model using MCMC.

        Args:
            data: Input data to fit.
            seed: Random seed for reproducibility.

        Returns:
            Fitted MCMC object containing samples.
        """
        if self._cached_mcmc is None or self._cached_data_size != data.shape[0]:
            nuts_kernel = NUTS(self.mixture_model)
            self._cached_mcmc = MCMC(
                nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, progress_bar=False
            )
            self._cached_data_size = data.shape[0]

        self._cached_mcmc.run(PRNGKey(seed), data=data)
        return self._cached_mcmc

    def fit_model_batched(self, data_batch: jnp.ndarray, mask_batch: jnp.ndarray, seed: int = 0) -> MCMC:
        """Fit batched mixture model using MCMC.

        Args:
            data_batch: Batched input data [batch_size, max_cells].
            mask_batch: Boolean mask for valid data [batch_size, max_cells].
            seed: Random seed for reproducibility.

        Returns:
            Fitted MCMC object containing samples.
        """
        nuts_kernel = NUTS(self.mixture_model_batched)
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, progress_bar=False)
        mcmc.run(PRNGKey(seed), data=data_batch, mask=mask_batch)
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

    def run_model_batched(
        self, data_batch: jnp.ndarray, mask_batch: jnp.ndarray, seed: int = 0
    ) -> tuple[list[np.ndarray], list[dict]]:
        """Run batched model fitting and assign components.

        Args:
            data_batch: Batched input data [batch_size, max_cells].
            mask_batch: Boolean mask for valid data [batch_size, max_cells].
            seed: Random seed.

        Returns:
            Tuple of (list of assignment arrays, list of parameter dicts) for each gene in batch.
        """
        mcmc = self.fit_model_batched(data_batch, mask_batch, seed)
        samples = mcmc.get_samples()

        assignments_list = []
        params_list = []

        for i in range(self.batch_size):
            gene_params = {key: samples[key][:, i].mean(axis=0) for key in samples}
            params_list.append(gene_params)

            log_likelihoods = self.log_likelihood(data_batch[i], gene_params)
            guide_assignments = jnp.argmax(log_likelihoods, axis=-1)

            assignments = np.array(["Negative" if assign == 0 else "Positive" for assign in guide_assignments])
            assignments_list.append(assignments)

        return assignments_list, params_list

    def mixture_model(self, data: jnp.ndarray) -> None:
        """Define mixture model structure for NumPyro.

        Args:
            data: Input data array.
        """
        params = self.initialize_params()

        with plate("data", data.shape[0]):
            log_likelihoods = self.log_likelihood(data, params)
            mixture_probs = jnp.exp(log_likelihoods - logsumexp(log_likelihoods, axis=-1, keepdims=True))
            z = sample("z", Categorical(mixture_probs), infer={"enumerate": "parallel"})

            # Observe under selected component
            poisson_ll = Poisson(params["poisson_rate"]).log_prob(data)
            gaussian_ll = Normal(params["gaussian_mean"], params["gaussian_std"]).log_prob(data)
            obs_ll = jnp.where(z == 0, poisson_ll, gaussian_ll)
            factor("obs", obs_ll)

    def mixture_model_batched(self, data: jnp.ndarray, mask: jnp.ndarray) -> None:
        """Define batched mixture model structure for NumPyro.

        Args:
            data: Batched input data [batch_size, max_cells].
            mask: Boolean mask [batch_size, max_cells] for valid data.
        """
        params = self.initialize_params_batched()

        with plate("data", data.shape[1]):
            log_likelihoods = self.log_likelihood_batched(data, params, mask)
            mixture_probs = jnp.exp(log_likelihoods - logsumexp(log_likelihoods, axis=-1, keepdims=True))
            z = sample("z", Categorical(mixture_probs), infer={"enumerate": "parallel"})

            # Observe under selected component
            poisson_ll = Poisson(params["poisson_rate"][..., None]).log_prob(data)
            gaussian_ll = Normal(params["gaussian_mean"][..., None], params["gaussian_std"][..., None]).log_prob(data)
            obs_ll = jnp.where(z == 0, poisson_ll, gaussian_ll)

            obs_ll = jnp.where(mask, obs_ll, 0.0)
            factor("obs", obs_ll)

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

    def log_likelihood_batched(self, data: jnp.ndarray, params: ParamsDict, mask: jnp.ndarray) -> jnp.ndarray:
        """Calculate component-wise log likelihoods for batched data.

        Args:
            data: Batched input data [batch_size, max_cells].
            params: Current parameter values with batch dimension.
            mask: Boolean mask [batch_size, max_cells] for valid data.

        Returns:
            Log likelihood values for each component [batch_size, max_cells, 2].
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
                jnp.log(mix_probs[:, 0:1]) + Poisson(poisson_rate[:, None]).log_prob(data),
                # Gaussian component
                jnp.log(mix_probs[:, 1:2]) + Normal(gaussian_mean[:, None], gaussian_std[:, None]).log_prob(data),
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

    def initialize_params_batched(self) -> ParamsDict:
        """Initialize batched model parameters via prior sampling.

        Returns:
            Dictionary of sampled parameter values with batch dimension.
        """
        params = {}
        with plate("genes", self.batch_size):
            params["poisson_rate"] = sample("poisson_rate", Exponential(self.poisson_rate_prior))
            params["gaussian_mean"] = sample("gaussian_mean", Normal(*self.gaussian_mean_prior))
            params["gaussian_std"] = sample("gaussian_std", HalfNormal(self.gaussian_std_prior))
            params["mix_probs"] = sample(
                "mix_probs",
                Dirichlet(jnp.array([1 - self.fraction_positive_expected, self.fraction_positive_expected])),
            )
        return params
