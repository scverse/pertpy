from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as npd
from anndata import AnnData
from jax import config, random
from lamin_utils import logger
from mudata import MuData
from numpyro.infer import Predictive
from rich import print

from pertpy.tools._coda._base_coda import CompositionalModel2, from_scanpy

if TYPE_CHECKING:
    import pandas as pd

config.update("jax_enable_x64", True)


class Sccoda(CompositionalModel2):
    r"""Statistical model for single-cell differential composition analysis with specification of a reference cell type.

    This is the standard scCODA model and recommended for all uses.

    The hierarchical formulation of the model for one sample is:

    .. math::
         y|x &\\sim DirMult(\\phi, \\bar{y}) \\\\
         \\log(\\phi) &= \\alpha + x \\beta \\\\
         \\alpha_k &\\sim N(0, 5) \\quad &\\forall k \\in [K] \\\\
         \\beta_{m, \\hat{k}} &= 0 &\\forall m \\in [M]\\\\
         \\beta_{m, k} &= \\tau_{m, k} \\tilde{\\beta}_{m, k} \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\tau_{m, k} &= \\frac{\\exp(t_{m, k})}{1+ \\exp(t_{m, k})} \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\frac{t_{m, k}}{50} &\\sim N(0, 1) \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\tilde{\\beta}_{m, k} &= \\sigma_m^2 \\cdot \\gamma_{m, k} \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\
         \\sigma_m^2 &\\sim HC(0, 1) \\quad &\\forall m \\in [M] \\\\
         \\gamma_{m, k} &\\sim N(0,1) \\quad &\\forall m \\in [M], k \\in \\{[K] \\smallsetminus \\hat{k}\\} \\\\

    with y being the cell counts and x the covariates.

    For further information, see `scCODA is a Bayesian model for compositional single-cell data analysis`
    (Büttner, Ostner et al., NatComms, 2021)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(
        self,
        adata: AnnData,
        type: Literal["cell_level", "sample_level"],
        generate_sample_level: bool = True,
        cell_type_identifier: str = None,
        sample_identifier: str = None,
        covariate_uns: str | None = None,
        covariate_obs: list[str] | None = None,
        covariate_df: pd.DataFrame | None = None,
        modality_key_1: str = "rna",
        modality_key_2: str = "coda",
    ) -> MuData:
        """Prepare a MuData object for subsequent processing. If type is "cell_level", then create a compositional analysis dataset from the input adata.

        When using ``type="cell_level"``, ``adata`` needs to have a column in ``adata.obs`` that contains the cell type assignment.
        Further, it must contain one column or a set of columns (e.g. subject id, treatment, disease status) that uniquely identify each (statistical) sample.
        Further covariates (e.g. subject age) can either be specified via addidional column names in ``adata.obs``, a key in ``adata.uns``, or as a separate DataFrame.

        Args:
            adata: AnnData object.
            type : Specify the input adata type, which could be either a cell-level AnnData or an aggregated sample-level AnnData.
            generate_sample_level: Whether to generate an AnnData object on the sample level or create an empty AnnData object.
            cell_type_identifier: If type is "cell_level", specify column name in adata.obs that specifies the cell types.
            sample_identifier: If type is "cell_level", specify column name in adata.obs that specifies the sample.
            covariate_uns: If type is "cell_level", specify key for adata.uns, where covariate values are stored.
            covariate_obs: If type is "cell_level", specify list of keys for adata.obs, where covariate values are stored.
            covariate_df: If type is "cell_level", specify dataFrame with covariates.
            modality_key_1: Key to the cell-level AnnData in the MuData object.
            modality_key_2: Key to the aggregated sample-level AnnData object in the MuData object.

        Returns:
            :class:`mudata.MuData` object with cell-level AnnData (`mudata[modality_key_1]`) and aggregated sample-level AnnData (`mudata[modality_key_2]`).

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells,
            >>>                     type="cell_level",
            >>>                     generate_sample_level=True,
            >>>                     cell_type_identifier="cell_label",
            >>>                     sample_identifier="batch", covariate_obs=["condition"])
        """
        if type == "cell_level":
            if generate_sample_level:
                adata_coda = from_scanpy(
                    adata=adata,
                    cell_type_identifier=cell_type_identifier,
                    sample_identifier=sample_identifier,
                    covariate_uns=covariate_uns,
                    covariate_obs=covariate_obs,
                    covariate_df=covariate_df,
                )
            else:
                adata_coda = AnnData()
            mdata = MuData({modality_key_1: adata, modality_key_2: adata_coda})
            mdata.pull_obs()
            mdata.pull_var()
        else:
            mdata = MuData({modality_key_1: AnnData(), modality_key_2: adata})
            mdata.pull_obs()
            mdata.pull_var()
        return mdata

    def prepare(
        self,
        data: AnnData | MuData,
        formula: str,
        reference_cell_type: str = "automatic",
        automatic_reference_absence_threshold: float = 0.05,
        modality_key: str = "coda",
    ) -> AnnData | MuData:
        """Handles data preprocessing, covariate matrix creation, reference selection, and zero count replacement for scCODA.

        Args:
            data: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            formula: R-style formula for building the covariate matrix.
                Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
                To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
            reference_cell_type: Column name that sets the reference cell type.
                Reference the name of a column. If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen.
            automatic_reference_absence_threshold: If using reference_cell_type = "automatic", determine the maximum fraction of zero entries for a cell type
                to be considered as a possible reference cell type.
            modality_key: If data is a MuData object, specify key to the aggregated sample-level AnnData object in the MuData object.

        Returns:
            Return an AnnData (if input data is an AnnData object) or return a MuData (if input data is a MuData object)

            Specifically, parameters have been set:

            - `adata.uns["param_names"]` or `data[modality_key].uns["param_names"]`: List with the names of all tracked latent model parameters (through `npy.sample` or `npy.deterministic`)
            - `adata.uns["scCODA_params"]["model_type"]` or `data[modality_key].uns["scCODA_params"]["model_type"]`: String indicating the model type ("classic")
            - `adata.uns["scCODA_params"]["select_type"]` or `data[modality_key].uns["scCODA_params"]["select_type"]`: String indicating the type of spike_and_slab selection ("spikeslab")

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells,
            >>>                     type="cell_level",
            >>>                     generate_sample_level=True,
            >>>                     cell_type_identifier="cell_label",
            >>>                     sample_identifier="batch",
            >>>                     covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
        """
        if isinstance(data, MuData):
            adata = data[modality_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        adata = super().prepare(adata, formula, reference_cell_type, automatic_reference_absence_threshold)
        # All parameters that are returned for analysis
        adata.uns["scCODA_params"]["param_names"] = [
            "sigma_d",
            "b_offset",
            "ind_raw",
            "alpha",
            "ind",
            "b_raw",
            "beta",
            "concentration",
            "prediction",
        ]

        adata.uns["scCODA_params"]["model_type"] = "classic"
        adata.uns["scCODA_params"]["select_type"] = "spikeslab"

        if is_MuData:
            data.mod[modality_key] = adata
            return data
        else:
            return adata

    def set_init_mcmc_states(self, rng_key: None, ref_index: np.ndarray, sample_adata: AnnData) -> AnnData:  # type: ignore
        """Sets initial MCMC state values for scCODA model.

        Args:
            rng_key: RNG value to be set
            ref_index: Index of reference feature
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.

        Returns:
            Return AnnData object.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells,
            >>>                     type="cell_level",
            >>>                     generate_sample_level=True,
            >>>                     cell_type_identifier="cell_label",
            >>>                     sample_identifier="batch",
            >>>                     covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> adata = sccoda.set_init_mcmc_states(rng_key=42, ref_index=0, sample_adata=mdata["coda"])
        """
        # data dimensions
        N, D = sample_adata.obsm["covariate_matrix"].shape
        P = sample_adata.X.shape[1]

        # Sizes of different parameter matrices
        alpha_size = [P]
        sigma_size = [D, 1]
        beta_nobl_size = [D, P - 1]

        # Initial MCMC states
        rng = np.random.default_rng(seed=rng_key)

        sample_adata.uns["scCODA_params"]["mcmc"]["init_params"] = {
            "sigma_d": np.ones(dtype=np.float64, shape=sigma_size),
            "b_offset": rng.normal(0.0, 1.0, beta_nobl_size),
            "ind_raw": np.zeros(dtype=np.float64, shape=beta_nobl_size),
            "alpha": rng.normal(0.0, 1.0, alpha_size),
        }

        return sample_adata

    def model(  # type: ignore
        self,
        counts: np.ndarray,
        covariates: np.ndarray,
        n_total: np.ndarray,
        ref_index,
        sample_adata: AnnData,
    ):
        """Implements scCODA model in numpyro.

        Args:
            counts: Count data array
            covariates: Covariate matrix
            n_total: Number of counts per sample
            ref_index: Index of reference feature
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.

        Returns:
            predictions (see numpyro documentation for details on models)
        """
        # data dimensions
        N, D = sample_adata.obsm["covariate_matrix"].shape
        P = sample_adata.X.shape[1]

        # numpyro plates for all dimensions
        covariate_axis = npy.plate("covs", D, dim=-2)
        cell_type_axis = npy.plate("ct", P, dim=-1)
        cell_type_axis_nobl = npy.plate("ctnb", P - 1, dim=-1)
        sample_axis = npy.plate("sample", N, dim=-2)

        # Effect priors
        with covariate_axis:
            sigma_d = npy.sample("sigma_d", npd.HalfCauchy(1.0))

        with covariate_axis, cell_type_axis_nobl:
            b_offset = npy.sample("b_offset", npd.Normal(0.0, 1.0))

            # spike-and-slab
            ind_raw = npy.sample("ind_raw", npd.Normal(0.0, 1.0))
            ind_scaled = ind_raw * 50
            ind = npy.deterministic("ind", jnp.exp(ind_scaled) / (1 + jnp.exp(ind_scaled)))

            b_raw = sigma_d * b_offset

            beta_raw = npy.deterministic("b_raw", ind * b_raw)

        with cell_type_axis:
            # Intercepts
            alpha = npy.sample("alpha", npd.Normal(0.0, 5.0))

            # Add 0 effect reference feature
            with covariate_axis:
                beta_full = jnp.concatenate(
                    (beta_raw[:, :ref_index], jnp.zeros(shape=[D, 1]), beta_raw[:, ref_index:]), axis=-1
                )
                beta = npy.deterministic("beta", beta_full)

        # Combine intercepts and effects
        with sample_axis:
            concentrations = npy.deterministic(
                "concentrations", jnp.nan_to_num(jnp.exp(alpha + jnp.matmul(covariates, beta)), 0.0001)
            )

        # Calculate DM-distributed counts
        predictions = npy.sample("counts", npd.DirichletMultinomial(concentrations, n_total), obs=counts)

        return predictions

    def make_arviz(  # type: ignore
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        rng_key=None,
        num_prior_samples: int = 500,
        use_posterior_predictive: bool = True,
    ) -> az.InferenceData:
        """Creates arviz object from model results for MCMC diagnosis.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.
            rng_key: The rng state used for the prior simulation. If None, a random state will be selected.
            num_prior_samples: Number of prior samples calculated.
            use_posterior_predictive: If True, the posterior predictive will be calculated.

        Returns:
            :class:`arviz.InferenceData`: arviz_data with all MCMC information

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells,
            >>>                     type="cell_level",
            >>>                     generate_sample_level=True,
            >>>                     cell_type_identifier="cell_label",
            >>>                     sample_identifier="batch",
            >>>                     covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> arviz_data = sccoda.make_arviz(mdata, num_prior_samples=100)
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data
        if not self.mcmc:
            raise ValueError("No MCMC sampling found. Please run a sampler first!")

        # feature names
        cell_types = sample_adata.var.index.to_list()

        # arviz dimensions
        dims = {
            "alpha": ["cell_type"],
            "sigma_d": ["covariate", "0"],
            "b_offset": ["covariate", "cell_type_nb"],
            "ind_raw": ["covariate", "cell_type_nb"],
            "ind": ["covariate", "cell_type_nb"],
            "b_raw": ["covariate", "cell_type_nb"],
            "beta": ["covariate", "cell_type"],
            "concentrations": ["sample", "cell_type"],
            "predictions": ["sample", "cell_type"],
            "counts": ["sample", "cell_type"],
        }

        # arviz coordinates
        reference_index = sample_adata.uns["scCODA_params"]["reference_index"]
        cell_types_nb = cell_types[:reference_index] + cell_types[reference_index + 1 :]
        coords = {
            "cell_type": cell_types,
            "cell_type_nb": cell_types_nb,
            "covariate": sample_adata.uns["scCODA_params"]["covariate_names"],
            "sample": sample_adata.obs.index,
        }

        dtype = "float64"

        # Prior and posterior predictive simulation
        numpyro_covariates = jnp.array(sample_adata.obsm["covariate_matrix"], dtype=dtype)
        numpyro_n_total = jnp.array(sample_adata.obsm["sample_counts"], dtype=dtype)
        ref_index = jnp.array(sample_adata.uns["scCODA_params"]["reference_index"])

        if rng_key is None:
            rng = np.random.default_rng()
            rng_key = random.key(rng.integers(0, 10000))

        if use_posterior_predictive:
            posterior_predictive = Predictive(self.model, self.mcmc.get_samples())(
                rng_key,
                counts=None,
                covariates=numpyro_covariates,
                n_total=numpyro_n_total,
                ref_index=ref_index,
                sample_adata=sample_adata,
            )
        else:
            posterior_predictive = None

        if num_prior_samples > 0:
            prior = Predictive(self.model, num_samples=num_prior_samples)(
                rng_key,
                counts=None,
                covariates=numpyro_covariates,
                n_total=numpyro_n_total,
                ref_index=ref_index,
                sample_adata=sample_adata,
            )
        else:
            prior = None

        # Create arviz object
        arviz_data = az.from_numpyro(
            self.mcmc, prior=prior, posterior_predictive=posterior_predictive, dims=dims, coords=coords
        )

        return arviz_data

    def run_nuts(
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        num_samples: int = 10000,
        num_warmup: int = 1000,
        rng_key: int = 0,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Examples:
        >>> import pertpy as pt
        >>> haber_cells = pt.dt.haber_2017_regions()
        >>> sccoda = pt.tl.Sccoda()
        >>> mdata = sccoda.load(haber_cells,
        >>>                     type="cell_level",
        >>>                     generate_sample_level=True,
        >>>                     cell_type_identifier="cell_label",
        >>>                     sample_identifier="batch",
        >>>                     covariate_obs=["condition"])
        >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
        >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42).
        """  # noqa: D205
        return super().run_nuts(data, modality_key, num_samples, num_warmup, rng_key, copy, *args, **kwargs)

    run_nuts.__doc__ = CompositionalModel2.run_nuts.__doc__ + run_nuts.__doc__

    def credible_effects(self, data: AnnData | MuData, modality_key: str = "coda", est_fdr: float = None) -> pd.Series:
        """Examples:
        >>> import pertpy as pt
        >>> haber_cells = pt.dt.haber_2017_regions()
        >>> sccoda = pt.tl.Sccoda()
        >>> mdata = sccoda.load(haber_cells,
        >>>                     type="cell_level",
        >>>                     generate_sample_level=True,
        >>>                     cell_type_identifier="cell_label",
        >>>                     sample_identifier="batch",
        >>>                     covariate_obs=["condition"])
        >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
        >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
        >>> credible_effects = sccoda.credible_effects(mdata).
        """  # noqa: D205
        return super().credible_effects(data, modality_key, est_fdr)

    credible_effects.__doc__ = CompositionalModel2.credible_effects.__doc__ + credible_effects.__doc__

    def summary(self, data: AnnData | MuData, extended: bool = False, modality_key: str = "coda", *args, **kwargs):
        """Examples:
        >>> import pertpy as pt
        >>> haber_cells = pt.dt.haber_2017_regions()
        >>> sccoda = pt.tl.Sccoda()
        >>> mdata = sccoda.load(haber_cells,
        >>>                     type="cell_level",
        >>>                     generate_sample_level=True,
        >>>                     cell_type_identifier="cell_label",
        >>>                     sample_identifier="batch",
        >>>                     covariate_obs=["condition"])
        >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
        >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
        >>> sccoda.summary(mdata).
        """  # noqa: D205
        return super().summary(data, extended, modality_key, *args, **kwargs)

    summary.__doc__ = CompositionalModel2.summary.__doc__ + summary.__doc__

    def set_fdr(self, data: AnnData | MuData, est_fdr: float, modality_key: str = "coda", *args, **kwargs):
        """Examples:
        >>> import pertpy as pt
        >>> haber_cells = pt.dt.haber_2017_regions()
        >>> sccoda = pt.tl.Sccoda()
        >>> mdata = sccoda.load(haber_cells,
        >>>                     type="cell_level",
        >>>                     generate_sample_level=True,
        >>>                     cell_type_identifier="cell_label",
        >>>                     sample_identifier="batch",
        >>>                     covariate_obs=["condition"])
        >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
        >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
        >>> sccoda.set_fdr(mdata, est_fdr=0.4).
        """  # noqa: D205
        return super().set_fdr(data, est_fdr, modality_key, *args, **kwargs)

    set_fdr.__doc__ = CompositionalModel2.set_fdr.__doc__ + set_fdr.__doc__
