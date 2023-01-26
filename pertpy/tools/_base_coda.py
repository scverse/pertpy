from __future__ import annotations

from abc import ABC, abstractmethod

import arviz as az
import ete3 as ete
import jax.numpy as jnp
import numpy as np
import numpyro as npy
import pandas as pd
import patsy as pt
import toytree as tt
from anndata import AnnData
from jax import random
from jax._src.prng import PRNGKeyArray
from jax._src.typing import Array
from jax.config import config
from mudata import MuData
from numpyro.infer import HMC, MCMC, NUTS
from rich import box, print
from rich.console import Console
from rich.table import Table
from scipy.cluster import hierarchy as sp_hierarchy

config.update("jax_enable_x64", True)


class CompositionalModel2(ABC):
    """
    General compositional model framework for scCODA-type models

    This class serves as a template for scCODA-style models. It handles:

    - General data preprocessing

    - Inference

    - Result creation

    - Inference algorithms

    An instance of this class has two main attributes.
    `sample_adata` is an `AnnData` object that contains the aggregated counts of N samples and P features (e.g. cell types),
    with N on the `obs` axis and P on the `var` axis. All other information about the model, such as initial parameters,
    references, or the model formula are stored in `sample_adata.uns`.
    After running a numpyro inference algorithm, such as MCMC sampling, the result will be stored in another class attribute.

    Particular models can be implemented as children of this class.
    The following parameters must be set during subclass initialization:

    - `sample_adata.uns["param_names"]`:
    List with the names of all tracked latent model parameters (through `npy.sample` or `npy.deterministic`)

    - `sample_adata.uns["scCODA_params"]["model_type"]`:
    String indicating the model type ("classic" or "tree_agg")

    - `sample_adata.uns["scCODA_params"]["select_type"]`:
    String indicating the type of spike_and_slab selection ("spikeslab" or "sslasso")

    Additionally, a subclass must implement at least these functions (see subclasses for examples):

    - `model`: The model formulation

    - `make_arviz`: A function to generate an arviz result object
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def make_arviz(self, *args, **kwargs):
        pass

    @abstractmethod
    def model(self, *args, **kwargs):
        pass

    def prepare(
        self,
        sample_adata: AnnData,
        formula: str,
        reference_cell_type: str = "automatic",
        automatic_reference_absence_threshold: float = 0.05,
    ) -> AnnData:
        """Handles data preprocessing, covariate matrix creation, reference selection, and zero count replacement.

        Args:
            sample_adata: anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            formula: R-style formula for building the covariate matrix.
                Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
                To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
            reference_cell_type: Column name that sets the reference cell type.
                Reference the name of a column. If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen. Defaults to "automatic".
            automatic_reference_absence_threshold: If using reference_cell_type = "automatic", determine the maximum fraction of zero entries for a cell type
                to be considered as a possible reference cell type. Defaults to 0.05.

        Returns:
            AnnData
        """
        dtype = "float64"

        # Convert count data to float64 (needed for correct inference)
        sample_adata.X = sample_adata.X.astype(dtype)

        # Build covariate matrix from R-like formula, save in obsm
        covariate_matrix = pt.dmatrix(formula, sample_adata.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        sample_adata.obsm["covariate_matrix"] = np.array(covariate_matrix[:, 1:]).astype(dtype)

        cell_types = sample_adata.var.index.to_list()

        # Invoke instance of the correct model depending on reference cell type
        # Automatic reference selection (dispersion-based)
        if reference_cell_type == "automatic":
            percent_zero = np.sum(sample_adata.X == 0, axis=0) / sample_adata.X.shape[0]
            nonrare_ct = np.where(percent_zero < automatic_reference_absence_threshold)[0]

            if len(nonrare_ct) == 0:
                raise ValueError(
                    "No cell types that have large enough presence! Please increase automatic_reference_absence_threshold"
                )

            rel_abun = sample_adata.X / np.sum(sample_adata.X, axis=1, keepdims=True)

            # select reference
            cell_type_disp = np.var(rel_abun, axis=0) / np.mean(rel_abun, axis=0)
            min_var = np.min(cell_type_disp[nonrare_ct])
            ref_index = np.where(cell_type_disp == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            print(f"[bold blue]Automatic reference selection! Reference cell type set to {ref_cell_type}")

        # Column name as reference cell type
        elif reference_cell_type in cell_types:
            ref_index = cell_types.index(reference_cell_type)

        # None of the above: Throw error
        else:
            raise NameError("Reference index is not a valid cell type name or numerical index!")

        # Add pseudocount if zeroes are present.
        if np.count_nonzero(sample_adata.X) != np.size(sample_adata.X):
            print("Zero counts encountered in data! Added a pseudocount of 0.5.")
            sample_adata.X[sample_adata.X == 0] = 0.5

        sample_adata.obsm["sample_counts"] = np.sum(sample_adata.X, axis=1)

        # Check input data
        if covariate_matrix.shape[0] != sample_adata.X.shape[0]:
            row_len_covariate_matrix = sample_adata.obsm["covariate_matrix"].shape[0]
            row_len_sample_adata = sample_adata.X.shape[0]
            raise ValueError(f"Wrong input dimensions X[{row_len_covariate_matrix},:] != y[{row_len_sample_adata},:]")
        if covariate_matrix.shape[0] != len(sample_adata.obsm["sample_counts"]):
            covariate_matrix = sample_adata.obsm["covariate_matrix"]
            len_sample_counts = len(sample_adata.obsm["sample_counts"])
            raise ValueError(f"Wrong input dimensions X[{covariate_matrix},:] != n_total[{len_sample_counts}]")

        # Save important model parameters in uns
        sample_adata.uns["scCODA_params"] = {
            "formula": formula,
            "reference_cell_type": cell_types[ref_index],
            "reference_index": ref_index,
            "automatic_reference_absence_threshold": automatic_reference_absence_threshold,
            "covariate_names": covariate_names,
            "mcmc": {"init_params": []},
        }
        return sample_adata

    def __run_mcmc(
        self,
        sample_adata: AnnData,
        kernel: npy.infer.mcmc.MCMCKernel,
        rng_key: Array | PRNGKeyArray,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Background function that executes any numpyro MCMC algorithm and processes its results

        Args:
            sample_adata: anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            kernel: A `numpyro.infer.mcmc.MCMCKernel` object
            rng_key: The rng state used. If None, a random state will be selected
            copy: Return a copy instead of writing to adata. Defaults to False.
            args: Passed to `numpyro.infer.mcmc.MCMC`
            kwargs: Passed to `numpyro.infer.mcmc.MCMC`

        Returns:
            Saves all results into `sample_adata` and generates `self.mcmc` as a class attribute. If copy==True, return a copy of adata.
        """
        dtype = "float64"
        # The tracked MCMC parameters for diagnostic checks
        extra_fields = (
            "potential_energy",
            "num_steps",
            "adapt_state.step_size",
            "accept_prob",
            "mean_accept_prob",
        )

        # Convert all data needed for modeling to numpyro arrays
        numpyro_counts = jnp.array(sample_adata.X, dtype=dtype)
        numpyro_covariates = jnp.array(sample_adata.obsm["covariate_matrix"], dtype=dtype)
        numpyro_n_total = jnp.array(sample_adata.obsm["sample_counts"], dtype=dtype)

        # Create mcmc attribute and run inference
        self.mcmc = MCMC(kernel, *args, **kwargs)
        self.mcmc.run(
            rng_key,
            numpyro_counts,
            numpyro_covariates,
            numpyro_n_total,
            jnp.array(sample_adata.uns["scCODA_params"]["reference_index"]),
            sample_adata,
            init_params=sample_adata.uns["scCODA_params"]["mcmc"]["init_params"],
            extra_fields=extra_fields,
        )

        # Set acceptance rate and save sampled values to `sample_adata.uns`
        sample_adata.uns["scCODA_params"]["mcmc"]["acceptance_rate"] = np.array(self.mcmc.last_state.mean_accept_prob)
        samples = self.mcmc.get_samples()
        for k, v in samples.items():
            samples[k] = np.array(v)
        sample_adata.uns["scCODA_params"]["mcmc"]["samples"] = samples

        # Evaluate results and create result dataframes (based on tree-aggregation or not)
        if sample_adata.uns["scCODA_params"]["model_type"] == "classic":
            intercept_df, effect_df = self.summary_prepare(sample_adata)  # type: ignore
        elif sample_adata.uns["scCODA_params"]["model_type"] == "tree_agg":
            intercept_df, effect_df, node_df = self.summary_prepare(sample_adata)  # type: ignore
            # Save node df in `sample_adata.uns`
            sample_adata.uns["scCODA_params"]["node_df"] = node_df
        else:
            raise ValueError("No valid model type!")

        # Save intercept and effect dfs in `sample_adata.varm` (one effect df per covariate)
        sample_adata.varm["intercept_df"] = intercept_df
        for cov in effect_df.index.get_level_values("Covariate"):
            sample_adata.varm[f"effect_df_{cov}"] = effect_df.loc[cov, :]
        if copy:
            return sample_adata

    def run_nuts(
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        num_samples: int = 10000,
        num_warmup: int = 1000,
        rng_key: int = None,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Run No-U-turn sampling (Hoffman and Gelman, 2014), an efficient version of Hamiltonian Monte Carlo sampling to infer optimal model parameters.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            num_samples: Number of sampled values after burn-in. Defaults to 10000.
            num_warmup: Number of burn-in (warmup) samples. Defaults to 1000.
            rng_key: The rng state used. If None, a random state will be selected. Defaults to None.
            copy: Return a copy instead of writing to adata. Defaults to False.

        Returns:
            Calls `self.__run_mcmc`
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data
        if copy:
            sample_adata = sample_adata.copy()

        # Set rng key if needed
        if rng_key is None:
            rng_key_array = random.PRNGKey(np.random.randint(0, 10000))
        else:
            rng_key_array = random.PRNGKey(rng_key)
        sample_adata.uns["scCODA_params"]["mcmc"]["rng_key"] = np.array(rng_key_array)

        # Set up NUTS kernel
        nuts_kernel = NUTS(self.model, *args, **kwargs)
        # Save important parameters in `sample_adata.uns`
        sample_adata.uns["scCODA_params"]["mcmc"]["num_samples"] = num_samples
        sample_adata.uns["scCODA_params"]["mcmc"]["num_warmup"] = num_warmup
        sample_adata.uns["scCODA_params"]["mcmc"]["algorithm"] = "NUTS"

        return self.__run_mcmc(
            sample_adata, nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, rng_key=rng_key_array, copy=copy
        )

    def run_hmc(
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        num_samples: int = 20000,
        num_warmup: int = 5000,
        rng_key=None,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Run standard Hamiltonian Monte Carlo sampling (Neal, 2011) to infer optimal model parameters.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            num_samples: Number of sampled values after burn-in. Defaults to 20000.
            num_warmup: Number of burn-in (warmup) samples. Defaults to 5000.
            rng_key: The rng state used. If None, a random state will be selected. Defaults to None.
            copy: Return a copy instead of writing to adata. Defaults to False.
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data
        if copy:
            sample_adata = sample_adata.copy()

        # Set rng key if needed
        if rng_key is None:
            rng_key = random.PRNGKey(np.random.randint(0, 10000))
            sample_adata.uns["scCODA_params"]["mcmc"]["rng_key"] = rng_key
        else:
            rng_key = random.PRNGKey(rng_key)

        # Set up HMC kernel
        hmc_kernel = HMC(self.model, *args, **kwargs)

        # Save important parameters in `sample_adata.uns`
        sample_adata.uns["scCODA_params"]["mcmc"]["num_samples"] = num_samples
        sample_adata.uns["scCODA_params"]["mcmc"]["num_warmup"] = num_warmup
        sample_adata.uns["scCODA_params"]["mcmc"]["algorithm"] = "HMC"

        return self.__run_mcmc(
            sample_adata, hmc_kernel, num_samples=num_samples, num_warmup=num_warmup, rng_key=rng_key, copy=copy
        )

    def summary_prepare(
        self, sample_adata: AnnData, est_fdr: float = 0.05, *args, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates summary dataframes for intercepts, effects and node-level effect (if using tree aggregation).
            This function builds on and supports all functionalities from ``az.summary``.

        Args:
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            est_fdr: Desired FDR value. Defaults to 0.05.
            args: Passed to ``az.summary``
            kwargs: Passed to ``az.summary``

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Intercept, effect and node-level DataFrames

            intercept_df
                Summary of intercept parameters. Contains one row per cell type.

                - Final Parameter: Final intercept model parameter
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Expected sample: Expected cell counts for a sample with no present covariates. See the tutorial for more explanation

            effect_df
                Summary of effect (slope) parameters. Contains one row per covariate/cell type combination.

                - Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Expected sample: Expected cell counts for a sample with only the current covariate set to 1. See the tutorial for more explanation
                - log2-fold change: Log2-fold change between expected cell counts with no covariates and with only the current covariate
                - Inclusion probability: Share of MCMC samples, for which this effect was not set to 0 by the spike-and-slab prior.

            node_df
                Summary of effect (slope) parameters on the tree nodes (features or groups of features). Contains one row per covariate/cell type combination.

                - Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
                - Median: Median of parameter over MCMC chain
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Delta: Decision boundary value - threshold of practical significance
                - Is credible: Boolean indicator whether effect is credible
        """
        # Get model and effect selection types
        select_type = sample_adata.uns["scCODA_params"]["select_type"]
        model_type = sample_adata.uns["scCODA_params"]["model_type"]

        # Create arviz summary for intercepts, effects and node effects
        if model_type == "tree_agg":
            var_names = ["alpha", "b_tilde", "beta"]
        elif model_type == "classic":
            var_names = ["alpha", "beta"]
        else:
            raise ValueError("No valid model type!")

        summ = az.summary(data=self.make_arviz(sample_adata, num_prior_samples=0, use_posterior_predictive=False), var_names=var_names, kind="stats", stat_funcs={"median": np.median}, *args, **kwargs)  # type: ignore

        effect_df = summ.loc[summ.index.str.match("|".join([r"beta\["]))].copy()
        intercept_df = summ.loc[summ.index.str.match("|".join([r"alpha\["]))].copy()

        # Build neat index
        cell_types = sample_adata.var.index.to_list()
        covariates = sample_adata.uns["scCODA_params"]["covariate_names"]

        intercept_df.index = pd.Index(cell_types, name="Cell Type")
        effect_df.index = pd.MultiIndex.from_product([covariates, cell_types], names=["Covariate", "Cell Type"])
        intercept_df = self.__complete_alpha_df(sample_adata, intercept_df)

        # Processing only if using tree aggregation
        if model_type == "tree_agg":
            node_df = summ.loc[summ.index.str.match("|".join([r"b_tilde\["]))].copy()

            # Neat index for node df
            node_names = sample_adata.uns["scCODA_params"]["node_names"]
            covariates_node = [x + "_node" for x in covariates]
            node_df.index = pd.MultiIndex.from_product([covariates_node, node_names], names=["Covariate", "Node"])

            # Complete node df
            node_df = self.__complete_node_df(sample_adata, node_df)

            # Complete effect df
            effect_df = self.__complete_beta_df(
                sample_adata,
                intercept_df,
                effect_df,
                target_fdr=est_fdr,
                model_type="tree_agg",
                select_type=select_type,
                node_df=node_df,
            )
        else:
            # Complete effect df
            effect_df = self.__complete_beta_df(
                sample_adata, intercept_df, effect_df, target_fdr=est_fdr, select_type=select_type, model_type="classic"
            )

        # Give nice column names, remove unnecessary columns
        hdis = intercept_df.columns[intercept_df.columns.str.contains("hdi")]
        hdis_new = hdis.str.replace("hdi_", "HDI ")

        # Calculate credible intervals if using classical spike-and-slab
        if select_type == "spikeslab":
            # Credible interval
            ind_post = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["ind"])
            ind_post[ind_post < 1e-3] = np.nan

            b_raw_sel = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["b_raw"]) * ind_post

            res = az.convert_to_inference_data(np.array([b_raw_sel]))

            summary_sel = az.summary(data=res, kind="stats", var_names=["x"], skipna=True, *args, **kwargs)  # type: ignore

            ref_index = sample_adata.uns["scCODA_params"]["reference_index"]
            n_conditions = len(covariates)
            n_cell_types = len(cell_types)

            def insert_row(idx, df, df_insert):
                return pd.concat(
                    [
                        df.iloc[
                            :idx,
                        ],
                        df_insert,
                        df.iloc[
                            idx:,
                        ],
                    ]
                ).reset_index(drop=True)

            for i in range(n_conditions):
                summary_sel = insert_row(
                    (i * n_cell_types) + ref_index,
                    summary_sel,
                    pd.DataFrame.from_dict(data={"mean": [0], "sd": [0], hdis[0]: [0], hdis[1]: [0]}),
                )

            effect_df.loc[:, hdis[0]] = list(summary_sel[hdis[0]])
            effect_df.loc[:, hdis[1]] = list(summary_sel.loc[:, hdis[1]])  # type: ignore
        # For spike-and-slab LASSO, credible intervals are as calculated by `az.summary`
        elif select_type == "sslasso":
            pass
        else:
            raise ValueError("No valid select type!")

        # Select relevant columns and give nice column names for all result dfs, then return them
        intercept_df = intercept_df.loc[:, ["final_parameter", hdis[0], hdis[1], "sd", "expected_sample"]].copy()
        intercept_df = intercept_df.rename(
            columns=dict(
                zip(intercept_df.columns, ["Final Parameter", hdis_new[0], hdis_new[1], "SD", "Expected Sample"])
            )
        )

        if select_type == "sslasso":
            effect_df = effect_df.loc[
                :, ["final_parameter", "median", hdis[0], hdis[1], "sd", "expected_sample", "log_fold"]
            ].copy()
            effect_df = effect_df.rename(
                columns=dict(
                    zip(
                        effect_df.columns,
                        ["Effect", "Median", hdis_new[0], hdis_new[1], "SD", "Expected Sample", "log2-fold change"],
                    )
                )
            )
        else:
            effect_df = effect_df.loc[
                :, ["final_parameter", hdis[0], hdis[1], "sd", "inclusion_prob", "expected_sample", "log_fold"]
            ].copy()
            effect_df = effect_df.rename(
                columns=dict(
                    zip(
                        effect_df.columns,
                        [
                            "Final Parameter",
                            hdis_new[0],
                            hdis_new[1],
                            "SD",
                            "Inclusion probability",
                            "Expected Sample",
                            "log2-fold change",
                        ],
                    )
                )
            )

        if model_type == "tree_agg":
            node_df = node_df.loc[:, ["final_parameter", "median", hdis[0], hdis[1], "sd", "delta", "significant"]].copy()  # type: ignore
            node_df = node_df.rename(
                columns=dict(
                    zip(
                        node_df.columns,
                        ["Final Parameter", "Median", hdis_new[0], hdis_new[1], "SD", "Delta", "Is credible"],
                    )
                )  # type: ignore
            )  # type: ignore

            return intercept_df, effect_df, node_df
        else:
            return intercept_df, effect_df

    def __complete_beta_df(
        self,
        sample_adata: AnnData,
        intercept_df: pd.DataFrame,
        effect_df: pd.DataFrame,
        model_type: str,
        select_type: str,
        target_fdr: float = 0.05,
        node_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Evaluation of MCMC results for effect parameters. This function is only used within self.summary_prepare.
            This function also calculates the posterior inclusion probability for each effect and decides whether effects are significant.

        Args:
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            intercept_df: Intercept summary, see ``summary_prepare``
            effect_df: Effect summary, see ``summary_prepare``
            model_type: String indicating the model type ("classic" or "tree_agg")
            select_type:  String indicating the type of spike_and_slab selection ("spikeslab" or "sslasso")
            target_fdr: Desired FDR value. Defaults to 0.05.
            node_df: If using tree aggregation, the node-level effect DataFrame must be passed. Defaults to None.

        Returns:
            pd.DataFrame:  effect DataFrame with inclusion probability, final parameters, expected sample.
        """
        # Data dimensions
        D = len(effect_df.index.levels[0])
        K = len(effect_df.index.levels[1])

        # Effect processing for different models
        # Classic scCODA (spike-and-slab + no tree aggregation)
        if model_type == "classic" and select_type == "spikeslab":
            beta_inc_prob = []
            beta_nonzero_mean = []

            # Get MCMC samples for parameter "beta"
            beta_raw = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["beta"])

            # Calculate inclusion prob, nonzero mean for every effect
            for j in range(beta_raw.shape[1]):
                for i in range(beta_raw.shape[2]):
                    beta_i_raw = beta_raw[:, j, i]
                    beta_i_raw_nonzero = np.where(np.abs(beta_i_raw) > 1e-3)[0]
                    prob = beta_i_raw_nonzero.shape[0] / beta_i_raw.shape[0]
                    beta_inc_prob.append(prob)
                    if len(beta_i_raw[beta_i_raw_nonzero]) > 0:
                        beta_nonzero_mean.append(beta_i_raw[beta_i_raw_nonzero].mean())
                    else:
                        beta_nonzero_mean.append(0)

            effect_df.loc[:, "inclusion_prob"] = beta_inc_prob
            effect_df.loc[:, "mean_nonzero"] = beta_nonzero_mean

            # Inclusion prob threshold value. Direct posterior probability approach cf. Newton et al. (2004)
            def opt_thresh(result, alpha):
                incs = np.array(result.loc[result["inclusion_prob"] > 0, "inclusion_prob"])
                incs[::-1].sort()

                for c in np.unique(incs):
                    fdr = np.mean(1 - incs[incs >= c])

                    if fdr < alpha:
                        # ceiling with 3 decimals precision
                        c = np.floor(c * 10**3) / 10**3
                        return c, fdr
                return 1.0, 0

            threshold, fdr_ = opt_thresh(effect_df, target_fdr)

            # Save cutoff inclusion probability to scCODA params in uns
            sample_adata.uns["scCODA_params"]["threshold_prob"] = threshold

            # Decide whether betas are significant or not, set non-significant ones to 0
            effect_df.loc[:, "final_parameter"] = np.where(
                effect_df.loc[:, "inclusion_prob"] >= threshold, effect_df.loc[:, "mean_nonzero"], 0
            )

        # tascCODA model (spike-and-slab LASSO + tree aggregation)
        elif select_type == "sslasso" and model_type == "tree_agg":
            # Get ancestor matrix
            A = sample_adata.uns["scCODA_params"]["ancestor_matrix"]

            # Feature-level effects are just node-level effects time ancestor matrix
            effect_df["final_parameter"] = np.matmul(
                np.kron(np.eye(D, dtype=int), A), np.array(node_df["final_parameter"])
            )

        # Get expected sample, log-fold change
        y_bar = np.mean(np.sum(sample_adata.X, axis=1))
        alpha_par = intercept_df.loc[:, "final_parameter"]
        alphas_exp = np.exp(alpha_par)
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values

        beta_mean = np.array(alpha_par)
        for d in range(D):
            beta_d = effect_df.loc[:, "final_parameter"].values[(d * K) : ((d + 1) * K)]
            beta_d = beta_mean + beta_d
            beta_d = np.exp(beta_d)
            beta_d = beta_d / np.sum(beta_d) * y_bar
            if d == 0:
                beta_sample = beta_d
                log_sample = np.log2(beta_d / alpha_sample)
            else:
                beta_sample = np.append(beta_sample, beta_d)
                log_sample = np.append(log_sample, np.log2(beta_d / alpha_sample))
        effect_df.loc[:, "expected_sample"] = beta_sample
        effect_df.loc[:, "log_fold"] = log_sample

        return effect_df

    def __complete_node_df(
        self,
        sample_adata: AnnData,
        node_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Evaluation of MCMC results for node-level effect parameters. This function is only used within self.summary_prepare.
            This function determines whether node-level effects are credible or not

        Args:
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            node_df: Node-level effect summary, see ``summary_prepare``

        Returns:
            pd.DataFrame: node-level effect DataFrame with inclusion threshold, final parameters, significance indicator
        """
        # calculate inclusion threshold
        theta = np.median(np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["theta"]))
        l_0 = sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_0"]
        l_1 = sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_1_scaled"]

        def delta(l_0, l_1, theta):
            p_t = (theta * l_1 / 2) / ((theta * l_1 / 2) + ((1 - theta) * l_0 / 2))
            return 1 / (l_0 - l_1) * np.log(1 / p_t - 1)

        D = len(node_df.index.levels[0])

        # apply inclusion threshold
        deltas = delta(l_0, l_1, theta)
        refs = np.sort(sample_adata.uns["scCODA_params"]["reference_index"])
        deltas = np.insert(deltas, [refs[i] - i for i in range(len(refs))], 0)

        node_df["delta"] = np.tile(deltas, D)
        node_df["significant"] = np.abs(node_df["median"]) > node_df["delta"]
        node_df["final_parameter"] = np.where(node_df.loc[:, "significant"], node_df.loc[:, "median"], 0)

        return node_df

    def __complete_alpha_df(self, sample_adata: AnnData, intercept_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluation of MCMC results for intercepts. This function is only used within self.summary_prepare.

        Args:
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            intercept_df: Intercept summary, see ``summary_prepare``

        Returns:
            pd.DataFrame: intercept DataFrame with expected sample, final parameters
        """
        intercept_df = intercept_df.rename(columns={"mean": "final_parameter"})

        # Get expected sample
        y_bar = np.mean(np.sum(sample_adata.X, axis=1))
        alphas_exp = np.exp(intercept_df.loc[:, "final_parameter"])
        alpha_sample = (alphas_exp / np.sum(alphas_exp) * y_bar).values
        intercept_df.loc[:, "expected_sample"] = alpha_sample

        return intercept_df

    def summary(self, data: AnnData | MuData, extended: bool = False, modality_key: str = "coda", *args, **kwargs):
        """Printing method for the summary.

        Args:
            data: AnnData object or MuData object.
            extended: If True, return the extended summary with additional statistics. Defaults to False.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            args: Passed to az.summary
            kwargs: Passed to az.summary
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("[bold red]When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data
        # Get model and effect selection types
        select_type = sample_adata.uns["scCODA_params"]["select_type"]
        model_type = sample_adata.uns["scCODA_params"]["model_type"]

        # If other than default values for e.g. confidence interval are specified,
        # recalculate them for intercept and effect DataFrames
        if args or kwargs:
            if model_type == "tree_agg":
                intercept_df, effect_df, node_df = self.summary_prepare(sample_adata, *args, **kwargs)  # type: ignore
            else:
                intercept_df, effect_df = self.summary_prepare(sample_adata, *args, **kwargs)  # type: ignore
        # otherwise, get pre-calculated DataFrames. Effect DataFrame is stitched together from varm
        else:
            intercept_df = sample_adata.varm["intercept_df"]
            covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
            effect_dfs = [sample_adata.varm[f"effect_df_{cov}"] for cov in covariates]
            effect_df = pd.concat(effect_dfs)
            effect_df.index = pd.MultiIndex.from_product(
                (covariates, sample_adata.var.index.tolist()), names=["Covariate", "Cell Type"]
            )
            effect_df.index = effect_df.index.set_levels(
                effect_df.index.levels[0].str.replace("Condition", "").str.replace("[", "").str.replace("]", ""),
                level=0,
            )
            if model_type == "tree_agg":
                node_df = sample_adata.uns["scCODA_params"]["node_df"]

        # Get number of samples, cell types
        data_dims = sample_adata.X.shape

        console = Console()
        table = Table(title="Compositional Analysis summary", box=box.SQUARE, expand=True, highlight=True)
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Value", justify="left")
        table.add_row("Data", "Data: %d samples, %d cell types" % data_dims)
        table.add_row("Reference cell type", "%s" % str(sample_adata.uns["scCODA_params"]["reference_cell_type"]))
        table.add_row("Formula", "%s" % sample_adata.uns["scCODA_params"]["formula"])
        if extended:
            table.add_row("Reference index", "%s" % str(sample_adata.uns["scCODA_params"]["reference_index"]))
            if select_type == "spikeslab":
                table.add_row(
                    "Spike-and-slab threshold",
                    "{threshold:.3f}".format(threshold=sample_adata.uns["scCODA_params"]["threshold_prob"]),
                )
                table.add_row(
                    "Spike-and-slab threshold",
                    "{threshold:.3f}".format(threshold=sample_adata.uns["scCODA_params"]["threshold_prob"]),
                )
            num_results = sample_adata.uns["scCODA_params"]["mcmc"]["num_samples"]
            num_burnin = sample_adata.uns["scCODA_params"]["mcmc"]["num_warmup"]
            table.add_row("MCMC Sampling", f"Sampled {num_results} chain states ({num_burnin} burnin samples)")
            table.add_row(
                "Acceptance rate",
                "{ar:.1f}%".format(
                    ar=(100 * sample_adata.uns["scCODA_params"]["mcmc"]["acceptance_rate"]),
                ),
            )
        console.print(table)

        intercept_df_basic = intercept_df.loc[:, intercept_df.columns.isin(["Final Parameter", "Expected Sample"])]
        if model_type == "tree_agg":
            node_df_basic = node_df.loc[:, node_df.columns.isin(["Final Parameter", "Is credible"])]
            effect_df_basic = effect_df.loc[
                :, effect_df.columns.isin(["Effect", "Expected Sample", "log2-fold change"])
            ]
            effect_df_extended = effect_df.loc[
                :, ~effect_df.columns.isin(["Effect", "Expected Sample", "log2-fold change"])
            ]
        else:
            effect_df_basic = effect_df.loc[
                :, effect_df.columns.isin(["Final Parameter", "Expected Sample", "log2-fold change"])
            ]
            effect_df_extended = effect_df.loc[
                :, ~effect_df.columns.isin(["Final Parameter", "Expected Sample", "log2-fold change"])
            ]
        if extended:
            table = Table("Intercepts", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(intercept_df.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            table = Table("Effects", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(effect_df_basic.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            table = Table("Effects Extended", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(effect_df_extended.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            if model_type == "tree_agg":
                table = Table("Nodes", box=box.SQUARE, expand=True, highlight=True)
                for index in node_df.index.levels[0]:
                    table.add_row(f"Covariate={index}", end_section=True)
                    table.add_row(
                        node_df.loc[index].to_string(justify="center", float_format=lambda _: f"{_:.2f}"),
                        end_section=True,
                    )
                console.print(table)
        else:
            table = Table("Intercepts", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(intercept_df_basic.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            table = Table("Effects", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(effect_df_basic.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            if model_type == "tree_agg":
                table = Table("Nodes", box=box.SQUARE, expand=True, highlight=True)
                for index in node_df_basic.index.levels[0]:
                    table.add_row(f"Covariate={index}", end_section=True)
                    table.add_row(
                        node_df_basic.loc[index].to_string(justify="center", float_format=lambda _: f"{_:.2f}"),
                        end_section=True,
                    )
                console.print(table)

    def get_intercept_df(self, data: AnnData | MuData, modality_key: str = "coda"):
        """Get intercept dataframe as printed in the extended summary

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".

        Returns:
            pd.DataFrame: Intercept data frame.
        """

        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        return sample_adata.varm["intercept_df"]

    def get_effect_df(self, data: AnnData | MuData, modality_key: str = "coda"):
        """Get effect dataframe as printed in the extended summary

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".

        Returns:
            pd.DataFrame: Effect data frame.
        """

        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
        effect_dfs = [sample_adata.varm[f"effect_df_{cov}"] for cov in covariates]
        effect_df = pd.concat(effect_dfs)
        effect_df.index = pd.MultiIndex.from_product(
            (covariates, sample_adata.var.index.tolist()), names=["Covariate", "Cell Type"]
        )
        effect_df.index = effect_df.index.set_levels(
            effect_df.index.levels[0].str.replace("Condition", "").str.replace("[", "").str.replace("]", ""),
            level=0,
        )

        return effect_df

    def get_node_df(self, data: AnnData | MuData, modality_key: str = "coda"):
        """Get node effect dataframe as printed in the extended summary of a tascCODA model

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".

        Returns:
            pd.DataFrame: Node effect data frame.
        """

        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        return sample_adata.uns["scCODA_params"]["node_df"]

    def set_fdr(self, data: AnnData | MuData, est_fdr: float, modality_key: str = "coda", *args, **kwargs):
        """Direct posterior probability approach to calculate credible effects while keeping the expected FDR at a certain level
            Note: Does not work for spike-and-slab LASSO selection method

        Args:
            data: AnnData object or MuData object.
            est_fdr: Desired FDR value.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            args: passed to self.summary_prepare
            kwargs: passed to self.summary_prepare

        Returns:
            Adjusts intercept_df and effect_df
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        intercept_df, effect_df = self.summary_prepare(sample_adata, est_fdr, *args, **kwargs)  # type: ignore
        sample_adata.varm["intercept_df"] = intercept_df
        for cov in effect_df.index.get_level_values("Covariate"):
            sample_adata.varm[f"effect_df_{cov}"] = effect_df.loc[cov, :]

    def credible_effects(self, data: AnnData | MuData, modality_key: str = "coda", est_fdr: float = None) -> pd.Series:
        """Decides which effects of the scCODA model are credible based on an adjustable inclusion probability threshold.
            Note: Parameter est_fdr has no effect for spike-and-slab LASSO selection method

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            est_fdr: Estimated false discovery rate. Must be between 0 and 1. Defaults to None.

        Returns:
            pd.Series: Credible effect decision series which includes boolean values indicate whether effects are credible under inc_prob_threshold.
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        # Get model and effect selection types
        select_type = sample_adata.uns["scCODA_params"]["select_type"]
        model_type = sample_adata.uns["scCODA_params"]["model_type"]

        # If other than None for est_fdr is specified, recalculate intercept and effect DataFrames
        if type(est_fdr) == float:
            if est_fdr < 0 or est_fdr > 1:
                raise ValueError("est_fdr must be between 0 and 1!")
            else:
                _, eff_df = self.summary_prepare(sample_adata, est_fdr=est_fdr)  # type: ignore
        # otherwise, get pre-calculated DataFrames. Effect DataFrame is stitched together from varm
        else:
            if model_type == "tree_agg" and select_type == "sslasso":
                eff_df = sample_adata.uns["scCODA_params"]["node_df"]
            else:
                covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
                effect_dfs = [sample_adata.varm[f"effect_df_{cov}"] for cov in covariates]
                eff_df = pd.concat(effect_dfs)
                eff_df.index = pd.MultiIndex.from_product(
                    (covariates, sample_adata.var.index.tolist()), names=["Covariate", "Cell Type"]
                )

        out = eff_df["Final Parameter"] != 0
        out.rename("credible change")

        return out


def get_a(
    tree: tt.tree,
) -> tuple[np.ndarray, int]:
    """
    Calculate ancestor matrix from a toytree tree

    Parameters
    ----------
    tree
        A toytree tree object

    Returns
    -------
    Ancestor matrix and number of nodes without root node

    A
        Ancestor matrix (numpy array)
    T
        number of nodes in the tree, excluding the root node
    """
    # Builds ancestor matrix

    n_tips = tree.ntips
    n_nodes = tree.nnodes

    A_ = np.zeros((n_tips, n_nodes))

    for i in np.arange(n_nodes):
        leaves_i = list(set(tree.get_node_descendant_idxs(i)) & set(np.arange(n_tips)))
        A_[leaves_i, i] = 1

    # collapsed trees may have scrambled leaves.
    # Therefore, we permute the rows of A such that they are in the original order. Columns (nodes) stay permuted.
    scrambled_leaves = list(tree.get_node_values("idx_orig", True, True)[-n_tips:])
    scrambled_leaves.reverse()
    if scrambled_leaves[0] == "":
        scrambled_leaves = list(np.arange(0, n_tips, 1))

    A = np.zeros((n_tips, n_nodes))
    for r in range(n_tips):
        A[scrambled_leaves[r], :] = A_[r, :]
    A = A[:, :-1]

    return A, n_nodes - 1


def collapse_singularities(tree: tt.tree) -> tt.tree:
    """
    Collapses (deletes) nodes in a toytree tree that are singularities (have only one child).

    Parameters
    ----------
    tree
        A toytree tree object

    Returns
    -------
    A toytree tree without singularities

    tree_new
        A toytree tree
    """

    A, _ = get_a(tree)
    A_T = A.T
    unq, count = np.unique(A_T, axis=0, return_counts=True)

    repeated_idx = []
    for repeated_group in unq[count > 1]:
        repeated_idx.append(np.argwhere(np.all(A_T == repeated_group, axis=1)).ravel())

    nodes_to_delete = [i for idx in repeated_idx for i in idx[1:]]

    # _coords.update() scrambles the idx of leaves. Therefore, keep track of it here
    tree_new = tree.copy()
    for node in tree_new.treenode.traverse():
        node.add_feature("idx_orig", node.idx)

    for n in nodes_to_delete:
        node = tree_new.idx_dict[n]
        node.delete()

    tree_new._coords.update()

    # remove node artifacts
    for k in list(tree_new.idx_dict):
        if k >= tree_new.nnodes:
            tree_new.idx_dict.pop(k)

    return tree_new


def traverse(df_, a, i, innerl):
    """
    Helper function for df2newick
    Adapted from https://stackoverflow.com/questions/15343338/how-to-convert-a-data-frame-to-tree-structure-object-such-as-dendrogram
    """
    if i + 1 < df_.shape[1]:
        a_inner = pd.unique(df_.loc[np.where(df_.iloc[:, i] == a)].iloc[:, i + 1])

        desc = []
        for b in a_inner:
            desc.append(traverse(df_, b, i + 1, innerl))
        if innerl:
            il = a
        else:
            il = ""
        out = f"({','.join(desc)}){il}"
    else:
        out = a

    return out


def df2newick(df: pd.DataFrame, levels: list[str], inner_label: bool = True) -> str:
    """
    Converts a pandas DataFrame with hierarchical information into a newick string.
    Adapted from https://stackoverflow.com/questions/15343338/how-to-convert-a-data-frame-to-tree-structure-object-such-as-dendrogram

    Parameters
    ----------
    df
        Pandas DataFrame that has one row for each leaf of the tree and columns that indicate a hierarchical ordering. See the tascCODA tutorial for an example.
    levels
        list that indicates how the columns in df are ordered as tree levels. Begins with the root level, ends with the leaf level
    inner_label
        Indicator whether labels for inner nodes should be included in the newick string

    Returns
    -------
    Newick string describing the tree structure from df

    newick
        A newick string
    """
    df_tax = df.loc[:, [x for x in levels if x in df.columns]]

    alevel = pd.unique(df_tax.iloc[:, 0])
    strs = []
    for a in alevel:
        strs.append(traverse(df_tax, a, 0, inner_label))

    newick = f"({','.join(strs)});"
    return newick


def get_a_2(
    tree: ete.Tree,
    leaf_order: list[str] = None,
    node_order: list[str] = None,
) -> tuple[np.ndarray, int]:
    """
    Calculate ancestor matrix from a ete3 tree

    Parameters
    ----------
    tree
        A ete3 tree object
    leaf_order
        List of leaf names how they should appear as the rows of the ancestor matrix.
        If None, the ordering will be as in `tree.iter_leaves()`
    node_order
        List of node names how they should appear as the columns of the ancestor matrix
        If None, the ordering will be as in `tree.iter_descendants()`

    Returns
    -------
    Ancestor matrix and number of nodes without root node

    A
        Ancestor matrix (numpy array)
    T
        number of nodes in the tree, excluding the root node
    """
    n_tips = len(tree.get_leaves())
    n_nodes = len(tree.get_descendants())

    node_names = [n.name for n in tree.iter_descendants()]
    duplicates = [x for x in node_names if node_names.count(x) > 1]
    if len(duplicates) > 0:
        raise ValueError(f"Tree nodes have duplicate names: {duplicates}. Make sure that node names are unique!")

    # Initialize ancestor matrix
    A_ = pd.DataFrame(np.zeros((n_tips, n_nodes)))
    A_.index = tree.get_leaf_names()
    A_.columns = [n.name for n in tree.iter_descendants()]

    # Fill in 1's for all connections
    for node in tree.iter_descendants():
        for leaf in tree.get_leaves():
            if leaf in node.get_leaves():
                A_.loc[leaf.name, node.name] = 1

    # Order rows and columns
    if leaf_order is not None:
        A_ = A_.loc[leaf_order]
    if node_order is not None:
        A_ = A_[node_order]
    A_ = np.array(A_)

    return A_, n_nodes


def collapse_singularities_2(tree: ete.Tree) -> ete.Tree:
    """
    Collapses (deletes) nodes in a ete3 tree that are singularities (have only one child).

    Parameters
    ----------
    tree
        A ete3 tree object

    Returns
    -------
    A ete3 tree without singularities

    tree
        A ete3 tree
    """
    for node in tree.iter_descendants():
        if len(node.get_children()) == 1:
            node.delete()

    return tree


def linkage_to_newick(
    Z: np.ndarray,
    labels: list[str],
) -> str:
    """Convert a linkage matrix to newick tree string.
    Adapted from https://stackoverflow.com/a/31878514/20299702.

    Args:
        Z: linkage matrix
        labels: leaf labels

    Returns:
        str: Newick string describing the tree structure
    """
    tree = sp_hierarchy.to_tree(Z, False)

    def build_newick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{(parentdist - node.dist)/2}{newick}"
        else:
            if len(newick) > 0:
                newick = f"):{(parentdist - node.dist)/2}{newick}"
            else:
                newick = ");"
            newick = build_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = build_newick(node.get_right(), f",{newick}", node.dist, leaf_names)
            newick = f"({newick}"
            return newick

    return build_newick(tree, "", tree.dist, labels)


def import_tree(
    data: AnnData | MuData,
    modality_1: str = None,
    modality_2: str = None,
    dendrogram_key: str = None,
    levels_orig: list[str] = None,
    levels_agg: list[str] = None,
    add_level_name: bool = True,
    key_added: str = "tree",
):
    """Generate ete tree for tascCODA models from dendrogram information or cell-level observations.
    Trees can either be generated from scipy dendrogram information e.g. from scanpy.tl.dendrogram,
    or from hierarchical information for each cell type - either saved in `.obs` of the cell-level data object, or in `.var` of the aggregated data.

    Notes:
    - Either `dendrogram_key`, `levels_orig` or `levels_agg` must be not None. Priority is `dendrogram_key` -> `levels_orig` -> `levels_agg`
    - If `data` is a MuData object, `modality_1` and `modality_2` must be specified
    - The node names of the generated tree must be unique. Often, setting `add_level_name=True` is enough to achieve that.

    Args:
        data: A tascCODA-compatible data object.
        modality_1: If `data` is MuData, specifiy the modality name to the original cell level anndata object. Defaults to None.
        modality_2: If `data` is MuData, specifiy the modality name to the aggregated level anndata object. Defaults to None.
        dendrogram_key: Key to the scanpy.tl.dendrogram result in `.uns` of original cell level anndata object. Defaults to None.
        levels_orig: List that indicates which columns in `.obs` of the original data correspond to tree levels. The list must begin with the root level, and end with the leaf level. Defaults to None.
        levels_agg: List that indicates which columns in `.var` of the aggregated data correspond to tree levels. The list must begin with the root level, and end with the leaf level. Defaults to None.
        add_level_name: If True, internal nodes in the tree will be named as "{level_name}_{node_name}" instead of just {level_name}. Defaults to True.
        key_added: If not specified, the tree is stored in .uns[‘tree’]. If `data` is AnnData, save tree in `data`. If `data` is MuData, save tree in data[modality_2]. Defaults to "tree".
        copy: Return a copy instead of writing to `data`. Defaults to False.

    Returns:
        Updates data with the following:

        See `key_added` parameter description for the storage path of tree.

        tree: A ete3 tree object.
    """
    if isinstance(data, MuData):
        try:
            data_1 = data[modality_1]
            data_2 = data[modality_2]
        except KeyError as name:
            print(f"No {name} slot in MuData")
            raise
        except IndexError:
            print("Please specify modality_1 and modality_2 to indicate modalities in MuData")
            raise
    else:
        data_1 = data
        data_2 = data

    if dendrogram_key is not None:
        newick = linkage_to_newick(
            data_1.uns["dendrogram_cell_label"]["linkage"],
            labels=data_1.uns["dendrogram_cell_label"]["categories_ordered"],
        )
        tree = ete.Tree(newick, format=1)
        node_id = 0
        for n in tree.iter_descendants():
            if not n.is_leaf():
                n.name = str(node_id)
                node_id += 1
    elif levels_orig is not None:
        newick = df2newick(data_1.obs.reset_index(), levels=levels_orig)
        tree = ete.Tree(newick, format=8)
        if add_level_name:
            for n in tree.iter_descendants():
                if not n.is_leaf():
                    dist = n.get_distance(n, tree)
                    n.name = f"{levels_orig[int(dist) - 1]}_{n.name}"
    elif levels_agg is not None:
        newick = df2newick(data_2.var.reset_index(), levels=levels_agg)
        tree = ete.Tree(newick, format=8)
        if add_level_name:
            for n in tree.iter_descendants():
                if not n.is_leaf():
                    dist = n.get_distance(n, tree)
                    n.name = f"{levels_agg[int(dist) - 1]}_{n.name}"
    else:
        raise ValueError("Either dendrogram_key, levels_orig or levels_agg must be specified!")

    node_names = [n.name for n in tree.iter_descendants()]
    duplicates = {x for x in node_names if node_names.count(x) > 1}
    if len(duplicates) > 0:
        raise ValueError(f"Tree nodes have duplicate names: {duplicates}. Make sure that node names are unique!")

    data_2.uns[key_added] = tree


def from_scanpy(
    adata: AnnData,
    cell_type_identifier: str,
    sample_identifier: str | list[str],
    covariate_uns: str | None = None,
    covariate_obs: list[str] | None = None,
    covariate_df: pd.DataFrame | None = None,
) -> AnnData:
    """
    Creates a compositional analysis dataset from a single anndata object, as it is produced by e.g. scanpy.

    The anndata object needs to have a column in adata.obs that contains the cell type assignment.
    Further, it must contain one column or a set of columns (e.g. subject id, treatment, disease status) that uniquely identify each (statistical) sample.
    Further covariates (e.g. subject age) can either be specified via addidional column names in adata.obs, a key in adata.uns, or as a separate DataFrame.

    NOTE: The order of samples in the returned dataset is determined by the first occurence of cells from each sample in `adata`

    Args:
        adata: An anndata object from scanpy
        cell_type_identifier: column name in adata.obs that specifies the cell types
        sample_identifier: column name or list of column names in adata.obs that uniquely identify each sample
        covariate_uns: key for adata.uns, where covariate values are stored
        covariate_obs: list of column names in adata.obs, where covariate values are stored. Note: If covariate values are not unique for a value of sample_identifier, this covaariate will be skipped.
        covariate_df: DataFrame with covariates

    Returns:
        AnnData: A data set with cells aggregated to the (sample x cell type) level

    """

    if type(sample_identifier) == str:
        sample_identifier = [sample_identifier]

    if covariate_obs:
        covariate_obs += sample_identifier
    else:
        covariate_obs = sample_identifier  # type: ignore

    # join sample identifiers
    if type(sample_identifier) == list:
        adata.obs["scCODA_sample_id"] = adata.obs[sample_identifier].agg("-".join, axis=1)
        sample_identifier = "scCODA_sample_id"

    # get cell type counts
    groups = adata.obs.value_counts([sample_identifier, cell_type_identifier])
    count_data = groups.unstack(level=cell_type_identifier)
    count_data = count_data.fillna(0)

    # get covariates from different sources
    covariate_df_ = pd.DataFrame(index=count_data.index)

    if covariate_df is None and covariate_obs is None and covariate_uns is None:
        print("No covariate information specified!")

    if covariate_uns is not None:
        covariate_df_uns = pd.DataFrame(adata.uns[covariate_uns])
        covariate_df_ = pd.concat((covariate_df_, covariate_df_uns), axis=1)

    if covariate_obs is not None:
        for c in covariate_obs:
            if any(adata.obs.groupby(sample_identifier).nunique()[c] != 1):
                print(f"Covariate {c} has non-unique values! Skipping...")
                covariate_obs.remove(c)

        covariate_df_obs = adata.obs.groupby(sample_identifier).first()[covariate_obs]
        covariate_df_ = pd.concat((covariate_df_, covariate_df_obs), axis=1)

    if covariate_df is not None:
        if set(covariate_df.index) != set(count_data.index):
            raise ValueError("anndata sample names and covariate_df index do not have the same elements!")
        covs_ord = covariate_df.reindex(count_data.index)
        covariate_df_ = pd.concat((covariate_df_, covs_ord), axis=1)

    covariate_df_.index = covariate_df_.index.astype(str)

    # create var (number of cells for each type as only column)
    var_dat = count_data.sum(axis=0).rename("n_cells").to_frame()
    var_dat.index = var_dat.index.astype(str)

    return AnnData(X=count_data.values, var=var_dat, obs=covariate_df_)
