from __future__ import annotations

from typing import Literal

import arviz as az
import ete3 as ete
import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as npd
import pandas as pd
import patsy as pt
import toytree as tt
from anndata import AnnData
from jax import random
from jax._src.prng import PRNGKeyArray
from jax._src.typing import Array
from jax.config import config
from mudata import MuData
from numpyro.infer import HMC, MCMC, NUTS, Predictive
from scipy.cluster import hierarchy as sp_hierarchy

config.update("jax_enable_x64", True)


class CompositionalModel2:
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

    def __init__(self):
        pass

    def make_arviz(self, *args, **kwargs):
        pass

    def model(self, *args, **kwargs):
        pass

    def prepare_anndata(
        self,
        sample_adata: AnnData,
        formula: str,
        reference_cell_type: str = "automatic",
        automatic_reference_absence_threshold: float = 0.05,
    ) -> AnnData:
        """Handles data preprocessing, covariate matrix creation, reference selection, and zero count replacement.

        Args:
            sample_adata (AnnData): anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            formula (str): R-style formula for building the covariate matrix.
                Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
                To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
            reference_cell_type (str, optional): Column name that sets the reference cell type.
                Reference the name of a column. If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen. Defaults to "automatic".
            automatic_reference_absence_threshold (float, optional): If using reference_cell_type = "automatic", determine the maximum fraction of zero entries for a cell type
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
            print(f"Automatic reference selection! Reference cell type set to {ref_cell_type}")

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
            raise ValueError(
                "Wrong input dimensions X[{},:] != y[{},:]".format(
                    sample_adata.obsm["covariate_matrix"].shape[0], sample_adata.X.shape[0]
                )
            )
        if covariate_matrix.shape[0] != len(sample_adata.obsm["sample_counts"]):
            raise ValueError(
                "Wrong input dimensions X[{},:] != n_total[{}]".format(
                    sample_adata.obsm["covariate_matrix"], len(sample_adata.obsm["sample_counts"])
                )
            )

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
            sample_adata (AnnData): anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            kernel (npy.infer.mcmc.MCMCKernel): A `numpyro.infer.mcmc.MCMCKernel` object
            rng_key (Array or random.PRNGKey): The rng state used. If None, a random state will be selected
            copy (bool, optional): Return a copy instead of writing to adata. Defaults to False.
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
        sample_adata.uns["scCODA_params"]["mcmc"]["acceptance_rate"] = self.mcmc.last_state.mean_accept_prob.to_py()
        sample_adata.uns["scCODA_params"]["mcmc"]["samples"] = self.mcmc.get_samples()

        # Evaluate results and create result dataframes (based on tre-aggregation or not)
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

    def go_nuts(
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
        """No-U-turn sampling

        Args:
            data (AnnData | MuData): AnnData object or MuData object.
            modality_key (str, optional): If data is a MuData object, specify which modality to use. Defaults to "coda".
            num_samples (int, optional): Number of sampled values after burn-in. Defaults to 10000.
            num_warmup (int, optional): Number of burn-in (warmup) samples. Defaults to 1000.
            rng_key (int, optional): The rng state used. If None, a random state will be selected. Defaults to None.
            copy (bool, optional): Return a copy instead of writing to adata. Defaults to False.

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
            sample_adata.uns["scCODA_params"]["mcmc"]["rng_key"] = rng_key_array
        else:
            rng_key_array = random.PRNGKey(rng_key)

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
        """Hamiltonian Monte Carlo sampling

        Args:
            data (AnnData | MuData): AnnData object or MuData object.
            modality_key (str, optional): If data is a MuData object, specify which modality to use. Defaults to "coda".
            num_samples (int, optional): Number of sampled values after burn-in. Defaults to 20000.
            num_warmup (int, optional): Number of burn-in (warmup) samples. Defaults to 5000.
            rng_key (int, optional): The rng state used. If None, a random state will be selected. Defaults to None.
            copy (bool, optional): Return a copy instead of writing to adata. Defaults to False.
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
            sample_adata (AnnData): Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            est_fdr (float, optional): Desired FDR value. Defaults to 0.05.
            args: Passed to ``az.summary``
            kwargs: Passed to ``az.summary``

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Intercept, effect and node-level DataFrames
            pd.DataFrame: intercept_df
                Summary of intercept parameters. Contains one row per cell type.

                Columns:
                - Final Parameter: Final intercept model parameter
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Expected sample: Expected cell counts for a sample with no present covariates. See the tutorial for more explanation

            pd.DataFrame: effect_df
                Summary of effect (slope) parameters. Contains one row per covariate/cell type combination.

                Columns:
                - Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Expected sample: Expected cell counts for a sample with only the current covariate set to 1. See the tutorial for more explanation
                - log2-fold change: Log2-fold change between expected cell counts with no covariates and with only the current covariate
                - Inclusion probability: Share of MCMC samples, for which this effect was not set to 0 by the spike-and-slab prior.

            pd.DataFrame: node_df
                Summary of effect (slope) parameters on the tree nodes (features or groups of features). Contains one row per covariate/cell type combination.

                Columns:
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
            sample_adata (AnnData): Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            intercept_df (pd.DataFrame): Intercept summary, see ``summary_prepare``
            effect_df (pd.DataFrame): Effect summary, see ``summary_prepare``
            model_type (str): String indicating the model type ("classic" or "tree_agg")
            select_type (str):  String indicating the type of spike_and_slab selection ("spikeslab" or "sslasso")
            target_fdr (float, optional): Desired FDR value. Defaults to 0.05.
            node_df (pd.DataFrame, optional): If using tree aggregation, the node-level effect DataFrame must be passed. Defaults to None.

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

        beta_mean = alpha_par
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
            sample_adata (AnnData): Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            node_df (pd.DataFrame): Node-level effect summary, see ``summary_prepare``

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
            sample_adata (AnnData): Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            intercept_df (pd.DataFrame): Intercept summary, see ``summary_prepare``

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
            data (AnnData | MuData): AnnData object or MuData object.
            extended (bool, optional): If True, return the extended summary with additional statistics. Defaults to False.
            modality_key (str, optional): If data is a MuData object, specify which modality to use. Defaults to "coda".
            args: Passed to az.summary
            kwargs: Passed to az.summary

        Returns:
            Prints to console
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

        # If other than default values for e.g. confidence interval are specified,
        # recalculate them for intercept and effect DataFrames
        if args or kwargs:
            if model_type == "tree_agg":
                intercept_df, effect_df, node_df = self.summary_prepare(sample_adata, *args, **kwargs)  # type: ignore
            else:
                intercept_df, effect_df, _ = self.summary_prepare(sample_adata, *args, **kwargs)  # type: ignore
        # otherwise, get pre-calculated DataFrames. Effect DataFrame is stitched together from varm
        else:
            intercept_df = sample_adata.varm["intercept_df"]
            covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
            effect_dfs = [sample_adata.varm[f"effect_df_{cov}"] for cov in covariates]
            effect_df = pd.concat(effect_dfs)
            effect_df.index = pd.MultiIndex.from_product(
                (covariates, sample_adata.var.index.tolist()), names=["Covariate", "Cell Type"]
            )
            if model_type == "tree_agg":
                node_df = sample_adata.uns["scCODA_params"]["node_df"]

        # Get number of samples, cell types
        data_dims = sample_adata.X.shape

        if not extended:
            # Cut down DataFrames to relevant info
            intercept_df = intercept_df.loc[:, ["Final Parameter", "Expected Sample"]]

            if model_type == "tree_agg":
                node_df = node_df.loc[:, ["Final Parameter", "Is credible"]]
                effect_df = effect_df.loc[:, ["Effect", "Expected Sample", "log2-fold change"]]
            else:
                effect_df = effect_df.loc[:, ["Final Parameter", "Expected Sample", "log2-fold change"]]

        # Print everything neatly
        print("Compositional Analysis summary:")
        print("")
        print("Data: %d samples, %d cell types" % data_dims)
        print("Reference cell type: %s" % str(sample_adata.uns["scCODA_params"]["reference_cell_type"]))
        if extended:
            print("Reference index: %s" % str(sample_adata.uns["scCODA_params"]["reference_index"]))
        print("Formula: %s" % sample_adata.uns["scCODA_params"]["formula"])
        if extended:
            if select_type == "spikeslab":
                print(
                    "Spike-and-slab threshold: {threshold:.3f}".format(
                        threshold=sample_adata.uns["scCODA_params"]["threshold_prob"]
                    )
                )
            print("")
            print(
                "MCMC Sampling: Sampled {num_results} chain states ({num_burnin} burnin samples). "
                "Acceptance rate: {ar:.1f}%".format(
                    num_results=sample_adata.uns["scCODA_params"]["mcmc"]["num_samples"],
                    num_burnin=sample_adata.uns["scCODA_params"]["mcmc"]["num_warmup"],
                    ar=(100 * sample_adata.uns["scCODA_params"]["mcmc"]["acceptance_rate"]),
                )
            )
        print("")
        print("Intercepts:")
        print(intercept_df)
        print("")
        print("")
        print("Effects:")
        print(effect_df)
        print("")
        print("")
        if model_type == "tree_agg":
            print("Nodes:")
            print(node_df)

    def set_fdr(self, data: AnnData | MuData, est_fdr: float, modality_key: str = "coda", *args, **kwargs):
        """Direct posterior probability approach to calculate credible effects while keeping the expected FDR at a certain level
            Note: Does not work for spike-and-slab LASSO selection method

        Args:
            data (AnnData | MuData): AnnData object or MuData object.
            est_fdr (float): Desired FDR value.
            modality_key (str, optional): If data is a MuData object, specify which modality to use. Defaults to "coda".
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
            data (AnnData | MuData): AnnData object or MuData object.
            modality_key (str, optional): If data is a MuData object, specify which modality to use. Defaults to "coda".
            est_fdr (float, optional): Estimated false discovery rate. Must be between 0 and 1. Defaults to None.

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


class SccodaModel2(CompositionalModel2):

    """
    Statistical model for single-cell differential composition analysis with specification of a reference cell type.
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
        generate_sample_level: bool = False,
        cell_type_identifier: str = None,
        sample_identifier: str = None,
        covariate_key: str | None = None,
        covariate_df: pd.DataFrame | None = None,
        modality_key_1: str = "rna",
        modality_key_2: str = "coda",
    ) -> MuData:
        """Prepare a MuData object for subsequent processing. If type is "cell_level", then create a compositional analysis dataset from the input adata.

        Args:
            adata (AnnData): AnnData object.
            type (Literal[&quot;cell_level&quot;, &quot;sample_level&quot;]): Specify the input adata type, which could be either a cell-level AnnData or an aggregated sample-level AnnData.
            cell_type_identifier (str, optional): If type is "cell_level", specify column name in adata.obs that specifies the cell types. Defaults to None.
            sample_identifier (str, optional): If type is "cell_level", specify column name in adata.obs that specifies the sample. Defaults to None.
            covariate_key (Optional[str], optional): If type is "cell_level", specify key for adata.uns, where covariate values are stored. Defaults to None.
            covariate_df (Optional[pd.DataFrame], optional): If type is "cell_level", specify dataFrame with covariates. Defaults to None.
            modality_key_1 (str, optional): Key to the cell-level AnnData in the MuData object. Defaults to "rna".
            modality_key_2 (str, optional): Key to the aggregated sample-level AnnData object in the MuData object. Defaults to "coda".

        Returns:
            MuData: MuData object with cell-level AnnData (`mudata[modality_key_1]`) and aggregated sample-level AnnData (`mudata[modality_key_2]`).
        """

        if type == "cell_level":
            if generate_sample_level:
                adata_coda = from_scanpy(
                    adata=adata,
                    cell_type_identifier=cell_type_identifier,
                    sample_identifier=sample_identifier,
                    covariate_key=covariate_key,
                    covariate_df=covariate_df,
                )
            else:
                adata_coda = AnnData()
            mdata = MuData({modality_key_1: adata, modality_key_2: adata_coda})
        else:
            mdata = MuData({modality_key_1: AnnData(), modality_key_2: adata})
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
            data (AnnData or MuData): Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            formula (str): R-style formula for building the covariate matrix.
                Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
                To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
            reference_cell_type (str, optional): Column name that sets the reference cell type.
                Reference the name of a column. If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen. Defaults to "automatic".
            automatic_reference_absence_threshold (float, optional): If using reference_cell_type = "automatic", determine the maximum fraction of zero entries for a cell type
                to be considered as a possible reference cell type. Defaults to 0.05.
            modality_key (str, optional): If data is a MuData object, specify key to the aggregated sample-level AnnData object in the MuData object. Defaults to "coda".

        Returns:
            Return an AnnData (if input data is an AnnData object) or return a MuData (if input data is a MuData object)

            Specifically, parameters have been set:
            - `adata.uns["param_names"]` or `data[modality_key].uns["param_names"]`: List with the names of all tracked latent model parameters (through `npy.sample` or `npy.deterministic`)
            - `adata.uns["scCODA_params"]["model_type"]` or `data[modality_key].uns["scCODA_params"]["model_type"]`: String indicating the model type ("classic")
            - `adata.uns["scCODA_params"]["select_type"]` or `data[modality_key].uns["scCODA_params"]["select_type"]`: String indicating the type of spike_and_slab selection ("spikeslab")

        """
        if isinstance(data, MuData):
            adata = data[modality_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        adata = super().prepare_anndata(adata, formula, reference_cell_type, automatic_reference_absence_threshold)
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

    def model(
        self,
        counts: np.ndarray,
        covariates: np.ndarray,
        n_total: np.ndarray,
        ref_index,
        sample_adata: AnnData,
    ):
        """
        Implements scCODA model in numpyro

        Args:
            counts (np.ndarray): Count data array
            covariates (np.ndarray): Covariate matrix
            n_total (int): Number of counts per sample
            ref_index (np.ndarray): Index of reference feature
            sample_adata (AnnData): Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.

        Returns:
            predictions (see numpyro documentation for details on models)
        """
        # data dimensions
        N, D = sample_adata.obsm["covariate_matrix"].shape
        P = sample_adata.X.shape[1]

        # Sizes of different parameter matrices
        alpha_size = [P]
        sigma_size = [D, 1]
        beta_nobl_size = [D, P - 1]

        # Initial MCMC states
        sample_adata.uns["scCODA_params"]["mcmc"]["init_params"] = {
            "sigma_d": np.ones(dtype=np.float64, shape=sigma_size),
            "b_offset": np.random.normal(0.0, 1.0, beta_nobl_size),
            "ind_raw": np.zeros(dtype=np.float64, shape=beta_nobl_size),
            "alpha": np.random.normal(0.0, 1.0, alpha_size),
        }

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

    def make_arviz(
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        rng_key=None,
        num_prior_samples: int = 500,
        use_posterior_predictive: bool = True,
    ) -> az.InferenceData:
        """Creates arviz object from model results for MCMC diagnosis

        Args:
            data (AnnData | MuData): AnnData object or MuData object.
            modality_key (str, optional): If data is a MuData object, specify which modality to use. Defaults to "coda".
            rng_key (int, optional): The rng state used for the prior simulation. If None, a random state will be selected. Defaults to None.
            num_prior_samples (int, optional): Number of prior samples calculated. Defaults to 500.
            use_posterior_predictive (bool, optional): If True, the posterior predictive will be calculated. Defaults to True.

        Returns:
            az.InferenceData: arviz_data with all MCMC information
        """

        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
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
            rng_key = random.PRNGKey(np.random.randint(0, 10000))

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


class TasccodaModel2(CompositionalModel2):

    """
    Statistical model for tree-aggregated differential composition analysis (tascCODA, Ostner et al., 2021).

    The hierarchical formulation of the model for one sample is:

    .. math::
         \\begin{align*}
            Y_i &\\sim \\textrm{DirMult}(\\bar{Y}_i, \\textbf{a}(\\textbf{x})_i)\\\\
            \\log(\\textbf{a}(X))_i &= \\alpha + X_{i, \\cdot} \\beta\\\\
            \\alpha_j &\\sim \\mathcal{N}(0, 10) & \\forall j\\in[p]\\\\
            \\beta &= \\hat{\\beta} A^T \\\\
            \\hat{\\beta}_{l, k} &= 0 & \\forall k \\in \\hat{v}, l \\in [d]\\\\
            \\hat{\\beta}_{l, k} &= \\theta \\tilde{\\beta}_{1, l, k} + (1- \\theta) \\tilde{\\beta}_{0, l, k} \\quad & \\forall k\\in\\{[v] \\smallsetminus \\hat{v}\\}, l \\in [d]\\\\
            \\tilde{\\beta}_{m, l, k} &= \\sigma_{m, l, k} * b_{m, l, k} \\quad & \\forall k\\in\\{[v] \\smallsetminus \\hat{v}\\}, m \\in \\{0, 1\\}, l \\in [d]\\\\
            \\sigma_{m, l, k} &\\sim \\textrm{Exp}(\\lambda_{m, l, k}^2/2) \\quad & \\forall k\\in\\{[v] \\smallsetminus \\hat{v}\\}, l \\in \\{0, 1\\}, l \\in [d]\\\\
            b_{m, l, k} &\\sim N(0,1) \\quad & \\forall k\\in\\{[v] \\smallsetminus \\hat{v}\\}, l \\in \\{0, 1\\}, l \\in [d]\\\\
            \\theta &\\sim \\textrm{Beta}(1, \\frac{1}{|\\{[v] \\smallsetminus \\hat{v}\\}|})
        \\end{align*}

    with Y being the cell counts, X the covariates, and v the set of nodes of the underlying tree structure.

    For further information, see `tascCODA: Bayesian Tree-Aggregated Analysis of Compositional Amplicon and Single-Cell Data`
    (Ostner et al., 2021)

    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

    def load(
        self,
        adata: AnnData,
        type: Literal["cell_level", "sample_level"],
        cell_type_identifier: str = None,
        sample_identifier: str = None,
        covariate_key: str | None = None,
        covariate_df: pd.DataFrame | None = None,
        dendrogram_key: str = None,
        levels_orig: list[str] = None,
        levels_agg: list[str] = None,
        add_level_name: bool = False,
        key_added: str = "tree",
        modality_key_1: str = "rna",
        modality_key_2: str = "coda",
    ) -> MuData:
        """Prepare a MuData object for subsequent processing. If type is "cell_level", then create a compositional analysis dataset from the input adata. If type is "sample_level", generate ete tree for tascCODA models from dendrogram information or cell-level observations.

        Args:
            adata (AnnData): AnnData object.
            type (Literal[&quot;cell_level&quot;, &quot;sample_level&quot;]): Specify the input adata type, which could be either a cell-level AnnData or an aggregated sample-level AnnData.
            cell_type_identifier (str, optional): If type is "cell_level", specify column name in adata.obs that specifies the cell types. Defaults to None.
            sample_identifier (str, optional): If type is "cell_level", specify column name in adata.obs that specifies the sample. Defaults to None.
            covariate_key (Optional[str], optional): If type is "cell_level", specify key for adata.uns, where covariate values are stored. Defaults to None.
            covariate_df (Optional[pd.DataFrame], optional): If type is "cell_level", specify dataFrame with covariates. Defaults to None.
            dendrogram_key (str, optional): Key to the scanpy.tl.dendrogram result in `.uns` of original cell level anndata object. Defaults to None.
            levels_orig (List[str], optional): List that indicates which columns in `.obs` of the original data correspond tree levels. The list must begin with the root level, and end with the leaf level. Defaults to None.
            levels_agg (List[str], optional): List that indicates which columns in `.var` of the aggregated data correspond tree levels. The list must begin with the root level, and end with the leaf level. Defaults to None.
            add_level_name (bool, optional): If True, internal nodes in the tree will be named as "{level_name}_{node_name}" instead of just {level_name}. Defaults to False.
            key_added (str, optional): If not specified, the tree is stored in .uns[‘tree’]. If `data` is AnnData, save tree in `data`. If `data` is MuData, save tree in data[modality_2]. Defaults to "tree".
            modality_key_1 (str, optional): Key to the cell-level AnnData in the MuData object. Defaults to "rna".
            modality_key_2 (str, optional): Key to the aggregated sample-level AnnData object in the MuData object. Defaults to "coda".

        Returns:
            MuData: MuData object with cell-level AnnData (`mudata[modality_key_1]`) and aggregated sample-level AnnData (`mudata[modality_key_2]`).
        """

        if type == "cell_level":
            adata_coda = from_scanpy(
                adata=adata,
                cell_type_identifier=cell_type_identifier,
                sample_identifier=sample_identifier,
                covariate_key=covariate_key,
                covariate_df=covariate_df,
            )
            mdata = MuData({modality_key_1: adata, modality_key_2: adata_coda})
        else:
            mdata = MuData({modality_key_1: AnnData(), modality_key_2: adata})
        import_tree(
            data=mdata,
            modality_1=modality_key_1,
            modality_2=modality_key_2,
            dendrogram_key=dendrogram_key,
            levels_orig=levels_orig,
            levels_agg=levels_agg,
            add_level_name=add_level_name,
            key_added=key_added,
        )
        return mdata

    def prepare(
        self,
        data: AnnData | MuData,
        tree_key: str,
        formula: str,
        pen_args: dict = None,
        reference_cell_type: str = "automatic",
        automatic_reference_absence_threshold: float = 0.05,
        modality_key: str = "coda",
    ) -> MuData:
        """Handles data preprocessing, covariate matrix creation, reference selection, and zero count replacement for tascCODA. Also sets model parameters, model type (tree_agg), effect selection type (sslaso) and performs tree processing.

        Args:
            data (AnnData | MuData): Anndata object with cell counts as .X and covariates saved in .obs or a MuData object.
            tree_key (str): Key in `sample_adata.uns` that contains the tree structure as a `toytree.tree` object
            formula (str): R-style formula for building the covariate matrix.
                Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
                To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
            pen_args (dict, optional): Dictionary with penalty arguments. With `reg="scaled_3"`, the parameters phi (aggregation bias), lambda_1, lambda_0 can be set here.
                See the tascCODA paper for an explanation of these parameters. Default: lambda_0 = 50, lambda_1 = 5, phi = 0.
            reference_cell_type (str, optional): Column name that sets the reference cell type.
                Reference the name of a column. If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen. Defaults to "automatic".
            automatic_reference_absence_threshold (float, optional): If using reference_cell_type = "automatic", determine the maximum fraction of zero entries for a cell type
                to be considered as a possible reference cell type. Defaults to 0.05.
            modality_key (str, optional): If data is a MuData object, specify key to the aggregated sample-level AnnData object in the MuData object. Defaults to "coda".

        Returns:
            Return an AnnData (if input data is an AnnData object) or return a MuData (if input data is a MuData object)

                Specifically, parameters have been set:
                - `adata.uns["param_names"]` or `data[modality_key].uns["param_names"]`: List with the names of all tracked latent model parameters (through `npy.sample` or `npy.deterministic`)
                - `adata.uns["scCODA_params"]["model_type"]` or `data[modality_key].uns["scCODA_params"]["model_type"]`: String indicating the model type ("classic")
                - `adata.uns["scCODA_params"]["select_type"]` or `data[modality_key].uns["scCODA_params"]["select_type"]`: String indicating the type of spike_and_slab selection ("spikeslab")
        """
        if pen_args is None:
            pen_args = {"lambda_1": 5}
        if isinstance(data, MuData):
            adata = data[modality_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        adata = super().prepare_anndata(adata, formula, reference_cell_type, automatic_reference_absence_threshold)

        # toytree tree - only for legacy reasons, can be removed in the final version
        if type(adata.uns[tree_key]) == tt.tree:
            # Collapse singularities in the tree
            phy_tree = collapse_singularities(adata.uns[tree_key])

            # Get ancestor matrix
            A, T = get_a(phy_tree)
            adata.uns["scCODA_params"]["ancestor_matrix"] = A
            adata.uns["scCODA_params"]["T"] = T

            # Bring names of nodes back in order, they might have been scrambled during collapse_singularities
            node_names = [n.name for n in phy_tree.idx_dict.values()][1:]
            node_names.reverse()
            adata.uns["scCODA_params"]["node_names"] = node_names

            order = [n.name for n in adata.uns[tree_key].treenode.traverse() if n.is_leaf()]
            order.reverse()
            order_ind = [adata.var.index.tolist().index(x) for x in order]

            adata = adata[:, order_ind]

            ref_node_index = adata.uns["scCODA_params"]["reference_index"]
            # Ancestors of reference are a reference, too!
            refs = [p.idx for p in phy_tree.idx_dict[ref_node_index].get_ancestors()][:-1]
            refs = [ref_node_index] + refs
            adata.uns["scCODA_params"]["reference_index"] = refs
            adata.uns["scCODA_params"]["reference_leaf"] = ref_node_index

            # number of leaves for each internal node (important for aggregation penalty lambda_1)
            if "node_leaves" not in pen_args:
                node_leaves = [len(n.get_leaves()) for n in phy_tree.idx_dict.values()]
                node_leaves.reverse()
                pen_args["node_leaves"] = np.delete(np.array(node_leaves[:-1]), refs)

        # ete tree
        elif type(adata.uns[tree_key]) == ete.Tree:
            # Collapse singularities in the tree
            phy_tree = collapse_singularities_2(adata.uns[tree_key])

            node_names = [n.name for n in phy_tree.iter_descendants()]

            # Get ancestor matrix
            A, T = get_a_2(phy_tree, leaf_order=adata.var.index.tolist(), node_order=node_names)
            adata.uns["scCODA_params"]["ancestor_matrix"] = A
            adata.uns["scCODA_params"]["T"] = T
            adata.uns["scCODA_params"]["node_names"] = node_names

            # Ancestors of reference are a reference, too!
            # Get names of reference nodes
            reference_cell_type = adata.uns["scCODA_params"]["reference_cell_type"]
            ref_nodes = [n.name for n in phy_tree.search_nodes(name=reference_cell_type)[0].get_ancestors()[:-1]]
            ref_nodes = [reference_cell_type] + ref_nodes
            adata.uns["scCODA_params"]["reference_nodes"] = ref_nodes

            # indices of reference nodes
            ref_idxs = [node_names.index(r) for r in ref_nodes]
            adata.uns["scCODA_params"]["reference_index"] = ref_idxs
            adata.uns["scCODA_params"]["reference_leaf"] = ref_idxs[0]

            # number of leaves for each internal node (important for aggregation penalty lambda_1)
            if "node_leaves" not in pen_args:
                node_leaves = [len(n.get_leaves()) for n in phy_tree.iter_descendants()]
                pen_args["node_leaves"] = np.delete(np.array(node_leaves), ref_idxs)

        # No valid tree structure
        else:
            raise ValueError("Tree structure is not a toytree or ete3 tree object")

        # Default spike-and-slab LASSO parameters
        if "lambda_0" not in pen_args:
            pen_args["lambda_0"] = 50
        if "lambda_1" not in pen_args:
            pen_args["lambda_1"] = 5
        if "phi" not in pen_args:
            pen_args["phi"] = 0

        adata.uns["scCODA_params"]["sslasso_pen_args"] = pen_args

        # All parameters that are returned for analysis
        adata.uns["scCODA_params"]["param_names"] = [
            "alpha_0",
            "b_0",
            "alpha_1",
            "b_1",
            "theta",
            "alpha",
            "bet_0",
            "bet_1",
            "beta_select",
            "beta",
            "concentration",
            "prediction",
        ]

        adata.uns["scCODA_params"]["model_type"] = "tree_agg"
        adata.uns["scCODA_params"]["select_type"] = "sslasso"
        if is_MuData:
            data.mod[modality_key] = adata
            return data
        else:
            return adata

    def model(
        self,
        counts: np.ndarray,
        covariates: np.ndarray,
        n_total: int,
        ref_index: np.ndarray,
        sample_adata: AnnData,
    ):
        """Implements tascCODA model in numpyro

        Args:
            counts (np.ndarray): Count data array
            covariates (np.ndarray): Covariate matrix
            n_total (int): Number of counts per sample
            ref_index (np.ndarray): Index of reference feature
            sample_adata (AnnData): Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.

        Returns:
            predictions (see numpyro documentation for details on models)
        """
        # data dimensions
        dtype = "float64"

        N, D = sample_adata.obsm["covariate_matrix"].shape
        P = sample_adata.X.shape[1]
        T = sample_adata.uns["scCODA_params"]["T"]

        # spike-and-slab LASSO parameters
        lambda_0 = sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_0"]
        # tree-based scaled penalty
        penalty_scale_factor = np.array(
            (
                1
                / (
                    1
                    + np.exp(
                        -1
                        * sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["phi"]
                        * (sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["node_leaves"] / P - 0.5)
                    )
                )
            ),
            dtype,
        )
        lambda_1 = 2 * sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_1"] * penalty_scale_factor
        sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_1_scaled"] = lambda_1

        # Reference nodes must be sorted by index
        ref_index = jnp.sort(ref_index)
        num_ref_nodes = len(ref_index)

        # Sizes of different parameter matrices
        alpha_size = [P]
        beta_nobl_size = [D, T - num_ref_nodes]

        # Size of inferred parameter matrix
        d = D * (T - num_ref_nodes)

        # Initial MCMC states
        sample_adata.uns["scCODA_params"]["mcmc"]["init_params"] = {
            "a_0": np.ones(dtype=np.float64, shape=beta_nobl_size) * 1 / lambda_0,
            "b_raw_0": np.random.normal(0.0, 1.0, beta_nobl_size),
            "a_1": np.ones(dtype=np.float64, shape=beta_nobl_size) * 1 / lambda_1,
            "b_raw_1": np.random.normal(0.0, 1.0, beta_nobl_size),
            "theta": np.ones(dtype=np.float64, shape=1) * 0.5,
            "alpha": np.random.normal(0.0, 1.0, alpha_size),
        }

        # numpyro plates for all dimensions
        covariate_axis = npy.plate("covs", D, dim=-2)
        node_axis = npy.plate("ct", T, dim=-1)
        node_axis_nobl = npy.plate("ctnb", T - num_ref_nodes, dim=-1)
        cell_type_axis = npy.plate("ct", P, dim=-1)
        sample_axis = npy.plate("sample", N, dim=-2)

        # Spike-and-slab LASSO effects
        theta = npy.sample("theta", npd.Beta(concentration1=1.0, concentration0=d))

        with covariate_axis, node_axis_nobl:
            a_0 = npy.sample("a_0", npd.Exponential((lambda_0**2) / 2))
            b_raw_0 = npy.sample("b_raw_0", npd.Normal(0.0, 1.0))
            b_tilde_0 = npy.deterministic("b_tilde_0", a_0 * b_raw_0)

            a_1 = npy.sample("a_1", npd.Exponential((lambda_1**2) / 2))
            b_raw_1 = npy.sample("b_raw_1", npd.Normal(0.0, 1.0))
            b_tilde_1 = npy.deterministic("b_tilde_1", a_1 * b_raw_1)

            # calculate proposed beta and perform spike-and-slab
            b_tilde = (1 - theta) * b_tilde_0 + theta + b_tilde_1

        with node_axis, covariate_axis:
            # Include effect 0 for reference nodes
            ref_inserts = jnp.array([ref_index[i] - i for i in range(num_ref_nodes)])
            b_tilde = jnp.insert(b_tilde, ref_inserts, jnp.zeros(shape=[D, 1]), axis=-1)
            b_tilde = npy.deterministic("b_tilde", b_tilde)

        with cell_type_axis:
            # Intercepts
            alpha = npy.sample("alpha", npd.Normal(0.0, 10.0))

            with covariate_axis:
                # sum up tree levels
                beta = npy.deterministic(
                    "beta", jnp.matmul(b_tilde, sample_adata.uns["scCODA_params"]["ancestor_matrix"].T)
                )

        # Combine intercepts and effects
        with sample_axis:
            concentrations = npy.deterministic(
                "concentrations", jnp.nan_to_num(jnp.exp(alpha + jnp.matmul(covariates, beta)), 0.0001)
            )

        # Calculate DM-distributed counts
        predictions = npy.sample("counts", npd.DirichletMultinomial(concentrations, n_total), obs=counts)

        return predictions

    def make_arviz(
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        rng_key=None,
        num_prior_samples: int = 500,
        use_posterior_predictive: bool = True,
    ) -> az.InferenceData:
        """Creates arviz object from model results for MCMC diagnosis

        Args:
            data (AnnData | MuData): AnnData object or MuData object.
            modality_key (str, optional): If data is a MuData object, specify which modality to use. Defaults to "coda".
            rng_key (optional): The rng state used for the prior simulation. If None, a random state will be selected. Defaults to None.
            num_prior_samples (int, optional): Number of prior samples calculated. Defaults to 500.
            use_posterior_predictive (bool, optional): If True, the posterior predictive will be calculated. Defaults to True.

        Returns:
            arviz.InferenceData: arviz_data
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                print("When data is a MuData object, modality_key must be specified!")
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
            "theta": [],
            "a_0": ["covariate", "node_nb"],
            "b_raw_0": ["covariate", "node_nb"],
            "a_1": ["covariate", "node_nb"],
            "b_raw_1": ["covariate", "node_nb"],
            "b_tilde_0": ["covariate", "node_nb"],
            "b_tilde_1": ["covariate", "node_nb"],
            "b_tilde": ["covariate", "node"],
            "beta": ["covariate", "cell_type"],
            "concentrations": ["sample", "cell_type"],
            "counts": ["sample", "cell_type"],
        }

        # arviz coordinates
        reference_index = sample_adata.uns["scCODA_params"]["reference_index"]
        node_names = sample_adata.uns["scCODA_params"]["node_names"]
        nodes_nb = [val for n, val in enumerate(node_names) if n not in reference_index]
        coords = {
            "cell_type": cell_types,
            "node_nb": nodes_nb,
            "node": node_names,
            "covariate": sample_adata.uns["scCODA_params"]["covariate_names"],
            "sample": sample_adata.obs.index,
        }

        dtype = "float64"

        # Prior and posterior predictive simulation
        numpyro_covariates = jnp.array(sample_adata.obsm["covariate_matrix"], dtype=dtype)
        numpyro_n_total = jnp.array(sample_adata.obsm["sample_counts"], dtype=dtype)
        ref_index = jnp.array(sample_adata.uns["scCODA_params"]["reference_index"])

        if rng_key is None:
            rng_key = random.PRNGKey(np.random.randint(0, 10000))

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
        Z (np.ndarray): linkage matrix
        labels (List[str]): leaf labels

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
        data (AnnData | MuData): A tascCODA-compatible data object.
        modality_1 (str, optional): If `data` is MuData, specifiy the modality name to the original cell level anndata object. Defaults to None.
        modality_2 (str, optional): If `data` is MuData, specifiy the modality name to the aggregated level anndata object. Defaults to None.
        dendrogram_key (str, optional): Key to the scanpy.tl.dendrogram result in `.uns` of original cell level anndata object. Defaults to None.
        levels_orig (List[str], optional): List that indicates which columns in `.obs` of the original data correspond tree levels. The list must begin with the root level, and end with the leaf level. Defaults to None.
        levels_agg (List[str], optional): List that indicates which columns in `.var` of the aggregated data correspond tree levels. The list must begin with the root level, and end with the leaf level. Defaults to None.
        add_level_name (bool, optional): If True, internal nodes in the tree will be named as "{level_name}_{node_name}" instead of just {level_name}. Defaults to True.
        key_added (str, optional): If not specified, the tree is stored in .uns[‘tree’]. If `data` is AnnData, save tree in `data`. If `data` is MuData, save tree in data[modality_2]. Defaults to "tree".
        copy (bool, optional): Return a copy instead of writing to `data`. Defaults to False.

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
    sample_identifier: str,
    covariate_key: str | None = None,
    covariate_df: pd.DataFrame | None = None,
) -> AnnData:

    """
    Creates a compositional analysis dataset from a single anndata object, as it is produced by e.g. scanpy.

    The anndata object needs to have a column in adata.obs that contains the cell type assignment,
    and one column that specifies the grouping into samples.
    Covariates can either be specified via a key in adata.uns, or as a separate DataFrame.

    NOTE: The order of samples in the returned dataset is determined by the first occurence of cells from each sample in `adata`

    Args:
        adata (AnnData): An anndata object from scanpy
        cell_type_identifier (str): column name in adata.obs that specifies the cell types
        sample_identifier (str): column name in adata.obs that specifies the sample
        covariate_key (str, optional): key for adata.uns, where covariate values are stored
        covariate_df (pd.DataFrame, optional): DataFrame with covariates

    Returns:
        AnnData: A data set with cells aggregated to the (sample x cell type) level

    """

    groups = adata.obs.value_counts([sample_identifier, cell_type_identifier])
    count_data = groups.unstack(level=cell_type_identifier)
    count_data = count_data.fillna(0)

    if covariate_key is not None:
        covariate_df = pd.DataFrame(adata.uns[covariate_key])
    elif covariate_df is None:
        print("No covariate information specified!")
        covariate_df = pd.DataFrame(index=count_data.index)

    if set(covariate_df.index) != set(count_data.index):
        raise ValueError("anndata sample names and covariate_df index do not have the same elements!")
    covs_ord = covariate_df.reindex(count_data.index)
    covs_ord.index = covs_ord.index.astype(str)

    var_dat = count_data.sum(axis=0).rename("n_cells").to_frame()
    var_dat.index = var_dat.index.astype(str)

    return AnnData(X=count_data.values, var=var_dat, obs=covs_ord)
