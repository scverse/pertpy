from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as npd
import toytree as tt
from anndata import AnnData
from jax import config, random
from lamin_utils import logger
from mudata import MuData
from numpyro.infer import Predictive

from pertpy.tools._coda._base_coda import (
    CompositionalModel2,
    collapse_singularities,
    collapse_singularities_2,
    from_scanpy,
    get_a,
    get_a_2,
    import_tree,
)

if TYPE_CHECKING:
    import pandas as pd

config.update("jax_enable_x64", True)


class Tasccoda(CompositionalModel2):
    r"""Statistical model for tree-aggregated differential composition analysis (tascCODA, Ostner et al., 2021).

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
        covariate_uns: str | None = None,
        covariate_obs: list[str] | None = None,
        covariate_df: pd.DataFrame | None = None,
        dendrogram_key: str = None,
        levels_orig: list[str] = None,
        levels_agg: list[str] = None,
        add_level_name: bool = True,
        key_added: str = "tree",
        modality_key_1: str = "rna",
        modality_key_2: str = "coda",
    ) -> MuData:
        """Prepare a MuData object for subsequent processing. If type is "cell_level", then create a compositional analysis dataset from the input adata. If type is "sample_level", generate ete tree for tascCODA models from dendrogram information or cell-level observations.

        When using ``type="cell_level"``, ``adata`` needs to have a column in ``adata.obs`` that contains the cell type assignment.
        Further, it must contain one column or a set of columns (e.g. subject id, treatment, disease status) that uniquely identify each (statistical) sample.
        Further covariates (e.g. subject age) can either be specified via addidional column names in ``adata.obs``, a key in ``adata.uns``, or as a separate DataFrame.

        Args:
            adata: AnnData object.
            type: Specify the input adata type, which could be either a cell-level AnnData or an aggregated sample-level AnnData.
            cell_type_identifier: If type is "cell_level", specify column name in adata.obs that specifies the cell types.
            sample_identifier: If type is "cell_level", specify column name in adata.obs that specifies the sample.
            covariate_uns: If type is "cell_level", specify key for adata.uns, where covariate values are stored.
            covariate_obs: If type is "cell_level", specify list of keys for adata.obs, where covariate values are stored.
            covariate_df: If type is "cell_level", specify dataFrame with covariates.
            dendrogram_key: Key to the scanpy.tl.dendrogram result in `.uns` of original cell level anndata object.
            levels_orig: List that indicates which columns in `.obs` of the original data correspond to tree levels. The list must begin with the root level, and end with the leaf level.
            levels_agg: List that indicates which columns in `.var` of the aggregated data correspond to tree levels. The list must begin with the root level, and end with the leaf level.
            add_level_name: If True, internal nodes in the tree will be named as "{level_name}_{node_name}" instead of just {level_name}.
            key_added: If not specified, the tree is stored in .uns[‘tree’]. If `data` is AnnData, save tree in `data`. If `data` is MuData, save tree in data[modality_2].
            modality_key_1: Key to the cell-level AnnData in the MuData object.
            modality_key_2: Key to the aggregated sample-level AnnData object in the MuData object.

        Returns:
            :class:`mudata.MuData` object with cell-level AnnData (`mudata[modality_key_1]`) and aggregated sample-level AnnData (`mudata[modality_key_2]`).

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.tasccoda_example()
            >>> tasccoda = pt.tl.Tasccoda()
            >>> mdata = tasccoda.load(
            >>>     adata, type="sample_level",
            >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            >>>     key_added="lineage", add_level_name=True
            >>> )
        """
        if type == "cell_level":
            adata_coda = from_scanpy(
                adata=adata,
                cell_type_identifier=cell_type_identifier,
                sample_identifier=sample_identifier,
                covariate_uns=covariate_uns,
                covariate_obs=covariate_obs,
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
        formula: str,
        reference_cell_type: str = "automatic",
        automatic_reference_absence_threshold: float = 0.05,
        tree_key: str = None,
        pen_args: dict = None,
        modality_key: str = "coda",
    ) -> AnnData | MuData:
        """Handles data preprocessing, covariate matrix creation, reference selection, and zero count replacement for tascCODA.

        Args:
            data: Anndata object with cell counts as .X and covariates saved in .obs or a MuData object.
            formula: R-style formula for building the covariate matrix.
                     Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
                     To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
            reference_cell_type: Column name that sets the reference cell type.
                                 If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen.
            automatic_reference_absence_threshold: If using reference_cell_type = "automatic",
                                                   determine the maximum fraction of zero entries for a cell type
                                                   to be considered as a possible reference cell type.
            tree_key: Key in `adata.uns` that contains the tree structure
            pen_args: Dictionary with penalty arguments. With `reg="scaled_3"`, the parameters phi (aggregation bias), lambda_1, lambda_0 can be set here.
                See the tascCODA paper for an explanation of these parameters. Default: lambda_0 = 50, lambda_1 = 5, phi = 0.
            modality_key: If data is a MuData object, specify key to the aggregated sample-level AnnData object in the MuData object.

        Returns:
            Return an AnnData (if input data is an AnnData object) or return a MuData (if input data is a MuData object)

            Specifically, parameters have been set:

            - `adata.uns["param_names"]` or `data[modality_key].uns["param_names"]`: List with the names of all tracked latent model parameters (through `npy.sample` or `npy.deterministic`)
            - `adata.uns["scCODA_params"]["model_type"]` or `data[modality_key].uns["scCODA_params"]["model_type"]`: String indicating the model type ("classic")
            - `adata.uns["scCODA_params"]["select_type"]` or `data[modality_key].uns["scCODA_params"]["select_type"]`: String indicating the type of spike_and_slab selection ("spikeslab")

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.tasccoda_example()
            >>> tasccoda = pt.tl.Tasccoda()
            >>> mdata = tasccoda.load(
            >>>     adata, type="sample_level",
            >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            >>>     key_added="lineage", add_level_name=True
            >>> )
            >>> mdata = tasccoda.prepare(
            >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
            >>> )
        """
        if pen_args is None:
            pen_args = {"lambda_1": 5}
        if isinstance(data, MuData):
            adata = data[modality_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        adata = super().prepare(adata, formula, reference_cell_type, automatic_reference_absence_threshold)

        if tree_key is None:
            raise ValueError("Please specify the key in .uns that contains the tree structure!")

        try:
            import ete4 as ete
        except ImportError:
            raise ImportError(
                "To use tasccoda please install additional dependencies as `pip install pertpy[coda]`"
            ) from None

        # toytree tree - only for legacy reasons, can be removed in the final version
        if isinstance(adata.uns[tree_key], tt.core.ToyTree):
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
                node_leaves = [len(n.leaves()) for n in phy_tree.idx_dict.values()]
                node_leaves.reverse()
                pen_args["node_leaves"] = np.delete(np.array(node_leaves[:-1]), refs)

        # ete tree
        elif isinstance(adata.uns[tree_key], ete.Tree):
            # Collapse singularities in the tree
            phy_tree = collapse_singularities_2(adata.uns[tree_key])

            node_names = [n.name for n in phy_tree.descendants()]

            # Get ancestor matrix
            A, T = get_a_2(phy_tree, leaf_order=adata.var.index.tolist(), node_order=node_names)
            adata.uns["scCODA_params"]["ancestor_matrix"] = A
            adata.uns["scCODA_params"]["T"] = T
            adata.uns["scCODA_params"]["node_names"] = node_names

            # Ancestors of reference are a reference, too!
            # Get names of reference nodes
            reference_cell_type = adata.uns["scCODA_params"]["reference_cell_type"]
            ref_nodes = [n.name for n in list(next(phy_tree.search_nodes(name=reference_cell_type)).ancestors())[:-1]]
            ref_nodes = [reference_cell_type] + ref_nodes
            adata.uns["scCODA_params"]["reference_nodes"] = ref_nodes

            # indices of reference nodes
            ref_idxs = [node_names.index(r) for r in ref_nodes]
            adata.uns["scCODA_params"]["reference_index"] = ref_idxs
            adata.uns["scCODA_params"]["reference_leaf"] = ref_idxs[0]

            # number of leaves for each internal node (important for aggregation penalty lambda_1)
            if "node_leaves" not in pen_args:
                node_leaves = [len(list(n.leaves())) for n in phy_tree.descendants()]
                pen_args["node_leaves"] = np.delete(np.array(node_leaves), ref_idxs)

        # No valid tree structure
        else:
            raise ValueError("Tree structure is not a toytree or ete4 tree object")

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

    def set_init_mcmc_states(self, rng_key: None, ref_index: np.ndarray, sample_adata: AnnData) -> AnnData:  # type: ignore
        """Sets initial MCMC state values for tascCODA model.

        Args:
            rng_key: RNG value to be set
            ref_index: Index of reference feature
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.

        Returns:
            Return AnnData

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.tasccoda_example()
            >>> tasccoda = pt.tl.Tasccoda()
            >>> mdata = tasccoda.load(
            >>>     adata, type="sample_level",
            >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            >>>     key_added="lineage", add_level_name=True
            >>> )
            >>> mdata = tasccoda.prepare(
            >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
            >>> )
            >>> adata = tasccoda.set_init_mcmc_states(rng_key=42, ref_index=[0, 1], sample_adata=mdata["coda"])
        """
        N, D = sample_adata.obsm["covariate_matrix"].shape
        P = sample_adata.X.shape[1]
        T = sample_adata.uns["scCODA_params"]["T"]

        # Reference nodes must be sorted by index
        ref_index = np.sort(ref_index)
        num_ref_nodes = len(ref_index)

        # Sizes of different parameter matrices
        alpha_size = [P]
        beta_nobl_size = [D, T - num_ref_nodes]

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
        )
        lambda_1 = 2 * sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_1"] * penalty_scale_factor
        sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_1_scaled"] = lambda_1

        # Initial MCMC states
        rng = np.random.default_rng(seed=rng_key)

        sample_adata.uns["scCODA_params"]["mcmc"]["init_params"] = {
            "a_0": np.ones(dtype=np.float64, shape=beta_nobl_size) * 1 / lambda_0,
            "b_raw_0": rng.normal(0.0, 1.0, beta_nobl_size),
            "a_1": np.ones(dtype=np.float64, shape=beta_nobl_size) * 1 / lambda_1,
            "b_raw_1": rng.normal(0.0, 1.0, beta_nobl_size),
            "theta": np.ones(dtype=np.float64, shape=1) * 0.5,
            "alpha": rng.normal(0.0, 1.0, alpha_size),
        }

        return sample_adata

    def model(  # type: ignore
        self,
        counts: np.ndarray,
        covariates: np.ndarray,
        n_total: int,
        ref_index: np.ndarray,
        sample_adata: AnnData,
    ):
        """Implements tascCODA model in numpyro.

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
        T = sample_adata.uns["scCODA_params"]["T"]

        # spike-and-slab LASSO parameters
        lambda_0 = sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_0"]
        lambda_1 = sample_adata.uns["scCODA_params"]["sslasso_pen_args"]["lambda_1_scaled"]

        # Reference nodes must be sorted by index
        ref_index = jnp.sort(ref_index)
        num_ref_nodes = len(ref_index)

        # Size of inferred parameter matrix
        d = D * (T - num_ref_nodes)

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
            :class:`arviz.InferenceData`: arviz_data

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.tasccoda_example()
            >>> tasccoda = pt.tl.Tasccoda()
            >>> mdata = tasccoda.load(
            >>>     adata, type="sample_level",
            >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            >>>     key_added="lineage", add_level_name=True
            >>> )
            >>> mdata = tasccoda.prepare(
            >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
            >>> )
            >>> tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100, rng_key=42)
            >>> arviz_data = tasccoda.make_arviz(mdata)
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
        >>> adata = pt.dt.tasccoda_example()
        >>> tasccoda = pt.tl.Tasccoda()
        >>> mdata = tasccoda.load(
        >>>     adata, type="sample_level",
        >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
        >>>     key_added="lineage", add_level_name=True
        >>> )
        >>> mdata = tasccoda.prepare(
        >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        >>> )
        >>> tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100, rng_key=42).
        """  # noqa: D205
        return super().run_nuts(data, modality_key, num_samples, num_warmup, rng_key, copy, *args, **kwargs)

    run_nuts.__doc__ = CompositionalModel2.run_nuts.__doc__ + run_nuts.__doc__

    def summary(self, data: AnnData | MuData, extended: bool = False, modality_key: str = "coda", *args, **kwargs):
        """Examples:
        >>> import pertpy as pt
        >>> adata = pt.dt.tasccoda_example()
        >>> tasccoda = pt.tl.Tasccoda()
        >>> mdata = tasccoda.load(
        >>>     adata, type="sample_level",
        >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
        >>>     key_added="lineage", add_level_name=True
        >>> )
        >>> mdata = tasccoda.prepare(
        >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        >>> )
        >>> tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100, rng_key=42)
        >>> tasccoda.summary(mdata).
        """  # noqa: D205
        return super().summary(data, extended, modality_key, *args, **kwargs)

    summary.__doc__ = CompositionalModel2.summary.__doc__ + summary.__doc__

    def credible_effects(self, data: AnnData | MuData, modality_key: str = "coda", est_fdr: float = None) -> pd.Series:
        """Examples:
        >>> import pertpy as pt
        >>> adata = pt.dt.tasccoda_example()
        >>> tasccoda = pt.tl.Tasccoda()
        >>> mdata = tasccoda.load(
        >>>     adata, type="sample_level",
        >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
        >>>     key_added="lineage", add_level_name=True
        >>> )
        >>> mdata = tasccoda.prepare(
        >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        >>> )
        >>> tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100, rng_key=42)
        >>> tasccoda.credible_effects(mdata).
        """  # noqa: D205
        return super().credible_effects(data, modality_key, est_fdr)

    credible_effects.__doc__ = CompositionalModel2.credible_effects.__doc__ + credible_effects.__doc__

    def set_fdr(self, data: AnnData | MuData, est_fdr: float, modality_key: str = "coda", *args, **kwargs):
        """Examples:
        >>> import pertpy as pt
        >>> adata = pt.dt.tasccoda_example()
        >>> tasccoda = pt.tl.Tasccoda()
        >>> mdata = tasccoda.load(
        >>>     adata, type="sample_level",
        >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
        >>>     key_added="lineage", add_level_name=True
        >>> )
        >>> mdata = tasccoda.prepare(
        >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
        >>> )
        >>> tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100, rng_key=42)
        >>> tasccoda.set_fdr(mdata, est_fdr=0.4).
        """  # noqa: D205
        return super().set_fdr(data, est_fdr, modality_key, *args, **kwargs)

    set_fdr.__doc__ = CompositionalModel2.set_fdr.__doc__ + set_fdr.__doc__
