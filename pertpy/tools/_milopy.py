from __future__ import annotations

import logging
import random
import re
from typing import List

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from rpy2.robjects import conversion, numpy2ri, pandas2ri
from rpy2.robjects.packages import STAP, PackageNotInstalledError, importr
from scipy.sparse import csr_matrix


class Milopy:
    """Python implementation of Milo"""

    def __init__(self):
        pass

    def make_nhoods(
        self,
        adata: AnnData,
        neighbors_key: str | None = None,
        prop: float = 0.1,
        seed: int = 42,
        copy: bool = False,
    ):
        """Randomly sample vertcies on a graph to define neighbourhoods.

        The implementation resembels https://github.com/emdann/milopy
        These are then refined by computing the median profile for the neighbourhood in reduced dimensional space and selecting the nearest vertex to this position.
        Thus, multiple neighbourhoods may be collapsed down together to prevent over-sampling the graph space.

        Args:
            adata (AnnData): Should contain a knn graph in `adata.obsp`.
            neighbors_key (str | None, optional): The key in `adata.obsp` to use as KNN graph. If not specified, `make_nhoods` looks .obsp[‘connectivities’] for connectivities (default storage places for `scanpy.pp.neighbors`). If specified, it looks .obsp[.uns[neighbors_key][‘connectivities_key’]] for connectivities. Defaults to None.
            prop (float, optional): Fraction of cells to sample for neighbourhood index search. Defaults to 0.1.
            seed (int, optional): Random seed for cell sampling. Defaults to 42.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            If `copy=True`, returns the copy of `adata` with the result in `.obs`, `.obsm`, and `.uns`.
            Otherwise:

            nhoods: scipy.sparse._csr.csr_matrix in `adata.obsm['nhoods']`.
            A binary matrix of cell to neighbourhood assignments

            nhood_ixs_refined: pandas.Series in `adata.obs['nhood_ixs_refined']`.
            A boolean indicating whether a cell is an index for a neighbourhood

            nhood_kth_distance: pandas.Series in `adata.obs['nhood_kth_distance']`.
            The distance to the kth nearest neighbour for each index cell (used for SpatialFDR correction)

            nhood_neighbors_key: `adata.uns["nhood_neighbors_key"]`
            KNN graph key, used for neighbourhood construction
        """
        if copy:
            adata = adata.copy()

        # Get reduced dim used for KNN graph
        if neighbors_key is None:
            try:
                use_rep = adata.uns["neighbors"]["params"]["use_rep"]
            except KeyError:
                logging.warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            try:
                knn_graph = adata.obsp["connectivities"].copy()
            except KeyError:
                raise KeyError(
                    'No "connectivities" slot in adata.obsp -- please run scanpy.pp.neighbors(adata) first'
                ) from None
        else:
            try:
                use_rep = adata.uns["neighbors"]["params"]["use_rep"]
            except KeyError:
                logging.warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            knn_graph = adata.obsp[neighbors_key + "_connectivities"].copy()

        X_dimred = adata.obsm[use_rep]
        n_ixs = int(np.round(adata.n_obs * prop))
        knn_graph[knn_graph != 0] = 1
        random.seed(seed)
        random_vertices = random.sample(range(adata.n_obs), k=n_ixs)
        random_vertices.sort()
        ixs_nn = knn_graph[random_vertices, :]
        non_zero_rows = ixs_nn.nonzero()[0]
        non_zero_cols = ixs_nn.nonzero()[1]
        refined_vertices = np.empty(
            shape=[
                len(random_vertices),
            ]
        )

        from sklearn.metrics.pairwise import euclidean_distances

        for i in range(len(random_vertices)):
            nh_pos = np.median(X_dimred[non_zero_cols[non_zero_rows == i], :], 0).reshape(-1, 1)
            nn_ixs = non_zero_cols[non_zero_rows == i]
            # Find closest real point (amongst nearest neighbors)
            dists = euclidean_distances(X_dimred[non_zero_cols[non_zero_rows == i], :], nh_pos.T)
            # Update vertex index
            refined_vertices[i] = nn_ixs[dists.argmin()]

        refined_vertices = np.unique(refined_vertices.astype("int"))
        refined_vertices.sort()

        nhoods = knn_graph[:, refined_vertices]
        adata.obsm["nhoods"] = nhoods

        # Add ixs to adata
        adata.obs["nhood_ixs_random"] = adata.obs_names.isin(adata.obs_names[random_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs_names.isin(adata.obs_names[refined_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs["nhood_ixs_refined"].astype("int")
        adata.obs["nhood_ixs_random"] = adata.obs["nhood_ixs_random"].astype("int")
        # Store info on neighbor_key used
        adata.uns["nhood_neighbors_key"] = neighbors_key
        # Store distance to K-th nearest neighbor (used for spatial FDR correction)
        if neighbors_key is None:
            knn_dists = adata.obsp["distances"]
        else:
            knn_dists = adata.obsp[neighbors_key + "_distances"]

        nhood_ixs = adata.obs["nhood_ixs_refined"] == 1
        dist_mat = knn_dists[nhood_ixs, :]
        k_distances = dist_mat.max(1).toarray().ravel()
        adata.obs["nhood_kth_distance"] = 0
        adata.obs.loc[adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"] = k_distances

        if copy:
            return adata

    def count_nhoods(
        self,
        adata: AnnData,
        sample_col: str,
    ):
        """Builds a sample-level AnnData object storing the matrix of cell counts per sample per neighbourhood.

        Args:
            adata (AnnData): AnnData object with neighbourhoods defined in `adata.obsm['nhoods']`
            sample_col (str): Column in adata.obs that contains sample information

        Returns:
            MuData object storing the original (i.e. cell-level) AnnData in `mudata['cells']`
            and the sample-level anndata storing the neighbourhood cell counts in `mudata['samples']`.
            Here:
            - `mudata['samples'].obs_names` are samples (defined from `adata.obs['sample_col']`)
            - `mudata['samples'].var_names` are neighbourhoods
            - `mudata['samples'].X` is the matrix counting the number of cells from each
            sample in each neighbourhood
        """
        try:
            nhoods = adata.obsm["nhoods"]
        except KeyError:
            print('Cannot find "nhoods" slot in adata.obsm -- please run milopy.make_nhoods(adata)')
        #  Make nhood abundance matrix
        sample_dummies = pd.get_dummies(adata.obs[sample_col])
        all_samples = sample_dummies.columns
        sample_dummies = csr_matrix(sample_dummies.values)
        nhood_count_mat = nhoods.T.dot(sample_dummies)
        sample_obs = pd.DataFrame(index=all_samples)
        sample_adata = AnnData(X=nhood_count_mat.T, obs=sample_obs)
        sample_adata.uns["sample_col"] = sample_col
        # Save nhood index info
        sample_adata.var["index_cell"] = adata.obs_names[adata.obs["nhood_ixs_refined"] == 1]
        sample_adata.var["kth_distance"] = adata.obs.loc[
            adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"
        ].values
        # adata.uns["sample_adata"] = sample_adata
        # Make MuData object
        milo_mdata = MuData({"cells": adata, "samples": sample_adata})
        return milo_mdata

    def da_nhoods(
        self,
        milo_mdata: MuData,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
    ):
        """Performs differential abundance testing on neighbourhoods using QLF test implementation from edgeR (using R code under the hood)

        Args:
            milo_mdata (MuData): MuData object
            design (str): formula for the test, following glm syntax from R (e.g. '~ condition'). Terms should be columns in `milo_mdata['cells'].obs`.
            model_contrasts (str | None, optional): A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl"). If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group. Defaults to None.
            subset_samples (List[str] | None, optional): subset of samples (obs in `milo_mdata['samples']`) to use for the test. Defaults to None.
            add_intercept (bool, optional): whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default. Defaults to True.

        Returns:
            None, modifies `milo_mdata['samples']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure
        """
        try:
            sample_adata = milo_mdata["samples"]
        except KeyError:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: 'cells' and 'samples' - please run milopy.count_nhoods(adata) first"
            ) from None
        adata = milo_mdata["cells"]

        # Set up rpy2 to run edgeR
        numpy2ri.activate()
        pandas2ri.activate()
        edgeR = self._try_import_bioc_library("edgeR")
        limma = self._try_import_bioc_library("limma")
        stats = importr("stats")
        base = importr("base")
        covariates = [x.strip(" ") for x in set(re.split("\\+|\\*", design.lstrip("~ ")))]

        # Add covariates used for testing to sample_adata.var
        sample_col = sample_adata.uns["sample_col"]
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            print("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))

        sample_obs = sample_obs[covariates + [sample_col]]
        sample_obs.index = sample_obs[sample_col].astype("str")

        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except ValueError:
            print(
                "Covariates cannot be unambiguously assigned to each sample -- each sample value should match a single covariate value"
            )
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]
        # Get design dataframe
        try:
            design_df = sample_adata.obs[covariates]
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            print('Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov)))

        # Get count matrix
        count_mat = sample_adata.X.T.toarray()
        lib_size = count_mat.sum(0)

        # Filter out samples with zero counts
        keep_smp = lib_size > 0

        # Subset samples
        if subset_samples is not None:
            keep_smp = keep_smp & sample_adata.obs_names.isin(subset_samples)
            design_df = design_df[keep_smp]
            for i, e in enumerate(design_df.columns):
                if design_df.dtypes[i].name == "category":
                    design_df[e] = design_df[e].cat.remove_unused_categories()

        # Filter out nhoods with zero counts (they can appear after sample filtering)
        keep_nhoods = count_mat[:, keep_smp].sum(1) > 0

        # Define model matrix
        if not add_intercept or model_contrasts is not None:
            design = design + " + 0"
        model = stats.model_matrix(object=stats.formula(design), data=design_df)

        # Fit NB-GLM
        dge = edgeR.DGEList(counts=count_mat[keep_nhoods, :][:, keep_smp], lib_size=lib_size[keep_smp])
        dge = edgeR.calcNormFactors(dge, method="TMM")
        dge = edgeR.estimateDisp(dge, model)
        fit = edgeR.glmQLFit(dge, model, robust=True)

        # Test
        n_coef = model.shape[1]
        if model_contrasts is not None:
            r_str = """
            get_model_cols <- function(design_df, design){
                m = model.matrix(object=formula(design), data=design_df)
                return(colnames(m))
            }
            """
            get_model_cols = STAP(r_str, "get_model_cols")
            model_mat_cols = get_model_cols.get_model_cols(design_df, design)
            model_df = pd.DataFrame(model)
            model_df.columns = model_mat_cols
            try:
                mod_contrast = limma.makeContrasts(contrasts=model_contrasts, levels=model_df)
            except ValueError:
                print("Model contrasts must be in the form 'A-B' or 'A+B'")
            res = base.as_data_frame(
                edgeR.topTags(edgeR.glmQLFTest(fit, contrast=mod_contrast), sort_by="none", n=np.inf)
            )
        else:
            res = base.as_data_frame(edgeR.topTags(edgeR.glmQLFTest(fit, coef=n_coef), sort_by="none", n=np.inf))
        res = conversion.rpy2py(res)
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)

        # Save outputs
        res.index = sample_adata.var_names[keep_nhoods]
        if any([x in sample_adata.var.columns for x in res.columns]):
            sample_adata.var = sample_adata.var.drop(res.columns, axis=1)
        sample_adata.var = pd.concat([sample_adata.var, res], axis=1)

        # Run Graph spatial FDR correction
        self._graph_spatialFDR(sample_adata, neighbors_key=adata.uns["nhood_neighbors_key"])

    def annotate_nhoods(self, milo_mdata: MuData, anno_col: str):
        """Assigns a categorical label to neighbourhoods, based on the most frequent label among cells in each neighbourhood. This can be useful to stratify DA testing results by cell types or samples.

        Args:
            milo_mdata (MuData): MuData object
            anno_col (str): Column in adata.obs containing the cell annotations to use for nhood labelling

        Returns:
            None. Adds in place:
            - `milo_mdata['samples'].var["nhood_annotation"]`: assigning a label to each nhood
            - `milo_mdata['samples'].var["nhood_annotation_frac"]` stores the fraciton of cells in the neighbourhood with the assigned label
            - `milo_mdata['samples'].varm['frac_annotation']`: stores the fraction of cells from each label in each nhood
            - `milo_mdata['samples'].uns["annotation_labels"]`: stores the column names for `milo_mdata['samples'].varm['frac_annotation']`
        """
        try:
            sample_adata = milo_mdata["samples"]
        except KeyError:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: 'cells' and 'samples' - please run milopy.count_nhoods(adata) first"
            ) from None
        adata = milo_mdata["cells"]

        # Check value is not numeric
        if pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of categorical type - please use milopy.utils.annotate_nhoods_continuous for continuous variables"
            )

        anno_dummies = pd.get_dummies(adata.obs[anno_col])
        anno_count = adata.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
        anno_frac = np.array(anno_count / anno_count.sum(1))

        anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)
        sample_adata.varm["frac_annotation"] = anno_frac_dataframe.values
        sample_adata.uns["annotation_labels"] = anno_frac_dataframe.columns
        sample_adata.uns["annotation_obs"] = anno_col
        sample_adata.var["nhood_annotation"] = anno_frac_dataframe.idxmax(1)
        sample_adata.var["nhood_annotation_frac"] = anno_frac_dataframe.max(1)

    def annotate_nhoods_continuous(self, milo_mdata: MuData, anno_col: str):
        """Assigns a continuous value to neighbourhoods, based on mean cell level covariate stored in adata.obs. This can be useful to correlate DA log-foldChanges with continuous covariates such as pseudotime, gene expression scores etc...

        Args:
            milo_mdata (MuData): MuData object
            anno_col (str): Column in adata.obs containing the cell annotations to use for nhood labelling

        Returns:
            None. Adds in place:
            - `milo_mdata['samples'].var["nhood_{anno_col}"]`: assigning a continuous value to each nhood
        """
        if "samples" not in milo_mdata:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: 'cells' and 'samples' - please run milopy.count_nhoods(adata) first"
            )
        adata = milo_mdata["cells"]

        # Check value is not categorical
        if not pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of continuous type - please use milopy.utils.annotate_nhoods for categorical variables"
            )

        anno_val = adata.obsm["nhoods"].T.dot(csr_matrix(adata.obs[anno_col]).T)

        mean_anno_val = anno_val.toarray() / np.array(adata.obsm["nhoods"].T.sum(1))

        milo_mdata["samples"].var[f"nhood_{anno_col}"] = mean_anno_val

    def add_covariate_to_nhoods_var(self, milo_mdata: MuData, new_covariates: list[str]):
        """Add covariate from cell-level obs to sample-level obs. These should be covariates for which a single value can be assigned to each sample.

        Args:
            milo_mdata (MuData): MuData object
            new_covariates (List[str]): columns in `milo_mdata['cells'].obs` to add to `milo_mdata['samples'].obs`.

        Returns:
            None, adds columns to `milo_mdata['samples']` in place
        """
        try:
            sample_adata = milo_mdata["samples"]
        except KeyError:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: 'cells' and 'samples' - please run milopy.count_nhoods(adata) first"
            ) from None
        adata = milo_mdata["cells"]

        sample_col = sample_adata.uns["sample_col"]
        covariates = list(
            set(sample_adata.obs.columns[sample_adata.obs.columns != sample_col].tolist() + new_covariates)
        )
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            print("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
        sample_obs = sample_obs[covariates + [sample_col]].astype("str")
        sample_obs.index = sample_obs[sample_col]
        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except ValueError:
            print(
                "Covariates cannot be unambiguously assigned to each sample -- each sample value should match a single covariate value"
            )
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

    def build_nhood_graph(self, milo_mdata: MuData, basis: str = "X_umap"):
        """Build graph of neighbourhoods used for visualization of DA results

        Args:
            milo_mdata (MuData): MuData object
            basis (str, optional): Name of the obsm basis to use for layout of neighbourhoods (key in `adata.obsm`). Defaults to "X_umap".

        Returns:
            - `milo_mdata['samples'].varp['nhood_connectivities']`: graph of overlap between neighbourhoods (i.e. no of shared cells)
            - `milo_mdata['samples'].var["Nhood_size"]`: number of cells in neighbourhoods
        """
        adata = milo_mdata["cells"]
        # # Add embedding positions
        milo_mdata["samples"].varm["X_milo_graph"] = adata[adata.obs["nhood_ixs_refined"] == 1].obsm[basis]
        # Add nhood size
        milo_mdata["samples"].var["Nhood_size"] = np.array(adata.obsm["nhoods"].sum(0)).flatten()
        # Add adjacency graph
        milo_mdata["samples"].varp["nhood_connectivities"] = adata.obsm["nhoods"].T.dot(adata.obsm["nhoods"])
        milo_mdata["samples"].varp["nhood_connectivities"].setdiag(0)
        milo_mdata["samples"].varp["nhood_connectivities"].eliminate_zeros()
        milo_mdata["samples"].uns["nhood"] = {"connectivities_key": "nhood_connectivities", "distances_key": ""}

    def add_nhood_expression(
        self,
        milo_mdata: MuData,
        layer: str | None = None,
    ):
        """Calculates the mean expression in neighbourhoods of each feature.

        Args:
            milo_mdata (MuData): MuData object
            layer (str | None, optional): If provided, use `milo_mdata['cells'][layer]` as expression matrix instead of `milo_mdata['cells'].X`. Defaults to None.

        Returns:
            Updates adata in place to store the matrix of average expression in each neighbourhood in `milo_mdata['samples'].varm['expr']`
        """
        try:
            sample_adata = milo_mdata["samples"]
        except KeyError:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: 'cells' and 'samples' - please run milopy.count_nhoods(adata) first"
            ) from None
        adata = milo_mdata["cells"]

        # Get gene expression matrix
        if layer is None:
            X = adata.X
            expr_id = "expr"
        else:
            X = adata.layers[layer]
            expr_id = "expr_" + layer

        # Aggregate over nhoods -- taking the mean
        nhoods_X = X.T.dot(adata.obsm["nhoods"])
        nhoods_X = csr_matrix(nhoods_X / adata.obsm["nhoods"].toarray().sum(0))
        sample_adata.varm[expr_id] = nhoods_X.T

    def _try_import_bioc_library(
        self,
        name: str,
    ):
        """Import R packages

        Args:
            name (str): R packages name
        """
        try:
            _r_lib = importr(name)
            return _r_lib
        except PackageNotInstalledError:
            raise RuntimeError(
                f"Install Bioconductor library `{name!r}` first as `BiocManager::install({name!r}).`"
            ) from None

    def _graph_spatial_fdr(
        self,
        sample_adata: AnnData,
        neighbors_key: str | None = None,
    ):
        """FDR correction weighted on inverse of connectivity of neighbourhoods. The distance to the k-th nearest neighbor is used as a measure of connectivity.

        Args:
            sample_adata (AnnData): Sample-level AnnData.
            neighbors_key (str, optional): The key in `adata.obsp` to use as KNN graph. Defaults to None.
        """
        # use 1/connectivity as the weighting for the weighted BH adjustment from Cydar
        w = 1 / sample_adata.var["kth_distance"]
        w[np.isinf(w)] = 0

        # Computing a density-weighted q-value.
        pvalues = sample_adata.var["PValue"]
        keep_nhoods = ~pvalues.isna()  #  Filtering in case of test on subset of nhoods
        o = pvalues[keep_nhoods].argsort()
        pvalues = pvalues[keep_nhoods][o]
        w = w[keep_nhoods][o]

        adjp = np.zeros(shape=len(o))
        adjp[o] = (sum(w) * pvalues / np.cumsum(w))[::-1].cummin()[::-1]
        adjp = np.array([x if x < 1 else 1 for x in adjp])

        # Store in anndata
        sample_adata.var["SpatialFDR"] = np.nan
        sample_adata.var.loc[keep_nhoods, "SpatialFDR"] = adjp
