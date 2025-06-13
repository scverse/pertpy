from __future__ import annotations

import random
import re
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from lamin_utils import logger
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from mudata import MuData

from pertpy._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

from scipy.sparse import coo_matrix, csr_matrix, issparse, spmatrix
from sklearn.metrics.pairwise import euclidean_distances


def _is_counts(array: np.ndarray | spmatrix) -> bool:
    """Check if the array is a count matrix."""
    if issparse(array):
        return bool(np.all(np.mod(array.data, 1) == 0))
    else:
        return bool(np.all(np.mod(array, 1) == 0))


class Milo:
    """Python implementation of Milo."""

    def load(
        self,
        input: AnnData,
        feature_key: str | None = "rna",
    ) -> MuData:
        """Prepare a MuData object for subsequent processing.

        Args:
            input: AnnData
            feature_key: Key to store the cell-level AnnData object in the MuData object

        Returns:
            :class:`mudata.MuData` object with original AnnData.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)

        """
        mdata = MuData({feature_key: input, "milo": AnnData()})

        return mdata

    def make_nhoods(
        self,
        data: AnnData | MuData,
        neighbors_key: str | None = None,
        feature_key: str | None = "rna",
        prop: float = 0.1,
        seed: int = 0,
        copy: bool = False,
    ):
        """Randomly sample vertices on a KNN graph to define neighbourhoods of cells.

        The set of neighborhoods get refined by computing the median profile for the neighbourhood in reduced dimensional space
        and by selecting the nearest vertex to this position.
        Thus, multiple neighbourhoods may be collapsed to prevent over-sampling the graph space.

        Args:
            data: AnnData object with KNN graph defined in `obsp` or MuData object with a modality with KNN graph defined in `obsp`
            neighbors_key: The key in `adata.obsp` or `mdata[feature_key].obsp` to use as KNN graph.
                           If not specified, `make_nhoods` looks .obsp[‘connectivities’] for connectivities (default storage places for `scanpy.pp.neighbors`).
                           If specified, it looks at .obsp[.uns[neighbors_key][‘connectivities_key’]] for connectivities.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.
            prop: Fraction of cells to sample for neighbourhood index search.
            seed: Random seed for cell sampling.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            If `copy=True`, returns the copy of `adata` with the result in `.obs`, `.obsm`, and `.uns`.
            Otherwise:

            nhoods: scipy.sparse._csr.csr_matrix in `adata.obsm['nhoods']`.
            A binary matrix of cell to neighbourhood assignments. Neighbourhoods in the columns are ordered by the order of the index cell in adata.obs_names

            nhood_ixs_refined: pandas.Series in `adata.obs['nhood_ixs_refined']`.
            A boolean indicating whether a cell is an index for a neighbourhood

            nhood_kth_distance: pandas.Series in `adata.obs['nhood_kth_distance']`.
            The distance to the kth nearest neighbour for each index cell (used for SpatialFDR correction)

            nhood_neighbors_key: `adata.uns["nhood_neighbors_key"]`
            KNN graph key, used for neighbourhood construction

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])

        """
        if isinstance(data, MuData):
            adata = data[feature_key]
        if isinstance(data, AnnData):
            adata = data
        if copy:
            adata = adata.copy()

        # Get reduced dim used for KNN graph
        if neighbors_key is None:
            try:
                use_rep = adata.uns["neighbors"]["params"]["use_rep"]
            except KeyError:
                logger.warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            try:
                knn_graph = adata.obsp["connectivities"].copy()
            except KeyError:
                logger.error('No "connectivities" slot in adata.obsp -- please run scanpy.pp.neighbors(adata) first')
                raise
        else:
            try:
                use_rep = adata.uns[neighbors_key]["params"]["use_rep"]
            except KeyError:
                logger.warning("Using X_pca as default embedding")
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
        adata.uns["nhood_neighbors_key"] = neighbors_key
        # Store distance to K-th nearest neighbor (used for spatial FDR correction)
        knn_dists = adata.obsp["distances"] if neighbors_key is None else adata.obsp[neighbors_key + "_distances"]

        nhood_ixs = adata.obs["nhood_ixs_refined"] == 1
        dist_mat = knn_dists[np.asarray(nhood_ixs), :]
        k_distances = dist_mat.max(1).toarray().ravel()
        adata.obs["nhood_kth_distance"] = 0
        adata.obs["nhood_kth_distance"] = adata.obs["nhood_kth_distance"].astype(float)
        adata.obs.loc[adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"] = k_distances

        if copy:
            return adata

    def count_nhoods(
        self,
        data: AnnData | MuData,
        sample_col: str,
        feature_key: str | None = "rna",
    ):
        """Builds a sample-level AnnData object storing the matrix of cell counts per sample per neighbourhood.

        Args:
            data: AnnData object with neighbourhoods defined in `obsm['nhoods']` or MuData object with a modality with neighbourhoods defined in `obsm['nhoods']`
            sample_col: Column in adata.obs that contains sample information
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            MuData object storing the original (i.e. rna) AnnData in `mudata[feature_key]`
            and the compositional anndata storing the neighbourhood cell counts in `mudata['milo']`.
            Here:
            - `mudata['milo'].obs_names` are samples (defined from `adata.obs['sample_col']`)
            - `mudata['milo'].var_names` are neighbourhoods
            - `mudata['milo'].X` is the matrix counting the number of cells from each
            sample in each neighbourhood

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")

        """
        if isinstance(data, MuData):
            adata = data[feature_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                logger.error('Cannot find "nhoods" slot in adata.obsm -- please run milopy.make_nhoods(adata)')
                raise
        # Make nhood abundance matrix
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

        if is_MuData is True:
            data.mod["milo"] = sample_adata
            return data
        else:
            milo_mdata = MuData({feature_key: adata, "milo": sample_adata})
            return milo_mdata

    def da_nhoods(
        self,
        mdata: MuData,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
        feature_key: str | None = "rna",
        solver: Literal["edger", "pydeseq2"] = "edger",
    ):
        """Performs differential abundance testing on neighbourhoods using QLF test implementation as implemented in edgeR.

        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `milo_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group.
            subset_samples: subset of samples (obs in `milo_mdata['milo']`) to use for the test.
            add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.
            solver: The solver to fit the model to.
                The "edger" solver requires R, rpy2 and edgeR to be installed and is the closest to the R implementation.
                The "pydeseq2" requires pydeseq2 to be installed. It is still very comparable to the "edger" solver but might be a bit slower.

        Returns:
            None, modifies `milo_mdata['milo']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots:"
                " feature_key and 'milo' - please run milopy.count_nhoods() first"
            )
            raise
        adata = mdata[feature_key]

        covariates = [x.strip(" ") for x in set(re.split("\\+|\\*", design.lstrip("~ ")))]

        # Add covariates used for testing to sample_adata.var
        sample_col = sample_adata.uns["sample_col"]
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            logger.warning("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]]
        sample_obs.index = sample_obs[sample_col].astype("str")

        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except AssertionError:
            logger.warning(
                f"Values in mdata[{feature_key}].obs[{covariates}] cannot be unambiguously assigned to each sample"
                f" -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

        # Get design dataframe
        try:
            design_df = sample_adata.obs[covariates]
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            logger.error(
                'Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov))
            )
            raise
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

        if solver == "edger":
            # Set up rpy2 to run edgeR
            edgeR, limma, stats, base = self._setup_rpy2()

            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.vectors import FloatVector

            # Define model matrix
            if not add_intercept or model_contrasts is not None:
                design = design + " + 0"
            design_df = design_df.astype(dict.fromkeys(design_df.select_dtypes(exclude=["number"]).columns, "category"))
            with localconverter(ro.default_converter + pandas2ri.converter):
                design_r = pandas2ri.py2rpy(design_df)
            formula_r = stats.formula(design)
            model = stats.model_matrix(object=formula_r, data=design_r)

            # Fit NB-GLM
            counts_filtered = count_mat[np.ix_(keep_nhoods, keep_smp)]
            lib_size_filtered = lib_size[keep_smp]
            with localconverter(ro.default_converter + numpy2ri.converter):
                count_mat_r = numpy2ri.py2rpy(counts_filtered)
            lib_size_r = FloatVector(lib_size_filtered)
            dge = edgeR.DGEList(counts=count_mat_r, lib_size=lib_size_r)
            dge = edgeR.calcNormFactors(dge, method="TMM")
            dge = edgeR.estimateDisp(dge, model)
            fit = edgeR.glmQLFit(dge, model, robust=True)
            # Test
            model_np = np.array(model)
            n_coef = model_np.shape[1]
            if model_contrasts is not None:
                r_str = """
                get_model_cols <- function(design_df, design){
                    m = model.matrix(object=formula(design), data=design_df)
                    return(colnames(m))
                }
                """
                from rpy2.robjects.packages import STAP

                get_model_cols = STAP(r_str, "get_model_cols")
                with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
                    model_mat_cols = get_model_cols.get_model_cols(design_df, design)
                with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                    model_df = pandas2ri.rpy2py(model)
                model_df = pd.DataFrame(model_df)
                model_df.columns = model_mat_cols
                try:
                    with localconverter(ro.default_converter + pandas2ri.converter):
                        mod_contrast = limma.makeContrasts(contrasts=model_contrasts, levels=model_df)
                except ValueError:
                    logger.error("Model contrasts must be in the form 'A-B' or 'A+B'")
                    raise
                with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
                    res = base.as_data_frame(
                        edgeR.topTags(edgeR.glmQLFTest(fit, contrast=mod_contrast), sort_by="none", n=np.inf)
                    )
            else:
                with localconverter(ro.default_converter + numpy2ri.converter + pandas2ri.converter):
                    res = base.as_data_frame(
                        edgeR.topTags(edgeR.glmQLFTest(fit, coef=n_coef), sort_by="none", n=np.inf)
                    )
            if not isinstance(res, pd.DataFrame):
                res = pd.DataFrame(res)
            # The columns of res looks like e.g. table.A, table.B, so remove the prefix
            res.columns = [col.replace("table.", "") for col in res.columns]
        elif solver == "pydeseq2":
            if find_spec("pydeseq2") is None:
                raise ImportError("pydeseq2 is required but not installed. Install with: pip install pydeseq2")

            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats

            counts_filtered = count_mat[np.ix_(keep_nhoods, keep_smp)]
            design_df_filtered = design_df.iloc[keep_smp].copy()

            design_df_filtered = design_df_filtered.astype(
                dict.fromkeys(design_df_filtered.select_dtypes(exclude=["number"]).columns, "category")
            )

            design_clean = design if design.startswith("~") else f"~{design}"

            dds = DeseqDataSet(
                counts=pd.DataFrame(counts_filtered.T, index=design_df_filtered.index),
                metadata=design_df_filtered,
                design=design_clean,
                refit_cooks=True,
            )

            dds.deseq2()

            if model_contrasts is not None and "-" in model_contrasts:
                if "(" in model_contrasts or "+" in model_contrasts.split("-")[1]:
                    raise ValueError(
                        f"Complex contrasts like '{model_contrasts}' are not supported by pydeseq2. "
                        "Use simple pairwise contrasts (e.g., 'GroupA-GroupB') or switch to solver='edger'."
                    )

                parts = model_contrasts.split("-")
                factor_name = design_clean.replace("~", "").split("+")[-1].strip()
                group1 = parts[0].replace(factor_name, "").strip()
                group2 = parts[1].replace(factor_name, "").strip()
                stat_res = DeseqStats(dds, contrast=[factor_name, group1, group2])
            else:
                factor_name = design_clean.replace("~", "").split("+")[-1].strip()
                if not isinstance(design_df_filtered[factor_name], pd.CategoricalDtype):
                    design_df_filtered[factor_name] = design_df_filtered[factor_name].astype("category")
                categories = design_df_filtered[factor_name].cat.categories
                stat_res = DeseqStats(dds, contrast=[factor_name, categories[-1], categories[0]])

            stat_res.summary()
            res = stat_res.results_df

            res = res.rename(
                columns={"baseMean": "logCPM", "log2FoldChange": "logFC", "pvalue": "PValue", "padj": "FDR"}
            )

            res = res[["logCPM", "logFC", "PValue", "FDR"]]

        res.index = sample_adata.var_names[keep_nhoods]  # type: ignore
        if any(col in sample_adata.var.columns for col in res.columns):
            sample_adata.var = sample_adata.var.drop(res.columns, axis=1)
        sample_adata.var = pd.concat([sample_adata.var, res], axis=1)

        self._graph_spatial_fdr(sample_adata)

    def annotate_nhoods(
        self,
        mdata: MuData,
        anno_col: str,
        feature_key: str | None = "rna",
    ):
        """Assigns a categorical label to neighbourhoods, based on the most frequent label among cells in each neighbourhood. This can be useful to stratify DA testing results by cell types or samples.

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            Adds in place.
            - `milo_mdata['milo'].var["nhood_annotation"]`: assigning a label to each nhood
            - `milo_mdata['milo'].var["nhood_annotation_frac"]` stores the fraciton of cells in the neighbourhood with the assigned label
            - `milo_mdata['milo'].varm['frac_annotation']`: stores the fraction of cells from each label in each nhood
            - `milo_mdata['milo'].uns["annotation_labels"]`: stores the column names for `milo_mdata['milo'].varm['frac_annotation']`

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.annotate_nhoods(mdata, anno_col="cell_type")

        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        # Check value is not numeric
        if pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of categorical type - please use milopy.utils.annotate_nhoods_continuous for continuous variables"
            )

        anno_dummies = pd.get_dummies(adata.obs[anno_col])
        anno_count = adata.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
        anno_count_dense = anno_count.toarray()
        anno_sum = anno_count_dense.sum(1)
        anno_frac = np.divide(anno_count_dense, anno_sum[:, np.newaxis])

        anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)
        sample_adata.varm["frac_annotation"] = anno_frac_dataframe.values
        sample_adata.uns["annotation_labels"] = anno_frac_dataframe.columns
        sample_adata.uns["annotation_obs"] = anno_col
        sample_adata.var["nhood_annotation"] = anno_frac_dataframe.idxmax(1)
        sample_adata.var["nhood_annotation_frac"] = anno_frac_dataframe.max(1)

    def annotate_nhoods_continuous(self, mdata: MuData, anno_col: str, feature_key: str | None = "rna"):
        """Assigns a continuous value to neighbourhoods, based on mean cell level covariate stored in adata.obs. This can be useful to correlate DA log-foldChanges with continuous covariates such as pseudotime, gene expression scores etc...

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            Adds in place.
            - `milo_mdata['milo'].var["nhood_{anno_col}"]`: assigning a continuous value to each nhood

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.annotate_nhoods_continuous(mdata, anno_col="nUMI")
        """
        if "milo" not in mdata.mod:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
        adata = mdata[feature_key]

        # Check value is not categorical
        if not pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of continuous type - please use milopy.utils.annotate_nhoods for categorical variables"
            )

        anno_val = adata.obsm["nhoods"].T.dot(csr_matrix(adata.obs[anno_col]).T)

        mean_anno_val = anno_val.toarray() / np.array(adata.obsm["nhoods"].T.sum(1))

        mdata["milo"].var[f"nhood_{anno_col}"] = mean_anno_val

    def add_covariate_to_nhoods_var(self, mdata: MuData, new_covariates: list[str], feature_key: str | None = "rna"):
        """Add covariate from cell-level obs to sample-level obs. These should be covariates for which a single value can be assigned to each sample.

        Args:
            mdata: MuData object
            new_covariates: columns in `milo_mdata[feature_key].obs` to add to `milo_mdata['milo'].obs`.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            None, adds columns to `milo_mdata['milo']` in place

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.add_covariate_to_nhoods_var(mdata, new_covariates=["label"])
        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        sample_col = sample_adata.uns["sample_col"]
        covariates = list(
            set(sample_adata.obs.columns[sample_adata.obs.columns != sample_col].tolist() + new_covariates)
        )
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [covar for covar in covariates if covar not in sample_adata.obs.columns]
            logger.error("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]].astype("str")
        sample_obs.index = sample_obs[sample_col]
        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except ValueError:
            logger.error(
                "Covariates cannot be unambiguously assigned to each sample -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

    def build_nhood_graph(self, mdata: MuData, basis: str = "X_umap", feature_key: str | None = "rna"):
        """Build graph of neighbourhoods used for visualization of DA results.

        Args:
            mdata: MuData object
            basis: Name of the obsm basis to use for layout of neighbourhoods (key in `adata.obsm`).
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            - `milo_mdata['milo'].varp['nhood_connectivities']`: graph of overlap between neighbourhoods (i.e. no of shared cells)
            - `milo_mdata['milo'].var["Nhood_size"]`: number of cells in neighbourhoods

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.build_nhood_graph(mdata)
        """
        adata = mdata[feature_key]
        # # Add embedding positions
        mdata["milo"].varm["X_milo_graph"] = adata[adata.obs["nhood_ixs_refined"] == 1].obsm[basis]
        # Add nhood size
        mdata["milo"].var["Nhood_size"] = np.array(adata.obsm["nhoods"].sum(0)).flatten()
        # Add adjacency graph
        mdata["milo"].varp["nhood_connectivities"] = adata.obsm["nhoods"].T.dot(adata.obsm["nhoods"])
        mdata["milo"].varp["nhood_connectivities"].setdiag(0)
        mdata["milo"].varp["nhood_connectivities"].eliminate_zeros()
        mdata["milo"].uns["nhood"] = {
            "connectivities_key": "nhood_connectivities",
            "distances_key": "",
        }

    def add_nhood_expression(self, mdata: MuData, layer: str | None = None, feature_key: str | None = "rna") -> None:
        """Calculates the mean expression in neighbourhoods of each feature.

        Args:
            mdata: MuData object
            layer: If provided, use `milo_mdata[feature_key][layer]` as expression matrix instead of `milo_mdata[feature_key].X`.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            Updates adata in place to store the matrix of average expression in each neighbourhood in `milo_mdata['milo'].varm['expr']`

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.add_nhood_expression(mdata)

        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots:"
                " feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

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

    def _setup_rpy2(
        self,
    ):
        """Set up rpy2 to run edgeR."""
        try:
            from rpy2.robjects import conversion, numpy2ri, pandas2ri
            from rpy2.robjects.packages import STAP, PackageNotInstalledError, importr
        except ModuleNotFoundError:
            raise ImportError("milo requires rpy2 to be installed.") from None

        try:
            importr("edgeR")
        except ImportError as e:
            raise ImportError("milo requires a valid R installation with edger installed.") from e

        from rpy2.robjects.packages import importr

        edgeR = self._try_import_bioc_library("edgeR")
        limma = self._try_import_bioc_library("limma")
        stats = importr("stats")
        base = importr("base")

        return edgeR, limma, stats, base

    def _try_import_bioc_library(
        self,
        r_package: str,
    ):
        """Import R packages.

        Args:
            r_package: R packages name
        """
        from rpy2.robjects.packages import PackageNotInstalledError, importr

        try:
            _r_lib = importr(r_package)
            return _r_lib
        except PackageNotInstalledError:
            logger.error(
                f"Install Bioconductor library `{r_package!r}` first as `BiocManager::install({r_package!r}).`"
            )
            raise

    def _graph_spatial_fdr(
        self,
        sample_adata: AnnData,
    ):
        """FDR correction weighted on inverse of connectivity of neighbourhoods.

        The distance to the k-th nearest neighbor is used as a measure of connectivity.

        Args:
            sample_adata: Sample-level AnnData.
        """
        # use 1/connectivity as the weighting for the weighted BH adjustment from Cydar
        w = 1 / sample_adata.var["kth_distance"]
        w[np.isinf(w)] = 0

        # Computing a density-weighted q-value.
        pvalues = sample_adata.var["PValue"]
        keep_nhoods = ~pvalues.isna()  # Filtering in case of test on subset of nhoods
        o = pvalues[keep_nhoods].argsort()
        pvalues = pvalues.loc[keep_nhoods].iloc[o]
        w = w.loc[keep_nhoods].iloc[o]

        adjp = np.zeros(shape=len(o))
        adjp[o] = (sum(w) * pvalues / np.cumsum(w))[::-1].cummin()[::-1]
        adjp = np.array([x if x < 1 else 1 for x in adjp])

        sample_adata.var["SpatialFDR"] = np.nan
        sample_adata.var.loc[keep_nhoods, "SpatialFDR"] = adjp

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood_graph(  # pragma: no cover # noqa: D417
        self,
        mdata: MuData,
        *,
        alpha: float = 0.1,
        min_logFC: float = 0,
        min_size: int = 10,
        plot_edges: bool = False,
        title: str = "DA log-Fold Change",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Visualize DA results on abstracted graph (wrapper around sc.pl.embedding).

        Args:
            mdata: MuData object
            alpha: Significance threshold. (default: 0.1)
            min_logFC: Minimum absolute log-Fold Change to show results. If is 0, show all significant neighbourhoods.
            min_size: Minimum size of nodes in visualization. (default: 10)
            plot_edges: If edges for neighbourhood overlaps whould be plotted.
            title: Plot title.
            {common_plot_args}
            **kwargs: Additional arguments to `scanpy.pl.embedding`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata,
            >>>            design='~label',
            >>>            model_contrasts='labelwithdraw_15d_Cocaine-labelwithdraw_48h_Cocaine')
            >>> milo.build_nhood_graph(mdata)
            >>> milo.plot_nhood_graph(mdata)

        Preview:
            .. image:: /_static/docstring_previews/milo_nhood_graph.png
        """
        nhood_adata = mdata["milo"].T.copy()

        if "Nhood_size" not in nhood_adata.obs.columns:
            raise KeyError(
                'Cannot find "Nhood_size" column in adata.uns["nhood_adata"].obs -- \
                    please run milopy.utils.build_nhood_graph(adata)'
            )

        nhood_adata.obs["graph_color"] = nhood_adata.obs["logFC"]
        nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] > alpha, "graph_color"] = np.nan
        nhood_adata.obs["abs_logFC"] = abs(nhood_adata.obs["logFC"])
        nhood_adata.obs.loc[nhood_adata.obs["abs_logFC"] < min_logFC, "graph_color"] = np.nan

        # Plotting order - extreme logFC on top
        nhood_adata.obs.loc[nhood_adata.obs["graph_color"].isna(), "abs_logFC"] = np.nan
        ordered = nhood_adata.obs.sort_values("abs_logFC", na_position="first").index
        nhood_adata = nhood_adata[ordered]

        vmax = np.max([nhood_adata.obs["graph_color"].max(), abs(nhood_adata.obs["graph_color"].min())])
        vmin = -vmax

        fig = sc.pl.embedding(
            nhood_adata,
            "X_milo_graph",
            color="graph_color",
            cmap="RdBu_r",
            size=nhood_adata.obs["Nhood_size"] * min_size,
            edges=plot_edges,
            neighbors_key="nhood",
            sort_order=False,
            frameon=False,
            vmax=vmax,
            vmin=vmin,
            title=title,
            color_map=color_map,
            palette=palette,
            ax=ax,
            show=False,
            **kwargs,
        )

        if return_fig:
            return fig
        plt.show()
        return None

    from collections.abc import Sequence
    from typing import Union

    def plot_nhood_annotation(  # pragma: no cover
        self,
        mdata: MuData,
        *,
        # -------------------------------------------------------------------
        # Styling / filtering parameters for logFC‐based coloring:
        alpha: float = 0.1,
        min_logFC: float = 0.0,
        min_size: int = 10,
        plot_edges: bool = False,
        title: str = "DA log‐Fold Change",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        return_fig: bool = False,
        # -------------------------------------------------------------------
        # New arguments:
        adata_key: str = "milo",
        annotation_key: str | None = "nhood_annotation",
        # -------------------------------------------------------------------
        **kwargs,
    ) -> Figure | None:
        """Visualize Milo differential‐abundance results on the abstracted neighborhood graph.

        By default (annotation_key=None), nodes are colored by filtered logFC (SpatialFDR ≤ alpha,
        |logFC| ≥ min_logFC).  If annotation_key is provided, instead draw node colors from
        mdata[adata_key].obs[annotation_key].

        Args:
            mdata: MuData object containing at least:
                • mdata["milo"] (the Milo‐neighborhood AnnData, transposed)
                • mdata[adata_key] (the AnnData where your annotation lives)
            alpha: Significance threshold for SpatialFDR (only used if annotation_key is None).
            min_logFC: Minimum absolute logFC to display (only used if annotation_key is None).
            min_size:  Scaling factor: actual marker size = Nhood_size × min_size.
            plot_edges: If True, draw edges of the neighborhood overlap graph.
            title: Plot title (ignored if annotation_key is not None; you can override if you like).
            color_map: Passed through to sc.pl.embedding for discrete palettes (optional).
            palette:   Passed through to sc.pl.embedding (optional).
            ax:        Matplotlib Axes to plot on (optional).
            return_fig: If True, return the Figure object instead of calling plt.show().

            adata_key: Key in mdata corresponding to the AnnData whose `.obs` has the annotation.
                    Default = "rna".

            annotation_key: If not None, the name of a column in mdata[adata_key].obs whose values
                            should be used to color the Milo neighborhood graph.  If provided,
                            we ignore all logFC / FDR logic and simply color by that  annotation.
                            Example: "nhood_annotation".  If None, revert to the original logFC‐based coloring.

            **kwargs: Additional keyword arguments passed to sc.pl.embedding.

        Returns:
            If return_fig == True → returns the matplotlib Figure.  Otherwise, shows the plot and returns None.
        """
        # -------------------------------------------------------------------
        # 1) Extract and copy the Milo neighborhood AnnData:
        if "milo" not in mdata.mod:
            raise KeyError('Cannot find "milo" modality in mdata. Did you run milo.build_nhood_graph()?')
        nhood_adata: AnnData = mdata["milo"].T.copy()  # transpose to get nhoods as “cells”

        # -------------------------------------------------------------------
        # 2) If annotation_key is provided, we skip the logFC logic and simply pull
        #    the annotation from mdata[adata_key].obs.  We assume that the neighborhood
        #    IDs in nhood_adata.obs.index correspond to the same index in mdata[adata_key].obs.
        if annotation_key is not None:
            if adata_key not in mdata.mod:
                raise KeyError(f'Cannot find "{adata_key}" modality in mdata.')
            if annotation_key not in nhood_adata.obs.columns:
                raise KeyError(f'Cannot find "{annotation_key}" column in mdata["{adata_key}"].obs.')
            # Copy the annotation over to the neighborhood AnnData’s obs:
            # We assume that nhood_adata.obs.index (e.g. neighborhood IDs) also appear
            # as an index in mdata[adata_key].obs so we can simply reindex.
            annots = mdata[adata_key].T.obs[annotation_key]
            # Subset / align:
            if not all(idx in annots.index for idx in nhood_adata.obs.index):
                missing = set(nhood_adata.obs.index) - set(annots.index)
                raise KeyError(f"The following neighborhood IDs are not found in mdata['{adata_key}'].obs: {missing}")
            nhood_adata.obs["graph_color"] = annots.reindex(nhood_adata.obs.index).values

            # We do not filter by logFC or FDR in this mode; we just plot all neighborhoods.
            # Sorting: if you want to put ‘NaN’ or a particular annotation at the bottom,
            # you could sort by graph_color, but for simplicity we’ll plot in dataset order.
            ordered = list(nhood_adata.obs.index)
            nhood_adata = nhood_adata[ordered]

            # We no longer need “abs_logFC” or SpatialFDR logic, so skip to plotting.
            vmax = None
            vmin = None

            # Call scanpy’s embedding plot:
            fig = sc.pl.embedding(
                nhood_adata,
                basis="X_milo_graph",
                color="graph_color",
                cmap=color_map or "tab20",  # default to a discrete palette if none provided
                size=nhood_adata.obs["Nhood_size"] * min_size,
                edges=plot_edges,
                neighbors_key="nhood",
                sort_order=False,
                frameon=False,
                title=f"{annotation_key}",
                palette=palette,
                ax=ax,
                show=False,
                **kwargs,
            )

            if return_fig:
                return fig
            plt.show()
            return None

        # -------------------------------------------------------------------
        # 3) Otherwise, annotation_key is None → we do the original logFC‐based coloring:
        if "Nhood_size" not in nhood_adata.obs.columns:
            raise KeyError(
                'Cannot find "Nhood_size" column in nhood_adata.obs; please run milo.build_nhood_graph() first.'
            )
        if "logFC" not in nhood_adata.obs.columns or "SpatialFDR" not in nhood_adata.obs.columns:
            raise KeyError(
                'Cannot find "logFC" / "SpatialFDR" columns in nhood_adata.obs; please run milo.da_nhoods() first.'
            )

        # Copy logFC into graph_color, then mask out nonsignificant / small‐effect neighborhoods:
        nhood_adata.obs["graph_color"] = nhood_adata.obs["logFC"]
        nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] > alpha, "graph_color"] = np.nan
        nhood_adata.obs["abs_logFC"] = np.abs(nhood_adata.obs["logFC"])
        nhood_adata.obs.loc[nhood_adata.obs["abs_logFC"] < min_logFC, "graph_color"] = np.nan

        # Plot order: neighborhoods with large |logFC| on top
        nhood_adata.obs.loc[nhood_adata.obs["graph_color"].isna(), "abs_logFC"] = np.nan
        ordered = nhood_adata.obs.sort_values("abs_logFC", na_position="first").index
        nhood_adata = nhood_adata[ordered]

        # Determine symmetric color limits:
        vmax = np.nanmax([nhood_adata.obs["graph_color"].max(), -nhood_adata.obs["graph_color"].min()])
        vmin = -vmax

        # Finally, call scanpy to draw the embedding:
        fig = sc.pl.embedding(
            nhood_adata,
            basis="X_milo_graph",
            color="graph_color",
            cmap=color_map or "RdBu_r",
            size=nhood_adata.obs["Nhood_size"] * min_size,
            edges=plot_edges,
            neighbors_key="nhood",
            sort_order=False,
            frameon=False,
            title=title,
            vmax=vmax,
            vmin=vmin,
            palette=palette,
            ax=ax,
            show=False,
            **kwargs,
        )

        if return_fig:
            return fig
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood(  # pragma: no cover # noqa: D417
        self,
        mdata: MuData,
        ix: int,
        *,
        feature_key: str | None = "rna",
        basis: str = "X_umap",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Visualize cells in a neighbourhood.

        Args:
            mdata: MuData object with feature_key slot, storing neighbourhood assignments in `mdata[feature_key].obsm['nhoods']`
            ix: index of neighbourhood to visualize
            feature_key: Key in mdata to the cell-level AnnData object.
            basis: Embedding to use for visualization.
            color_map: Colormap to use for coloring.
            palette: Color palette to use for coloring.
            ax: Axes to plot on.
            {common_plot_args}
            **kwargs: Additional arguments to `scanpy.pl.embedding`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> milo.plot_nhood(mdata, ix=0)

        Preview:
            .. image:: /_static/docstring_previews/milo_nhood.png
        """
        mdata[feature_key].obs["Nhood"] = mdata[feature_key].obsm["nhoods"][:, ix].toarray().ravel()
        fig = sc.pl.embedding(
            mdata[feature_key],
            basis,
            color="Nhood",
            size=30,
            title="Nhood" + str(ix),
            color_map=color_map,
            palette=palette,
            return_fig=return_fig,
            ax=ax,
            show=False,
            **kwargs,
        )

        if return_fig:
            return fig
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_da_beeswarm(  # pragma: no cover # noqa: D417
        self,
        mdata: MuData,
        *,
        feature_key: str | None = "rna",
        anno_col: str = "nhood_annotation",
        alpha: float = 0.1,
        subset_nhoods: list[str] = None,
        palette: str | Sequence[str] | dict[str, str] | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot beeswarm plot of logFC against nhood labels.

        Args:
            mdata: MuData object
            feature_key: Key in mdata to the cell-level AnnData object.
            anno_col: Column in adata.uns['nhood_adata'].obs to use as annotation. (default: 'nhood_annotation'.)
            alpha: Significance threshold. (default: 0.1)
            subset_nhoods: List of nhoods to plot. If None, plot all nhoods.
            palette: Name of Seaborn color palette for violinplots.
                     Defaults to pre-defined category colors for violinplots.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.annotate_nhoods(mdata, anno_col="cell_type")
            >>> milo.plot_da_beeswarm(mdata)

        Preview:
            .. image:: /_static/docstring_previews/milo_da_beeswarm.png
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run 'milopy.count_nhoods(adata)' first."
            ) from None

        try:
            nhood_adata.obs[anno_col]
        except KeyError:
            raise RuntimeError(
                f"Unable to find {anno_col} in mdata['milo'].var. Run 'milopy.utils.annotate_nhoods(adata, anno_col)' first"
            ) from None

        if subset_nhoods is not None:
            nhood_adata = nhood_adata[nhood_adata.obs[anno_col].isin(subset_nhoods)]

        try:
            nhood_adata.obs["logFC"]
        except KeyError:
            raise RuntimeError(
                "Unable to find 'logFC' in mdata.uns['nhood_adata'].obs. Run 'core.da_nhoods(adata)' first."
            ) from None

        sorted_annos = (
            nhood_adata.obs[[anno_col, "logFC"]].groupby(anno_col).median().sort_values("logFC", ascending=True).index
        )

        anno_df = nhood_adata.obs[[anno_col, "logFC", "SpatialFDR"]].copy()
        anno_df["is_signif"] = anno_df["SpatialFDR"] < alpha
        anno_df = anno_df[anno_df[anno_col] != "nan"]

        try:
            obs_col = nhood_adata.uns["annotation_obs"]
            if palette is None:
                palette = dict(
                    zip(
                        mdata[feature_key].obs[obs_col].cat.categories,
                        mdata[feature_key].uns[f"{obs_col}_colors"],
                        strict=False,
                    )
                )
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                inner=None,
                orient="h",
                palette=palette,
                linewidth=0,
                scale="width",
            )
        except BaseException:  # noqa: BLE001
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                inner=None,
                orient="h",
                linewidth=0,
                scale="width",
            )
        sns.stripplot(
            data=anno_df,
            y=anno_col,
            x="logFC",
            order=sorted_annos,
            size=2,
            hue="is_signif",
            palette=["grey", "black"],
            orient="h",
            alpha=0.5,
        )
        plt.legend(loc="upper left", title=f"< {int(alpha * 100)}% SpatialFDR", bbox_to_anchor=(1, 1), frameon=False)
        plt.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood_counts_by_cond(  # pragma: no cover # noqa: D417
        self,
        mdata: MuData,
        test_var: str,
        *,
        subset_nhoods: list[str] = None,
        log_counts: bool = False,
        return_fig: bool = False,
        ax=None,
        show: bool = True,
    ) -> Figure | None:
        """Plot boxplot of cell numbers vs condition of interest.

        Args:
            mdata: MuData object storing cell level and nhood level information
            test_var: Name of column in adata.obs storing condition of interest (y-axis for boxplot)
            subset_nhoods: List of obs_names for neighbourhoods to include in plot. If None, plot all nhoods.
            log_counts: Whether to plot log1p of cell counts.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run milopy.count_nhoods(mdata) first"
            ) from None

        if subset_nhoods is None:
            subset_nhoods = nhood_adata.obs_names

        pl_df = pd.DataFrame(nhood_adata[subset_nhoods].X.toarray(), columns=nhood_adata.var_names).melt(
            var_name=nhood_adata.uns["sample_col"], value_name="n_cells"
        )
        pl_df = pd.merge(pl_df, nhood_adata.var)
        pl_df["log_n_cells"] = np.log1p(pl_df["n_cells"])
        if not log_counts:
            sns.boxplot(data=pl_df, x=test_var, y="n_cells", color="lightblue", ax=ax)
            sns.stripplot(data=pl_df, x=test_var, y="n_cells", color="black", s=3, ax=ax)
            if ax:
                ax.set_ylabel("# cells")
            else:
                plt.ylabel("# cells")
        else:
            sns.boxplot(data=pl_df, x=test_var, y="log_n_cells", color="lightblue", ax=ax)
            sns.stripplot(data=pl_df, x=test_var, y="log_n_cells", color="black", s=3, ax=ax)
            if ax:
                ax.set_ylabel("log(# cells + 1)")
            else:
                plt.ylabel("log(# cells + 1)")

        if ax:
            ax.tick_params(axis="x", rotation=90)
            ax.set_xlabel(test_var)
        else:
            plt.xticks(rotation=90)
            plt.xlabel(test_var)

        if return_fig:
            return plt.gcf()

        if ax is None:
            plt.show()

        if return_fig:
            return plt.gcf()
        if show:
            plt.show()

        return None

    def _group_nhoods_from_adjacency(
        self,
        adjacency: spmatrix,
        da_res: pd.DataFrame,
        is_da: np.ndarray,
        merge_discord: bool = False,
        overlap: int = 1,
        max_lfc_delta: float | None = None,
        subset_nhoods=None,
    ) -> np.ndarray:
        """Core neighborhood‐grouping logic (vectorized, no Python loops).

        Inputs:
        - adjacency: scipy.sparse square matrix of shape (N, N),
                    storing neighborhood adjacency (overlap counts).
        - da_res:     pandas.DataFrame, length N, with columns 'SpatialFDR' and 'logFC'.
        - is_da:      1‐D boolean array of length N, True where da_res.SpatialFDR < cutoff.
        - merge_discord: if False, zero edges between DA‐pairs with opposite logFC sign.
        - overlap:    integer threshold; zero edges with weight < overlap.
        - max_lfc_delta: if not None, zero edges whose |logFC[i] - logFC[j]| > max_lfc_delta.
        - subset_nhoods:    None or one of:
                • boolean mask (length N),
                • list/array of integer indices,
                • list/array of string names (matching da_res.index).

        Returns:
        - labels: NumPy array of dtype string, length = (# of neighborhoods after subsetting),
                    giving a Louvain cluster label for each neighborhood (in the same order as da_res).
        """
        # 1) Optional subsetting of neighborhoods ---------------------------------------------------
        #    We allow subset_nhoods to be a boolean mask, a list of integer indices, or a list of names.
        if subset_nhoods is not None:
            # 1) boolean‐mask case first
            if isinstance(subset_nhoods, pd.Series | np.ndarray) and subset_nhoods.dtype == bool:
                if len(subset_nhoods) != adjacency.shape[0]:
                    raise ValueError("Boolean subset_nhoods must have length = number of neighborhoods.")
                mask = np.asarray(subset_nhoods, dtype=bool)

            # 2) integer‐index or name list next
            elif isinstance(subset_nhoods, list | np.ndarray):
                arr = np.asarray(subset_nhoods)
                # integer indices?
                if np.issubdtype(arr.dtype, np.integer):
                    mask = np.zeros(adjacency.shape[0], dtype=bool)
                    mask[arr.astype(int)] = True
                # name list?
                else:
                    names = da_res.index.to_numpy(dtype=str)
                    mask = np.isin(names, arr.astype(str))

            else:
                raise ValueError("subset_nhoods must be a boolean mask, a list of indices, or a list of names.")

            adjacency = adjacency[mask, :][:, mask]
            da_res = da_res.loc[mask].copy()
            is_da = is_da[mask]

        M = adjacency.shape[0]
        if da_res.shape[0] != M or is_da.shape[0] != M:
            raise ValueError("Length of da_res and is_da must match adjacency dimension after subsetting.")

        # 2) Convert adjacency to CSR (if not already) and then to COO for a flat edge list ----------------
        adjacency = csr_matrix(adjacency) if not issparse(adjacency) else adjacency.tocsr()

        Acoo = adjacency.tocoo()
        rows = Acoo.row  # array of length E = number of nonzero edges
        cols = Acoo.col
        data = Acoo.data  # the actual overlap counts

        # 3) Precompute logFC and sign arrays -------------------------------------------------------------------
        lfc_vals = da_res["logFC"].values  # shape = (M,)
        signs = np.sign(lfc_vals)  # sign(lfc_i), shape = (M,)

        # 4) Build Boolean masks (length E) for each filter ------------------------------------------------------

        # 4.1) “Discord” filter: if merge_discord=False, drop any edge (i,j) where both i,j are DA
        #      AND sign(lfc_i) * sign(lfc_j) < 0 (opposite signs).
        if merge_discord:
            keep_discord = np.ones_like(data, dtype=bool)
        else:
            # For each edge k at (i=rows[k], j=cols[k]), check if both are DA AND signs differ
            is_da_rows = is_da[rows]  # True if endpoint‐i is DA
            is_da_cols = is_da[cols]  # True if endpoint‐j is DA
            sign_rows = signs[rows]
            sign_cols = signs[cols]

            # discord_pair[k] = True if both DA and (signs multiply < 0)
            discord_pair = (is_da_rows & is_da_cols) & ((sign_rows * sign_cols) < 0)
            keep_discord = ~discord_pair

        # 4.2) “Overlap” filter: drop any edge whose current weight < overlap
        keep_overlap = np.ones_like(data, dtype=bool) if overlap <= 1 else data >= overlap

        # 4.3) “Δ logFC” filter: drop any edge where |lfc_i - lfc_j| > max_lfc_delta
        if max_lfc_delta is None:
            keep_lfc = np.ones_like(data, dtype=bool)
        else:
            # Compute |lfc_vals[rows] - lfc_vals[cols]| vectorized
            lfc_edge_diffs = np.abs(lfc_vals[rows] - lfc_vals[cols])
            keep_lfc = lfc_edge_diffs <= max_lfc_delta

        # 5) Combine all masks into a single “keep” mask ----------------------------------------------------------------
        keep_mask = keep_discord & keep_overlap & keep_lfc

        # 6) Rebuild a new, pruned adjacency in COO form (only edges where keep_mask=True) --------------------------
        new_rows = rows[keep_mask]
        new_cols = cols[keep_mask]
        new_data = data[keep_mask]

        # If you want an unweighted graph (just connectivity), you could do `new_data = np.ones_like(new_rows)`.
        # But to mirror MiloR exactly, we preserve the original overlap counts until the final binarization.
        pruned_adj = coo_matrix((new_data, (new_rows, new_cols)), shape=(M, M)).tocsr()

        # 7) Binarize: every surviving edge → 1, then convert to CSR ----------------------------------------------------------------
        pruned_adj = (pruned_adj > 0).astype(int).tocsr()

        # 8) Build an igraph from the final adjacency --------------------------------------------------------------------------------
        #    We can use scanpy’s utility to convert a sparse (0/1) matrix to igraph.
        # Issue with dematrix after subsetting adjacency matrix:
        #     dematrix in sc._utils.get_igraph_from_adjacency does not convert to dense numpy matrix.
        #     Trying direct conversion to igraph:
        g = sc._utils.get_igraph_from_adjacency(pruned_adj, directed=False)

        # 9) Run Louvain (multilevel) clustering on the unweighted graph ----------------------------------------------------------------
        #    By not providing a “weights” argument, igraph treats every edge as weight=1.
        clustering = g.community_multilevel(weights=None)
        labels = np.array(clustering.membership, dtype=str)  # length = M, dtype = 'str'

        # 10) Return the cluster labels array (strings), in the same order as da_res.index ---------------------------------------
        #     If subset_nhoods was not None, these labels correspond to rows where mask=True.
        return labels

    def group_nhoods(
        self,
        data: Any,
        key: str | None = "milo",
        da_res: pd.DataFrame | None = None,
        da_fdr: float = 0.1,
        overlap: int = 1,
        max_lfc_delta: float | None = None,
        merge_discord: bool = False,
        subset_nhoods=None,
    ) -> pd.DataFrame:
        """Python equivalent of MiloR’s groupNhoods(), using AnnData and its `varp["nhood_connectivities"]`.

        Parameters
        ----------
        adata : AnnData
            Must contain:
            - `adata.var` with columns "SpatialFDR" (float) and "logFC" (float)
            - `adata.varp["nhood_connectivities"]` as an (N×N) sparse adjacency matrix
        da_res : pd.DataFrame, optional
            If provided, must match `adata.var`. Otherwise, `adata.var` is used directly.
        da_fdr : float, default=0.1
            Neighborhoods with `SpatialFDR < da_fdr` are called “DA.”
        overlap : int, default=1
            Drop any adjacency entry (edge) with weight < overlap.
        max_lfc_delta : float or None, default=None
            If not None, drop edges where |lfc_i - lfc_j| > max_lfc_delta.
        merge_discord : bool, default=False
            If False, drop edges between DA neighborhoods whose logFC signs disagree.
        subset_nhoods : None or boolean mask / list of indices / list of names
            If provided, only cluster that subset of neighborhoods.

        Returns:
        -------
        pd.DataFrame
            A copy of `adata.var`, with a new column "nhood_groups" of dtype string giving each
            neighborhood’s cluster label (or `pd.NA` if it wasn’t in `subset_nhoods`).


        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.group_nhoods(mdata)
        """
        if isinstance(data, AnnData):
            adata = data
        elif isinstance(data, MuData):
            if key is None:
                raise ValueError("If `data` is a MuData object, `key` must be specified.")
            adata = data[key]
        else:
            raise ValueError("`data` must be an AnnData or MuData object.")

        # 1) Validate input ---------------------------------------------------------------------------------------------
        if not isinstance(adata, AnnData):
            raise ValueError("`adata` must be an AnnData object.")

        # 2) Get or check `da_res` --------------------------------------------------------------------------------------
        if da_res is None:
            da_res = adata.var
        # If user passed their own da_res, ensure indexes match
        elif not da_res.index.equals(adata.var.index):
            raise ValueError("`da_res` index must match `adata.var.index`.")

        # Ensure required columns exist
        if "SpatialFDR" not in da_res.columns or "logFC" not in da_res.columns:
            raise ValueError("`da_res` (adata.var) must contain columns 'SpatialFDR' and 'logFC'.")

        # 3) Identify “DA” neighborhoods by FDR cutoff -------------------------------------------------------------------
        fdr_values = da_res["SpatialFDR"].values
        if np.all(pd.isna(fdr_values)):
            raise ValueError("All `SpatialFDR` values are NA; cannot determine DA neighborhoods.")
        is_da = fdr_values < da_fdr

        n_da = int(is_da.sum())
        if n_da == 0:
            raise ValueError(f"No DA neighborhoods found at FDR < {da_fdr}.")

        # 4) Extract adjacency ------------------------------------------------------------------------------------------
        if "nhood_connectivities" not in adata.varp:
            raise KeyError("`adata.varp` does not contain 'nhood_connectivities'. Did you run buildNhoodGraph?")
        adjacency = adata.varp["nhood_connectivities"]

        # 5) Call core worker to get string labels ----------------------------------------------------------------------
        labels = self._group_nhoods_from_adjacency(
            adjacency=adjacency,
            da_res=da_res,
            is_da=is_da,
            merge_discord=merge_discord,
            overlap=overlap,
            max_lfc_delta=max_lfc_delta,
            subset_nhoods=subset_nhoods,
        )

        # 6) Write results back into `adata.var["NhoodGroup"]` -----------------------------------------------------------
        N_full = adata.var.shape[0]
        out = np.array([pd.NA] * N_full, dtype=object)

        if subset_nhoods is None:
            # no subsetting: every label goes into the full array
            out[:] = labels
        else:
            # 1) Boolean‐mask case first
            if isinstance(subset_nhoods, pd.Series | np.ndarray) and getattr(subset_nhoods, "dtype", None).kind == "b":
                if len(subset_nhoods) != N_full:
                    raise ValueError("Boolean subset_nhoods must have length = number of neighborhoods.")
                mask_idx = np.asarray(subset_nhoods, dtype=bool)

            # 2) Integer‐index or name‐list next
            elif isinstance(subset_nhoods, list | np.ndarray):
                arr = np.asarray(subset_nhoods)
                if np.issubdtype(arr.dtype, np.integer):
                    mask_idx = np.zeros(N_full, dtype=bool)
                    mask_idx[arr.astype(int)] = True
                else:
                    names = adata.var.index.to_numpy(dtype=str)
                    mask_idx = np.isin(names, arr.astype(str))

            else:
                raise ValueError("`subset_nhoods` must be a boolean mask, a list of indices, or a list of names.")

            # 3) Place the M labels back into the N-length output
            out[mask_idx] = labels

        adata.var["nhood_groups"] = out

    def _nhood_labels_to_cells_last_wins(self, mdata, nhood_group_obs: str = "nhood_groups", subset_nhoods=None):
        nhood_mat = mdata["rna"].obsm["nhoods"]

        da_res = mdata["milo"].var.copy()
        ### update for categorical nhood_group_obs to control order of levels
        # This turns the nhood_group_obs into a CategoricalDtype if it isn't already
        col = nhood_group_obs
        # if it isn’t already a CategoricalDtype, cast it
        if not isinstance(da_res[col].dtype, pd.api.types.CategoricalDtype):
            da_res[col] = da_res[col].astype("category")

        nhood_mat = AnnData(X=nhood_mat)
        nhood_mat.obs_names = mdata["rna"].obs_names
        nhood_mat.var_names = [str(i) for i in range(nhood_mat.shape[1])]

        nhs_da_gr = da_res[nhood_group_obs].copy()
        nhs_da_gr.index = da_res.index.to_numpy()

        # We want to drop NAs from the nhood_group_obs column, not the whole DataFrame
        # nhood_gr = da_res.dropna()[nhood_group_obs].unique()
        nhood_gr = da_res[nhood_group_obs].cat.categories

        nhs = nhood_mat.copy()

        # if(!is.null(subset.nhoods)){
        #  nhs <- nhs[,subset.nhoods]
        ##   # ## Remove cells out of neighbourhoods of interest
        #   # nhs <- nhs[rowSums(nhs) > 0,]
        # }

        if subset_nhoods is not None:
            nhs = nhs[:, subset_nhoods]
            nhs = nhs[np.asarray(nhs.X.sum(1)).ravel() > 0, :].copy()

        fake_meta = pd.DataFrame(
            {
                "CellID": nhs.obs_names[(np.asarray(nhs.X.sum(1).flatten()).ravel() != 0)],
                # "Nhood_Group": [np.nan for _ in range((np.asarray(nhs.X.sum(1).flatten()).ravel() != 0).sum())],
                "Nhood_Group": [pd.NA for _ in range((np.asarray(nhs.X.sum(1).flatten()).ravel() != 0).sum())],
            }
        )
        fake_meta.index = fake_meta["CellID"].copy()

        for i in range(len(nhood_gr)):
            cur_nh_group = nhood_gr[i]

            nhood_x = nhs_da_gr == cur_nh_group
            nhood_x = nhood_x[nhood_x]
            nhood_x = nhood_x.index
            nhood_x = np.asarray(nhood_x)

            nhs = nhs[nhs.X.sum(1) > 0, :].copy()

            mask = np.asarray(nhs[:, nhood_x].X.sum(1)).ravel() > 0
            nhood_gr_cells = nhs.obs_names[mask]

            fake_meta.loc[nhood_gr_cells, "Nhood_Group"] = np.where(
                (fake_meta.loc[nhood_gr_cells, "Nhood_Group"]).isna(),
                nhood_gr[i],
                pd.NA,
            )

        mdata["rna"].obs["nhood_groups"] = pd.NA
        mdata["rna"].obs.loc[fake_meta.CellID.to_list(), "nhood_groups"] = fake_meta.Nhood_Group.to_numpy()

    def _get_cells_in_nhoods(self, adata, nhood_ids):
        """Get cells in neighbourhoods of interest, store the number of neighbourhoods for each cell in adata.obs['in_nhoods']."""
        in_nhoods = np.array(adata.obsm["nhoods"][:, nhood_ids.astype("int")].sum(1))
        adata.obs["in_nhoods"] = in_nhoods

    def _nhood_labels_to_cells_exclude_overlaps(
        self,
        mdata,
        nhood_group_obs: str = "nhood_groups",
        min_n_nhoods: int = 3,
    ):
        groups = mdata["milo"].var[nhood_group_obs].dropna().unique()
        for g in groups:
            nhoods_oi = mdata["milo"].var_names[mdata["milo"].var[nhood_group_obs] == g]
            self._get_cells_in_nhoods(mdata["rna"], nhoods_oi)
            mdata["rna"].obs[f"in_nhoods_{g}"] = mdata["rna"].obs["in_nhoods"].copy()

        ## Find most representative group (cell belongs to mostly to neighbourhoods of that group)
        mdata["rna"].obs["nhood_groups"] = np.nan
        mdata["rna"].obs["nhood_groups"] = mdata["rna"].obs[[f"in_nhoods_{g}" for g in groups]].idxmax(1)
        ## Keep only if cell is in at least min_n_nhoods nhoods of the same group
        mdata["rna"].obs.loc[
            ~(mdata["rna"].obs[[f"in_nhoods_{g}" for g in groups]] > min_n_nhoods).any(axis=1), "nhood_groups"
        ] = np.nan
        ### Remove the in_nhoods in nhood_groups columns
        mdata["rna"].obs["nhood_groups"] = (
            mdata["rna"].obs["nhood_groups"].apply(lambda x: x.split("_")[-1] if isinstance(x, str) else x)
        )
        mdata["rna"].obs["nhood_groups"] = mdata["rna"].obs["nhood_groups"].str.removeprefix("in_nhoods_")

    def annotate_cells_from_nhoods(
        self,
        mdata: MuData,
        nhood_group_obs: str = "nhood_groups",
        subset_nhoods: list[str] | None = None,
        min_n_nhoods: int = 3,
        mode: Literal["last_wins", "exclude_overlaps"] = "last_wins",
    ) -> None:
        """Annotate cells with neighborhood group labels.

        Parameters:
        -----------
        mdata: MuData object with 'milo' modality.
        nhood_group_obs: Column in `mdata["milo"].var` to use for neighborhood group labels.
        subset_nhoods: List of neighborhood IDs to consider. If None, all neighborhoods are used.
        min_n_nhoods: Minimum number of neighborhoods a cell must belong to in order to be annotated. Used for mode "exclude_overlaps".
        mode: Mode for annotation. Options are:
            - "last_wins": Last neighborhood label wins, adapted from miloR.
            - "exclude_overlaps": Exclude overlaps, keeping only the most representative cells within groups.

        Returns:
        --------
        None: Modifies `mdata["rna"].obs` in place, adding a column `nhood_groups` with the assigned labels.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.group_nhoods(mdata)
            >>> milo.annotate_cells_from_nhoods(mdata, mode="last_wins")
            >>> milo.annotate_cells_from_nhoods(mdata, mode="exclude_overlaps", min_n_nhoods=3)
        """
        if mode == "last_wins":
            self._nhood_labels_to_cells_last_wins(mdata, nhood_group_obs, subset_nhoods)
        elif mode == "exclude_overlaps":
            self._nhood_labels_to_cells_exclude_overlaps(mdata, nhood_group_obs, min_n_nhoods)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'last_wins' or 'exclude_overlaps'.")

    def get_mean_expression(self, adata, groupby: str, var_names: list[str]) -> pd.DataFrame:
        """Compute the *mean* expression (counts) of each gene in `var_names`, stratified by a categorical column `groupby` in adata.obs.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing the expression matrix in `.X` and categorical metadata in `.obs`.
        groupby : str
            Column name in `adata.obs` that contains the categorical variable to group by.
        var_names : list of str
            List of gene names (or variable names) for which to compute the mean expression.

        Returns:
        -------
        mean_df : pandas.DataFrame (n_genes × n_groups)
            Rows are `var_names`, columns are the unique categories of `adata.obs[groupby]`.
            Each entry mean_df.loc[g, grp] = (sum of adata[:, g].X over all cells in `grp`)
                                    / (number of cells in that `grp`).
        """
        # 1) Subset the matrix to just the columns (genes) in var_names:
        subX = adata[:, var_names].X.copy()  # shape: (n_cells, n_genes)

        # 2) Build a one‐hot (dummy) matrix of shape (n_cells, n_groups):
        groups = pd.get_dummies(adata.obs[groupby], drop_first=False)
        # groups.values is (n_cells, n_groups).  groups.sum() is a Series: number of cells per group.
        n_per_group = groups.sum().astype(float)  # length = n_groups

        # 3) Compute Σ_counts_{gene i, group j} by matrix‐multiplication:
        #    - If subX is sparse, convert to a CSR; otherwise treat as dense.
        if issparse(subX):
            subX = csr_matrix(subX)
            sum_counts = subX.T.dot(csr_matrix(groups.values))  # shape (n_genes, n_groups), sparse
            sum_counts = sum_counts.toarray()  # convert to dense (n_genes, n_groups)
        else:
            # dense case: subX is (n_cells, n_genes), so subX.T is (n_genes, n_cells),
            # dot with (n_cells, n_groups) → (n_genes, n_groups)
            sum_counts = subX.T.dot(groups.values)

        # 4) Divide each column (group) by its total cell count to get means:
        #    We want mean_counts[i, j] = sum_counts[i, j] / n_per_group[j].
        #    n_per_group.values is shape (n_groups,), so broadcasting works.
        mean_mat = sum_counts / n_per_group.values[np.newaxis, :]

        # 5) Build a DataFrame, indexed by var_names, columns = groups.columns
        mean_df = pd.DataFrame(mean_mat, index=var_names, columns=groups.columns)
        return mean_df

    def _run_edger_contrasts(
        self,
        pdata: AnnData,
        nhood_group_obs: str,
        *,
        formula: str,
        group_to_compare: str | None = None,
        baseline: str | None = None,
        subset_samples: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run edgeR QLF tests on a pseudobulk AnnData.

        If `group_to_compare` and `baseline` are both provided, performs exactly that two‐level contrast.
        Otherwise, loops one‐vs‐rest over all levels of pdata.obs[nhood_group_obs].

        Returns a pandas DataFrame with columns:
        ["variable", "logFC", "PValue", "adj_PValue"]   (plus "group" if one‐vs‐rest).
        """
        if not _is_counts(pdata.X):
            raise ValueError("`pdata.X` appears to be raw counts, but this function expects continuous expression.")

        edger, limma, rstats, rbase = self._setup_rpy2()
        import rpy2.robjects as ro
        from rpy2.robjects import IntVector, StrVector, baseenv, numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter

        # 2) Build a pandas DataFrame for sample‐level covariates

        # 6) Single‐contrast vs one‐vs‐rest
        results_list: list[Any] = []

        # If a specific two‐level contrast was given:
        if group_to_compare is not None and baseline is not None:
            # subset pdata to only those two groups
            pdata = pdata[pdata.obs[nhood_group_obs].isin([baseline, group_to_compare])].copy()
            if pdata.shape[0] == 0:
                raise ValueError(f"No samples found with {nhood_group_obs} in [{baseline}, {group_to_compare}].")

            ### build R count matrix
            count_mat = pdata.X.toarray().T if hasattr(pdata.X, "toarray") else np.asarray(pdata.X).T
            with localconverter(ro.default_converter + numpy2ri.converter):
                rmat = numpy2ri.py2rpy(count_mat)

            r_colnames = StrVector(np.asarray(pdata.obs_names))
            r_rownames = StrVector(np.asarray(pdata.var_names))
            dim_list = ro.r.list(r_rownames, r_colnames)

            assign_dim = baseenv["dimnames<-"]
            rmat = assign_dim(rmat, dim_list)

            # Build the DGEList
            dge = edger.DGEList(counts=rmat)

            # build R model matrix from sample_obs, setting levels of nhood_group_obs to baseline and group_to_compare
            sample_obs = pdata.obs.copy()
            if group_to_compare is not None and baseline is not None:
                # If a specific two‐level contrast was given, subset to those samples only
                sample_obs[nhood_group_obs] = pd.Categorical(
                    sample_obs[nhood_group_obs].values, categories=[baseline, group_to_compare]
                )

            with localconverter(ro.default_converter + pandas2ri.converter):
                robs = pandas2ri.py2rpy(sample_obs)
            design_r = rstats.model_matrix(rstats.as_formula(formula), robs)

            # Fit the quasi‐likelihood model
            dge = edger.calcNormFactors(dge, method="TMM")

            fit = edger.glmQLFit(dge, design_r, robust=True)

            # Now run QLF test with that contrast
            qlf = edger.glmQLFTest(fit, coef=nhood_group_obs + group_to_compare)
            top = edger.topTags(qlf, sort_by="none", n=np.inf)[0]
            # Convert top (an R data.frame) to pandas
            with localconverter(ro.default_converter + pandas2ri.converter):
                top_df = pandas2ri.rpy2py(top)

            # Clean up column names (they come as “logFC”, “PValue”, “FDR”)
            top_df = top_df.rename(columns={"FDR": "adj_p_value", "PValue": "p_value", "logFC": "log_fc"})
            top_df = top_df.reset_index().rename(columns={"index": "variable"})
            return top_df
            # results_list.append(top_df[["variable", "logFC", "PValue", "adj_PValue"]])

        else:
            ### build R count matrix
            count_mat = pdata.X.toarray().T if hasattr(pdata.X, "toarray") else np.asarray(pdata.X).T
            with localconverter(ro.default_converter + numpy2ri.converter):
                rmat = numpy2ri.py2rpy(count_mat)

            r_colnames = StrVector(np.asarray(pdata.obs_names))
            r_rownames = StrVector(np.asarray(pdata.var_names))
            dim_list = ro.r.list(r_rownames, r_colnames)

            assign_dim = baseenv["dimnames<-"]
            rmat = assign_dim(rmat, dim_list)

            # Build the DGEList
            dge = edger.DGEList(counts=rmat)
            sample_obs = pdata.obs.copy()

            col = sample_obs[nhood_group_obs]
            if isinstance(col, pd.api.types.CategoricalDtype):
                unique_groups = col.cat.categories.tolist()
            else:
                unique_groups = col.unique().tolist()

            results_list = []

            for grp in unique_groups:
                # build group‐specific design matrix
                tmp_obs = pdata.obs.copy()

                tmp_obs[nhood_group_obs] = [x if x == grp else "rest" for x in tmp_obs[nhood_group_obs]]
                tmp_obs[nhood_group_obs] = pd.Categorical(tmp_obs[nhood_group_obs].values, categories=["rest", grp])
                with localconverter(ro.default_converter + pandas2ri.converter):
                    robs = pandas2ri.py2rpy(tmp_obs)
                design_r = rstats.model_matrix(rstats.as_formula(formula), robs)

                # 6) Build a DGEList for this subset (or reuse dge_full but safer to make a fresh one)
                dge = edger.calcNormFactors(dge, method="TMM")
                fit_sub = edger.glmQLFit(dge, design_r, robust=True)

                # 8) QLF test on that contrast
                qlf_sub = edger.glmQLFTest(fit_sub, coef=nhood_group_obs + grp)
                top_sub = edger.topTags(qlf_sub, sort_by="none", n=np.inf)[0]

                with localconverter(ro.default_converter + pandas2ri.converter):
                    top_df_sub = pandas2ri.rpy2py(top_sub)

                top_df_sub = top_df_sub.rename(columns={"FDR": "adj_p_value", "PValue": "p_value", "logFC": "log_fc"})
                top_df_sub = top_df_sub.reset_index().rename(columns={"index": "variable"})
                top_df_sub["group"] = grp

                results_list.append(top_df_sub[["variable", "log_fc", "p_value", "adj_p_value", "group"]])

            # 9) Concatenate and return
            final_df = pd.concat(results_list, ignore_index=True)
            return final_df

    def _run_pydeseq2_contrasts(
        self,
        pdata: AnnData,
        nhood_group_obs: str,
        *,
        formula: str,
        group_to_compare: str | None = None,
        baseline: str | None = None,
        alpha: float = 0.05,
        quiet: bool = True,
    ) -> pd.DataFrame:
        """Run PyDESeq2 on a pseudobulk AnnData (`pdata`) with a given neighborhood grouping, using exactly the design `formula` you supply.

        Parameters
        ----------
        pdata : AnnData
            Pseudobulk AnnData, where .obs[nhood_group_obs] is a categorical allowing
            you to compare levels.

        nhood_group_obs : str
            The column name in pdata.obs that holds the neighborhood groups.

        formula : str
            An R‐style design formula, e.g. "~ batch + Nhood_Group".  Must include
            `nhood_group_obs` as one of the terms.  This is used verbatim for both the
            single‐contrast and one‐vs‐rest calls.

        group_to_compare : Optional[str]
            If non‐None (and `baseline` is also non‐None), run only the single contrast
            [nhood_group_obs, group_to_compare, baseline] with design = `formula`.

        baseline : Optional[str]
            If non‐None (and `group_to_compare` is non‐None), run only that one contrast.
            If either is None, the function does a one‐vs‐rest loop over all levels of
            pdata.obs[nhood_group_obs].

        alpha : float, default=0.05
            Significance threshold passed to PyrDESeq2’s `DeseqStats`.

        quiet : bool, default=True
            Whether to suppress PyDESeq2’s “DESeq2()” progress messages.

        Returns:
        -------
        pd.DataFrame
            If `group_to_compare` and `baseline` are provided: a DataFrame with columns
            ["variable","log_fc","p_value","adj_p_value"], sorted by p_value.

            Otherwise (one‐vs‐rest): a concatenated DataFrame with those columns plus
            a “group” column indicating which level was tested vs “rest.”
        """
        if find_spec("pydeseq2") is None:
            raise ImportError("pydeseq2 is required but not installed. Install with: pip install pydeseq2")
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        # Basic check: if both group_to_compare & baseline are provided, do just that contrast
        if (group_to_compare is not None) ^ (baseline is not None):
            raise ValueError("You must supply either both `group_to_compare` and `baseline`, or neither.")

        # 1) Single contrast branch
        if group_to_compare is not None and baseline is not None:
            # 1a) Build the DESeqDataSet using exactly the provided `formula`
            dds = DeseqDataSet(adata=pdata, design=formula, quiet=quiet)
            dds.deseq2()

            # 1b) Run PyrDESeq2 with the single contrast
            stat_res = DeseqStats(
                dds,
                contrast=[nhood_group_obs, group_to_compare, baseline],
                alpha=alpha,
                quiet=quiet,
            )
            stat_res.summary()

            # 1c) Collect results into a pandas DataFrame
            df = (
                pd.DataFrame(stat_res.results_df)
                .rename(
                    columns={
                        "log2FoldChange": "log_fc",
                        "pvalue": "p_value",
                        "padj": "adj_p_value",
                    }
                )
                .sort_values("p_value")
                .reset_index(names=["variable"])
            )
            return df

        # 2) One‐vs‐rest: get all levels of nhood_group_obs
        col = pdata.obs[nhood_group_obs]
        unique_groups = (
            col.cat.categories.tolist() if isinstance(col, pd.api.types.CategoricalDtype) else col.unique().tolist()
        )

        all_results = []
        for grp in unique_groups:
            # 2a) Copy pdata so we can relabel group vs “rest”
            tmp = pdata.copy()

            # 2b) Ensure “rest” is a valid category, then recode everything not == grp → "rest"
            tmp.obs[nhood_group_obs] = tmp.obs[nhood_group_obs].cat.add_categories("rest")
            tmp.obs[nhood_group_obs] = tmp.obs[nhood_group_obs].apply(lambda x, grp=grp: x if x == grp else "rest")
            # Now tmp.obs[nhood_group_obs] has exactly two levels: grp and "rest"

            # 2c) Build DESeqDataSet on `tmp` using **the same** `formula`
            #     (The formula must reference nhood_group_obs so that “grp” vs “rest” is meaningful.)
            dds = DeseqDataSet(adata=tmp, design=formula, quiet=quiet)
            dds.deseq2()

            # 2d) Run PyrDESeq2 with contrast = [nhood_group_obs, grp, "rest"]
            stat_res = DeseqStats(
                dds,
                contrast=[nhood_group_obs, grp, "rest"],
                alpha=alpha,
                quiet=quiet,
            )
            stat_res.summary()

            # 2e) Extract results, rename, attach “group = grp”
            df = (
                pd.DataFrame(stat_res.results_df)
                .rename(
                    columns={
                        "log2FoldChange": "log_fc",
                        "pvalue": "p_value",
                        "padj": "adj_p_value",
                    }
                )
                .reset_index(names=["variable"])
                .assign(group=grp)
                .sort_values("p_value")
            )

            all_results.append(df)

        # 3) Concatenate and return
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df

    def _filter_by_expr_edger(self, pdata, formula, **kwargs):
        """Filter genes in `pdata` based on expression criteria using edgeR."""
        edger, _, rstats, rbase = self._setup_rpy2()
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter

        counts = pdata.X
        counts = counts.toarray().T if hasattr(counts, "toarray") else np.asarray(counts).T
        with localconverter(ro.default_converter + numpy2ri.converter):
            rcounts = numpy2ri.py2rpy(counts)
        obs = pdata.obs
        with localconverter(ro.default_converter + pandas2ri.converter):
            robs = pandas2ri.py2rpy(obs)
        rdesign = rstats.model_matrix(rstats.as_formula(formula), robs)
        rkeep = edger.filterByExpr(rcounts, design=rdesign, **kwargs)
        keep = list(rkeep)

        pdata._inplace_subset_var(keep)

    def _filter_highly_variable_scanpy(self, pdata, n_top_genes=7500, target_sum=1e6, **kwargs):
        if _is_counts(pdata.X):
            pdata.layers["normalized"] = pdata.X.copy()
            sc.pp.normalize_total(
                pdata,
                target_sum=target_sum,
                layer="normalized",
            )
            sc.pp.log1p(pdata, layer="normalized")
        else:
            pdata.layers["normalized"] = pdata.X.copy()

        sc.pp.highly_variable_genes(pdata, layer="normalized", n_top_genes=n_top_genes, subset=True, **kwargs)

    def _filter_highly_variable_scran(self, pdata, n_top_genes):
        scran = self._try_import_bioc_library("scran")
        scuttle = self._try_import_bioc_library("scuttle")
        singlecellexperiment = self._try_import_bioc_library("SingleCellExperiment")

        import rpy2.robjects as ro
        from rpy2.robjects import ListVector, numpy2ri, pandas2ri
        from rpy2.robjects.conversion import localconverter

        counts = pdata.X
        with localconverter(ro.default_converter + numpy2ri.converter):
            rcounts = numpy2ri.py2rpy(counts.T)
        obs = pdata.obs
        var = pdata.var

        with localconverter(ro.default_converter + pandas2ri.converter):
            robs = pandas2ri.py2rpy(obs)
            rvar = pandas2ri.py2rpy(var)

        if _is_counts(counts):
            sce = singlecellexperiment.SingleCellExperiment(ListVector({"counts": rcounts}), colData=robs, rowData=rvar)
            sce = scuttle.logNormCounts(sce)
        else:
            sce = singlecellexperiment.SingleCellExperiment(
                ListVector({"logcounts": rcounts}), colData=robs, rowData=rvar
            )

        dec = scran.modelGeneVar(sce)
        hvgs = scran.getTopHVGs(dec, n=n_top_genes)
        hvgs = list(hvgs)

        pdata._inplace_subset_var(hvgs)

    def find_nhood_group_markers(
        self,
        data: AnnData | MuData,
        group_to_compare: str | None = None,
        baseline: str | None = None,
        nhood_group_obs: str = "nhood_groups",
        sample_col: str = "sample",
        covariates: Collection[str] | None = None,
        key: str = "rna",
        pseudobulk_function: str = "sum",
        layer: str | None = None,
        target_sum: float = 1e6,
        n_top_genes: int = 7500,
        filter_method: str | None = "scanpy",
        var_names: Collection[str] | None = None,
        de_method: Literal["pydeseq2", "edger"] = "pydeseq2",
        quiet: bool = True,
        alpha: float = 0.05,
        use_eb: bool = False,
        **kwargs,
    ):
        """Perform differential expression analysis on neighborhood groups in a MuData object.

        The MuData object must contain a modality with the name `key`, which is used for pseudobulk aggregation.
        The column `nhood_group_obs` in `mdata[key]` must contain the neighborhood group labels, and `sample_col` must contain the sample labels.
        Neighborhood group labels can be assigned to the single-cell data using ´milo.group_nhoods(...)`, or manually set in `mdata[key].obs[nhood_group_obs]`.
        Neighborhood group labels must be strings or categorical values and are used for pseudobulk aggregation.
        If both `group_to_compare` and `baseline` are given, runs exactly that contrast. Otherwise, runs one‐vs‐rest for every level of `nhood_group_obs`.
        All NAs in mdata[key].obs[nhood_group_obs] are filtered out before running the analysis.
        Therefore, if annotating nhood_group_obs manually, introducing NAs before Milo().group_nhoods(...) is can exclude unwanted neighborhoods from the analysis.

        Parameters
        ----------
        mdata : MuData
            A MuData object containing the data. Must have a modality with the name `key`.
        group_to_compare : Optional[str]
            If provided, runs a single contrast comparing this group to `baseline`.
        baseline : Optional[str]
            The reference group for the contrast. Must be provided if `group_to_compare` is provided.
        nhood_group_obs : str, default="nhood_groups"
            The name of the column in `adata.obs` that contains the neighborhood group labels.
        sample_col : str, default="sample"
            The name of the column in `adata.obs` that contains the sample labels.
        covariates : Collection[str] | None, default=None
            A collection of additional covariates to include in the design formula.
            If None, no additional covariates are used.
        key : str, default="rna"
            The key in `mdata` that corresponds to the modality to be used for pseudobulk aggregation.
        pseudobulk_function : str, default="sum"
            The function to use for pseudobulk aggregation. Can be "sum" or "mean".
        layer : str | None, default=None
            If provided, the layer to use for pseudobulk aggregation. If None, uses the default layer.
        target_sum : float, default=1e6
            The target sum for normalization when using the "scanpy" filter method.
        n_top_genes : int, default=7500
            The number of top variable genes to retain after filtering. Only used if `filter_method` is `scanpy` `scran`.
        filter_method : str | None, default="scanpy"
            The method to use for filtering highly variable genes. Can be "scanpy", "scran", or "filterByExpr".
            If None, no filtering is applied.
        var_names : Collection[str] | None, default=None
            A collection of variable names to restrict the analysis to. If None, all variables are used.
        de_method : Literal["pydeseq2", "statsmodels", "edgeR", "limma"], default="pydeseq2"
            The method to use for differential expression analysis. Can be "pydeseq2", "statsmodels", "edgeR", or "limma".
        quiet : bool, default=True
            If True, suppresses output messages from pydeseq2.
        alpha : float, default=0.05
            The significance threshold for differential expression analysis in pydeseq2.
        use_eb : bool, default=False
            If True, applies empirical Bayes moderation to the results in statsmodels. Not for serious use, but a starting point for limma-like differential testing in pure Python.
        **kwargs : dict
            Additional keyword arguments passed to the filtering methods or differential expression methods.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.group_nhoods(mdata)
            >>> milo.annotate_cells_from_nhoods(mdata)
            >>> milo.find_nhood_group_markers(mdata, group_to_compare="3", baseline="1", nhood_group_obs="nhood_groups")
            >>> milo.find_nhood_group_markers(mdata, nhood_group_obs="nhood_groups")
        """
        func = pseudobulk_function

        # 1) Subset to cells that have a non‐NA group label
        if isinstance(data, AnnData):
            adata = data
        elif isinstance(data, MuData):
            mdata = data
            if key not in mdata.mod_names:
                raise KeyError(f"Modality '{key}' not found in mdata; available keys: {list(mdata.keys())}")
            adata = mdata[key]
        else:
            raise TypeError("data must be an AnnData or MuData object.")

        if nhood_group_obs not in adata.obs.columns:
            raise KeyError(f"Column '{nhood_group_obs}' not found in adata.obs")

        from pandas.api.types import CategoricalDtype

        if not isinstance(adata.obs[nhood_group_obs].dtype, CategoricalDtype):
            adata.obs[nhood_group_obs] = adata.obs[nhood_group_obs].astype("category")

        n_non_na = adata.obs[nhood_group_obs].notna().sum()
        if n_non_na == 0:
            raise ValueError(f"No non‐NA entries found in '{nhood_group_obs}'")

        if sample_col not in adata.obs.columns:
            raise KeyError(f"sample_col '{sample_col}' not in adata.obs")
        for cov in covariates or []:
            if cov not in adata.obs.columns:
                raise KeyError(f"Covariate '{cov}' not found in adata.obs")

        # If you expect “sum” or “mean” you might leave as is; if using a custom layer, check name:
        if pseudobulk_function not in ("sum", "mean"):
            raise KeyError(f"pseudobulk_function '{pseudobulk_function}' is not in 'sum'/'mean'")

        if var_names is not None:
            missing = set(var_names) - set(adata.var_names)
            if missing:
                raise KeyError(f"These var_names are not in adata.var_names: {missing}")

        if group_to_compare is not None or baseline is not None:
            levels = adata.obs[nhood_group_obs].cat.categories.tolist()
            if group_to_compare not in levels:
                raise ValueError(f"group_to_compare '{group_to_compare}' not a level of '{nhood_group_obs}' ({levels})")
            if baseline not in levels:
                raise ValueError(f"baseline '{baseline}' not a level of '{nhood_group_obs}' ({levels})")
            if group_to_compare == baseline:
                raise ValueError("group_to_compare and baseline cannot be the same")

        mask = adata.obs[nhood_group_obs].notna()
        tmp_data = adata[mask]

        # 2) Build the list of categorical variables to aggregate by
        covariates = [] if covariates is None else list(covariates)

        if sample_col in covariates:
            all_variables = [nhood_group_obs] + covariates
        else:
            all_variables = [sample_col, nhood_group_obs] + covariates

        # 3) Pseudobulk aggregation
        pdata = sc.get.aggregate(tmp_data, by=all_variables, func=func, axis=0, layer=layer)
        pdata.X = pdata.layers[func].copy()

        if pdata.obs[nhood_group_obs].nunique() < 2:
            raise ValueError(f"After aggregation, '{nhood_group_obs}' has fewer than 2 groups; DEA cannot proceed.")

        if group_to_compare is None and baseline is None:
            levels_after = pdata.obs[nhood_group_obs].cat.categories.tolist()
            if len(levels_after) < 2:
                raise ValueError(
                    f"Need at least two groups in '{nhood_group_obs}' to run one‐vs‐rest; found {levels_after}"
                )

        # Build the design formula string
        if not covariates:
            base_formula = "~" + nhood_group_obs
        else:
            base_formula = "~" + " + ".join(covariates) + " + " + nhood_group_obs

        if var_names is not None:
            missing = set(var_names) - set(pdata.var_names)
            if missing:
                raise KeyError(f"Some var_names not found in pdata.var_names: {missing}")
            # In‐place subset to exactly var_names:
            pdata._inplace_subset_var(var_names)
        # 2) If no var_names, but n_top_genes is None or zero skip filtering.
        elif not n_top_genes:
            pass
        # 3) var_names is None and n_top_genes is a positive integer
        elif filter_method == "scanpy":
            self._filter_highly_variable_scanpy(pdata, n_top_genes, target_sum)
        elif filter_method == "scran":
            self._filter_highly_variable_scran(pdata, n_top_genes)
        elif filter_method == "filterByExpr":
            import inspect

            sig = inspect.signature(self._filter_by_expr_edger)
            valid_filter_keys = set(sig.parameters)
            filter_kwargs = {}
            for kwargs_key in valid_filter_keys & set(kwargs):
                filter_kwargs[kwargs_key] = kwargs.pop(key)
            if not filter_kwargs:
                filter_kwargs = {"min_expr": 1, "min_total": 10, "min_prop": 0.1}
            if not _is_counts(pdata.X):
                raise ValueError("`pdata.X` appears to be continuous expression, but filterByExpr requires raw counts.")
            self._filter_by_expr_edger(pdata, base_formula, **filter_kwargs)
        else:
            raise ValueError(f"filter_method must be 'scanpy', 'scran' or 'filterByExpr', not '{filter_method}'")

        if de_method == "pydeseq2":
            if not _is_counts(pdata.X):
                raise ValueError("`pdata.X` appears to be raw counts, but this function expects raw counts.")
            return self._run_pydeseq2_contrasts(
                pdata,
                nhood_group_obs=nhood_group_obs,
                formula=base_formula,
                group_to_compare=group_to_compare,
                baseline=baseline,
                alpha=alpha,
                quiet=quiet,
            )

        if de_method == "edger":
            if not _is_counts(pdata.X):
                raise ValueError("`pdata.X` appears to be raw counts, but this function expects raw counts.")
            return self._run_edger_contrasts(
                pdata,
                nhood_group_obs=nhood_group_obs,
                formula=base_formula,
                group_to_compare=group_to_compare,
                baseline=baseline,
            )
        else:
            raise ValueError(f"de_method must be one of 'pydeseq2' or 'edger', not '{de_method}'")

    def plot_heatmap_with_dot_and_colorbar(
        self,
        mean_df: pd.DataFrame,
        logfc_ser: pd.Series | None = None,
        cmap: str = "YlGnBu",
        dot_scale: float = 200.0,
        figsize: tuple[float, float] = (6, 10),
        panel_ratios: tuple[float, float, float] = (5, 0.6, 0.3),
        cbar_tick_count: int = 5,
        show_dot: bool = True,
        legend_on_right: bool = False,
    ) -> plt.Figure:
        """Marker heatmap of mean expression across groups, with optional logFC dots and a colorbar.

        Plot a figure with:
        • Left: heatmap of mean_df (genes × groups), WITHOUT its default colorbar.
        • (Optional) Middle: a single column of dots (size ∝ |logFC|), one per gene.
        • Right: a slim vertical colorbar (ggplot2 style) that applies to the heatmap.
        • (Optional) A size legend for logFC dots, either just to the right of the colorbar
            (legend_on_right=False, the default) or further to the right (legend_on_right=True).

        If show_dot=False, `logfc_ser` may be omitted (and is ignored). If show_dot=True,
        then `logfc_ser` must be provided and must match `mean_df.index`.

        Parameters
        ----------
        mean_df : pandas.DataFrame, shape (n_genes, n_groups)
            Rows = gene names; columns = group labels; values = mean expression.

        logfc_ser : pandas.Series or None, default=None
            If show_dot=True, this Series of length n_genes (indexed by gene names) gives
            the logFC for each gene. If show_dot=False, you may leave this as None.

        cmap : str, default="YlGnBu"
            Colormap for the heatmap and its colorbar.

        dot_scale : float, default=200.0
            Controls the maximum dot area for the largest |logFC| (only used if show_dot=True).

        figsize : tuple (W, H), default=(6, 10)
            Total figure size in inches. Width W is split among panels according to ratios.

        panel_ratios : tuple (r1, r2, r3), default=(5, 0.6, 0.3)
            Relative widths for [heatmap, dot‐column, colorbar] when show_dot=True.
            If show_dot=False, only r1 and r3 are used to split the width.

        cbar_tick_count : int, default=5
            Number of ticks on the vertical colorbar.

        show_dot : bool, default=True
            If True, draw the dot column (requires `logfc_ser`). If False, omit dots and
            only draw [heatmap | colorbar].

        legend_on_right : bool, default=False
            If True, move the “size legend” further to the right of the figure,
            to avoid overlap when the figure is narrow. If False, place it just
            to the right of the colorbar (may overlap if figure is very narrow).

        Returns:
        -------
        fig : matplotlib.figure.Figure

        Examples:
            >>> varnames = (
            >>>     nhood_group_markers_results
            >>>         .query('logFC >= 0.5')
            >>>         .query('adj_PValue <= 0.01')
            >>>         .sort_values("logFC", ascending = False)
            >>>         .variable.to_list()
            >>> )
            >>> mean_df = milo.get_mean_expression(mdata["rna"], "nhood_groups", var_names=varnames)
            >>> logfc_ser = (
            >>>     nhood_group_markers_results
            >>>         .query('logFC >= 0.5')
            >>>         .query('adj_PValue <= 0.01')
            >>>         .set_index("variable")
            >>>         .logFC
            >>> )
            >>> fig = milo.plot_heatmap_with_dot_and_colorbar(
            >>>     mean_df,
            >>>     logfc_ser=logfc_ser,
            >>>     cmap="YlGnBu",
            >>>     dot_scale=200.0,
            >>>     figsize=(2, (1.5, len(logfc_ser)*0.15)),
            >>>     panel_ratios=(5, 0.6, 0.3),
            >>>     cbar_tick_count=5,
            >>>     show_dot=True,
            >>>     legend_on_right=1.3,
            >>> )

        """
        # ────────────────────────────────
        # 1) Validate / align logFC
        # ────────────────────────────────
        if show_dot:
            if logfc_ser is None:
                raise ValueError("`logfc_ser` must be provided when `show_dot=True`.")
            genes = list(mean_df.index)
            lfc_vals = logfc_ser.reindex(index=genes).fillna(0.0).values
            n_genes = len(genes)
        else:
            genes = list(mean_df.index)
            n_genes = len(genes)
            lfc_vals = None

        groups = list(mean_df.columns)

        # ────────────────────────────────
        # 2) Dot‐size scaling (if needed)
        # ────────────────────────────────
        if show_dot:
            max_abs_lfc = np.nanmax(np.abs(lfc_vals))
            if max_abs_lfc == 0 or np.isnan(max_abs_lfc):
                max_abs_lfc = 1.0

        # ────────────────────────────────
        # 3) Heatmap normalization
        # ────────────────────────────────
        vmin = mean_df.values.min()
        vmax = mean_df.values.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)

        # ────────────────────────────────
        # 4) Build a GridSpec
        # ────────────────────────────────
        W, H = figsize
        r1, r2, r3 = panel_ratios

        if show_dot:
            # three panels: [heatmap | dots | colorbar]
            total_ratio = r1 + r2 + r3
            width_ratios = [r1 / total_ratio, r2 / total_ratio, r3 / total_ratio]
            fig = plt.figure(figsize=(W, H))
            gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=width_ratios, wspace=0.02)
            ax_heat = fig.add_subplot(gs[0, 0])
        else:
            # two panels: [heatmap | colorbar]
            total_ratio = r1 + r3
            width_ratios = [r1 / total_ratio, r3 / total_ratio]
            fig = plt.figure(figsize=(W, H))
            gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=width_ratios, wspace=0.02)
            ax_heat = fig.add_subplot(gs[0, 0])

        # ────────────────────────────────
        # 5) Plot heatmap (no default colorbar)
        # ────────────────────────────────
        sns.heatmap(
            mean_df,
            ax=ax_heat,
            cmap=cmap,
            norm=norm,
            cbar=False,
            yticklabels=genes,
            xticklabels=groups,
            linewidths=0.5,
            linecolor="gray",
        )
        ax_heat.set_ylabel("Gene", fontsize=10)
        ax_heat.set_xlabel("Group", fontsize=10)
        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax_heat.get_yticklabels(), rotation=0, fontsize=6)

        # ────────────────────────────────
        # 6) Dot panel (if requested)
        # ────────────────────────────────
        if show_dot:
            ax_dot = fig.add_subplot(gs[0, 1])
            for i, val in enumerate(lfc_vals):
                if not np.isnan(val) and val != 0.0:
                    area = (abs(val) / max_abs_lfc) * dot_scale
                    ax_dot.scatter(0, i, s=area, color="black", alpha=0.8, edgecolors="none")
            ax_dot.set_xlim(-0.5, 0.5)
            ax_dot.set_ylim(n_genes - 0.5, -0.5)
            ax_dot.set_xticks([])
            ax_dot.set_yticks([])
            ax_dot.set_title("logFC", pad=10, fontdict={"fontsize": 7})
            ax_cbar = fig.add_subplot(gs[0, 2])
        else:
            ax_cbar = fig.add_subplot(gs[0, 1])

        # ────────────────────────────────
        # 7) Draw vertical colorbar for heatmap
        # ────────────────────────────────
        smap = ScalarMappable(norm=norm, cmap=cmap_obj)
        smap.set_array([])

        cbar = fig.colorbar(
            smap, cax=ax_cbar, orientation="vertical", ticks=np.linspace(vmin, vmax, num=cbar_tick_count)
        )
        cbar.ax.tick_params(labelsize=8, length=4, width=1)
        cbar.ax.set_title("Mean\nExpr.", fontsize=8, pad=6)
        cbar.outline.set_linewidth(0.5)

        # ────────────────────────────────
        # 8) Add a size‐legend for the dot‐column (optional)
        # ────────────────────────────────
        if show_dot:
            # Choose three reference |logFC| values: max, ½ max, ¼ max
            ref_vals = np.array([max_abs_lfc, 0.5 * max_abs_lfc, 0.25 * max_abs_lfc])
            legend_handles = []
            legend_labels = []
            for rv in ref_vals:
                sz = (rv / max_abs_lfc) * dot_scale
                handle = ax_dot.scatter(0, 0, s=sz, color="black", alpha=0.8, edgecolors="none")
                legend_handles.append(handle)
                legend_labels.append(f"|logFC| = {rv:.2f}")

            # Determine bounding box based on legend_on_right flag
            bbox_x = (1.2 if isinstance(legend_on_right, bool) else legend_on_right) if legend_on_right else 1.02

            fig.legend(
                legend_handles,
                legend_labels,
                title="Dot size legend",
                loc="center left",
                bbox_to_anchor=(bbox_x, 0.5),
                frameon=False,
                fontsize=7,
                title_fontsize=8,
                handletextpad=0.5,
                labelspacing=0.6,
            )

        plt.tight_layout()
        return fig
