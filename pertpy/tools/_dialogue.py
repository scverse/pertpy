from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from sparsecca import multicca_pmd


class Dialogue:
    """Python implementation of DIALOGUE"""

    def _get_pseudobulks(self, adata: AnnData, groupby: str) -> pd.DataFrame:
        """Return cell-averaged data by groupby.

        Copied from `https://github.com/schillerlab/sc-toolbox/blob/397e80dc5e8fb8017b75f6c3fa634a1e1213d484/sc_toolbox/tools/__init__.py#L458`

        Args:
            groupby: The key to groupby for pseudobulks

        Returns:
            A Pandas DataFrame of pseudobulk counts
        """
        pseudobulk = {"Genes": adata.var_names.values}

        for i in adata.obs.loc[:, groupby].cat.categories:
            temp = adata.obs.loc[:, groupby] == i
            pseudobulk[i] = adata[temp].X.median(axis=0).A1

        pseudobulk = pd.DataFrame(pseudobulk).set_index("Genes")

        return pseudobulk

    def _pseudobulk_pca(self, adata: AnnData, groupby: str, n_components: int = 50) -> pd.DataFrame:
        """Return cell-averaged PCA components.

        TODO: consider merging with `get_pseudobulks`
        TODO: DIALOGUE recommends running PCA on each cell type separately before running PMD - this should be implemented as an option here.

        Args:
            groupby: The key to groupby for pseudobulks
            n_components: The number of PCA components

        Returns:
            A pseudobulk of PCA components.
        """
        aggr = {}

        for i in adata.obs.loc[:, groupby].cat.categories:
            temp = adata.obs.loc[:, groupby] == i
            aggr[i] = adata[temp].obsm["X_pca"][:, :n_components].mean(axis=0)

        aggr = pd.DataFrame(aggr)

        return aggr

    def _scale_data(self, pseudobulks: pd.DataFrame, mimic_dialogue: bool = True) -> np.ndarray:
        """Row-wise mean center and scale by the standard deviation.

        TODO: the `scale` function we implemented to match the R `scale` fn should already contain this functionality.

        Args:
            pseudobulks: The pseudobulk PCA components.
            mimic_dialogue: Whether to mimic DIALOGUE behavior or not.

        Returns:
            The scaled count matrix.

        """
        # DIALOGUE doesn't scale the data before passing to multicca, unlike what is recommended by sparsecca.
        # However, performing this scaling _does_ increase overall correlation of the end result
        # WHEN SAMPLE ORDER AND DIALOGUE2+3 PROCESSING IS IGNORED.
        if mimic_dialogue:
            return pseudobulks.to_numpy()
        else:
            return ((pseudobulks - pseudobulks.mean()) / pseudobulks.std()).to_numpy()

    def _concat_adata_mcp_scores(
        self, adata: AnnData, ct_subs: dict[str, AnnData], mcp_scores: dict[str, np.ndarray], celltype_key: str
    ) -> AnnData:
        """Concatenates the AnnData object with the mcp scores.

        Args:
            adata: The AnnData object to append mcp scores to.
            mcp_scores: The MCP scores dictionary.

        Returns:
            AnnData object with concatenated MCP scores in obs
        """

        def __concat_obs(adata: AnnData, mcp_df: pd.DataFrame) -> AnnData:
            new_obs = pd.concat([adata.obs, mcp_df.set_index(adata.obs.index)], axis=1)
            adata.obs = new_obs

            return adata

        ad_mcp = {
            ct: __concat_obs(ct_subs[ct], pd.DataFrame(mcp_scores[ct]))
            for ct in adata.obs[celltype_key].cat.categories
        }

        adata = ad.concat(ad_mcp.values())

        return adata

    def load(
        self,
        adata: AnnData,
        groupby: str,
        celltype_key: str,
        ct_order: list[str],
        agg_pca: bool = True,
        mimic_dialogue: bool = True,
    ) -> tuple[list, dict]:
        """Separates cell into AnnDatas by celltype_key and creates the multifactor PMD input.

        Mimics DIALOGUE's `make.cell.types` and the pre-processing that occurs in DIALOGUE1.

        Args:
            adata: AnnData object to loa
            groupby: The key to aggregate the pseudobulks for.
            celltype_key: The key of the cell type column.
            ct_order: The order of cell types
            agg_pca: Whether to aggregate pseudobulks with PCA or not (default: True).
            mimic_dialogue: Whether to mimic DIALOGUE behavior or not (default: True).

        Returns:
            A celltype_label:array dictionary.
        """
        ct_subs = {ct: adata[adata.obs[celltype_key] == ct].copy() for ct in ct_order}
        fn = self._pseudobulk_pca if agg_pca else self._get_pseudobulks
        ct_aggr = {ct: fn(ad, groupby) for ct, ad in ct_subs.items()}  # type: ignore

        # TODO: implement check (as in https://github.com/livnatje/DIALOGUE/blob/55da9be0a9bf2fcd360d9e11f63e30d041ec4318/R/DIALOGUE.main.R#L114-L119)
        # that there are at least 5 share samples here

        # TODO: https://github.com/livnatje/DIALOGUE/blob/55da9be0a9bf2fcd360d9e11f63e30d041ec4318/R/DIALOGUE.main.R#L121-L131
        ct_preprocess = {ct: self._scale_data(ad, mimic_dialogue=mimic_dialogue).T for ct, ad in ct_aggr.items()}

        mcca_in = [ct_preprocess[ct] for ct in ct_order]

        return mcca_in, ct_subs

    def calculate_multifactor_PMD(  # noqa: N802
        self,
        adata: AnnData,
        groupby: str,
        celltype_key: str,
        n_components: int = 3,
        penalties: list[int] = None,
        ct_order: list[str] = None,
        agg_pca: bool = True,
        mimic_dialogue: bool = True,
    ) -> tuple[AnnData, dict[str, np.ndarray], dict[Any, Any], dict[Any, Any]]:
        """Runs multifactor PMD.

        Currently mimics DIALOGUE1.

        Args:
            adata: AnnData object to calculate PMD for.
            groupby: Key to use for pseudobulk determination.
            celltype_key:  Cell type column key.
            n_components: Number of PMD components.
            penalties: PMD penalties.
            ct_order: The order of cell types.
            agg_pca: Whether to calculate cell-averaged PCA components.
            mimic_dialogue: Whether to mimic DIALOGUE as close as possible

        Returns:
            MCP scores  # TODO this requires more detail
        """
        # IMPORTANT NOTE: the order in which matrices are passed to multicca matters. As such,
        # it is important here that to obtain the same result as in R, we pass the matrices in
        # in the same order.
        if ct_order is not None:
            cell_types = ct_order
        else:
            ct_order = cell_types = adata.obs[celltype_key].astype("category").cat.categories

        mcca_in, ct_subs = self.load(
            adata, groupby, celltype_key, ct_order=cell_types, agg_pca=agg_pca, mimic_dialogue=mimic_dialogue
        )

        # TODO: awaiting implementation of MultiCCA.permute in order to determine optimal penalty
        if penalties is None:
            penalties = [10 for x in cell_types]  # random number for now
        else:
            penalties = penalties

        ws, _ = multicca_pmd(mcca_in, penalties, K=n_components, standardize=True, niter=100, mimic_R=mimic_dialogue)
        ws_dict = {ct: ws[i] for i, ct in enumerate(ct_order)}

        mcp_scores = {
            ct: ct_subs[ct].obsm["X_pca"][:, :50] @ ws[i] for i, ct in enumerate(cell_types)  # TODO change from 50
        }

        # TODO: output format needs some cleanup, even though each MCP score is matched to one cell, it's not at all
        # matched at this point in the function and requires references to internals that shouldn't need exposing (ct_subs)

        adata = self._concat_adata_mcp_scores(adata, ct_subs=ct_subs, mcp_scores=mcp_scores, celltype_key=celltype_key)

        return adata, mcp_scores, ws_dict, ct_subs
