import pandas as pd
import pertpy as pt
import pytest
import scanpy as sc

# This is not a proper test!
# We are only testing a few functions to ensure that at least these run
# The pipeline is obtained from https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/dialogue.html


@pytest.mark.slow
def test_dialogue_pipeline():
    adata = pt.dt.dialogue_example()

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    isecs = pd.crosstab(adata.obs["cell.subtypes"], adata.obs["sample"])
    adata = adata[adata.obs["cell.subtypes"] != "CD8+ IL17+"]
    isecs = pd.crosstab(adata.obs["cell.subtypes"], adata.obs["sample"])

    keep_pts = list(isecs.loc[:, (isecs > 3).sum(axis=0) == isecs.shape[0]].columns.values)
    adata = adata[adata.obs["sample"].isin(keep_pts), :].copy()

    dl = pt.tl.Dialogue(
        sample_id="sample",
        celltype_key="cell.subtypes",
        n_counts_key="nCount_RNA",
        n_mpcs=3,
    )

    adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)

    dl.test_association(adata, "path_str")

    dl.get_extrema_MCP_genes(ct_subs)
