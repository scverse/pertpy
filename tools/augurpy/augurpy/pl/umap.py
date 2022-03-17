import scanpy as sc
from anndata import AnnData


def umap(adata: AnnData) -> AnnData:
    """Plot UMAP representation of anndata with augur_score labeling.

    Args:
        adata: AnnData result after running `predict()`

    Returns:
        AnnData object the UMAP is based on.
    """
    try:
        sc.pl.umap(adata=adata, color="augur_score")
    except KeyError:
        print(
            "[Bold yellow]Missing UMAP in obsm. Calculating UMAP using default pp.neighbors() and tl.umap() from scanpy."
        )
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)

    return adata
