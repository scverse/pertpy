from __future__ import annotations

from anndata import AnnData
from sklearn.decomposition import KernelPCA


def kernel_pca(
    adata: AnnData,
    n_comps: int = 50,
    kernel: str = "linear",
    copy: bool = False,
    return_transformer: bool = False,
    **kwargs,
):
    """Compute kernel PCA on `adata.X`.

    Compute Kernel Principal component analysis (KPCA) using sklearn.
    See also :class:`~sklearn.decomposition.KernelPCA`.

    Args:
        adata: The annotated data object.
        n_comps: Number of components. If None, all non-zero components are kept.
        kernel: Kernel used for PCA.
        copy: Determines whether a copy of the `adata` is returned.
        return_transformer: Determines if the `KernelPCA` transformer is returned.
        **kwargs: Additional arguments for the `KernelPCA` transformer.

    Returns:
        If `copy=True`, returns the copy of `adata` with kernel pca in `.obsm["X_kpca"]`.
        Otherwise writes kernel pca directly to `.obsm["X_kpca"]` of the provided `adata`.
        If `return_transformer=True`, returns also the fitted `KernelPCA` transformer.
    """
    if copy:
        adata = adata.copy()
    transformer = KernelPCA(n_components=n_comps, kernel=kernel, **kwargs)
    X_kpca = transformer.fit_transform(adata.X)
    adata.obsm["X_kpca"] = X_kpca

    if copy:
        if return_transformer:
            return adata, transformer
        else:
            return adata
    else:
        if return_transformer:
            return transformer
