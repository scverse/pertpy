from sklearn.decomposition import KernelPCA


def kernel_pca(adata, n_comps=None, kernel="linear", copy=False, return_transformer=False, **kwargs):
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
