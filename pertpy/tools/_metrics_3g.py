from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import pynndescent
import scanpy as sc
from scipy.sparse import issparse
from scipy.sparse import vstack as sp_vstack
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression


def compare_de(X: np.ndarray, Y: np.ndarray, C: np.ndarray, shared_top: int = 100, **kwargs) -> dict:
    """X - real, Y - simulated, C - control."""
    n_vars = X.shape[1]
    assert n_vars == Y.shape[1] == C.shape[1]

    shared_top = min(shared_top, n_vars)
    vars_ranks = np.arange(1, n_vars + 1)

    adatas_xy = {}
    adatas_xy["x"] = ad.AnnData(X, obs={"label": "comp"})
    adatas_xy["y"] = ad.AnnData(Y, obs={"label": "comp"})
    adata_c = ad.AnnData(C, obs={"label": "ctrl"})

    results = pd.DataFrame(index=adata_c.var_names)
    top_names = []
    for group in ("x", "y"):
        adata_joint = ad.concat((adatas_xy[group], adata_c), index_unique="-")

        sc.pp.log1p(adata_joint)
        sc.tl.rank_genes_groups(adata_joint, groupby="label", reference="ctrl", key_added="de", **kwargs)

        srt_idx = np.argsort(adata_joint.uns["de"]["names"]["comp"])
        results[f"scores_{group}"] = adata_joint.uns["de"]["scores"]["comp"][srt_idx]
        results[f"pvals_adj_{group}"] = adata_joint.uns["de"]["pvals_adj"]["comp"][srt_idx]
        # needed to avoid checking rankby_abs
        results[f"ranks_{group}"] = vars_ranks[srt_idx]

        top_names.append(adata_joint.uns["de"]["names"]["comp"][:shared_top])

    metrics = {}
    metrics["shared_top_genes"] = len(set(top_names[0]).intersection(top_names[1])) / shared_top
    metrics["scores_corr"] = results["scores_x"].corr(results["scores_y"], method="pearson")
    metrics["pvals_adj_corr"] = results["pvals_adj_x"].corr(results["pvals_adj_y"], method="pearson")
    metrics["scores_ranks_corr"] = results["ranks_x"].corr(results["ranks_y"], method="spearman")

    return metrics


def compare_class(
    X: np.ndarray, Y: np.ndarray, C: np.ndarray, clf: Optional[ClassifierMixin] = None, pca: bool = False
) -> float:
    """X - real, Y - simulated, C - control."""
    assert X.shape[1] == Y.shape[1] == C.shape[1]

    if clf is None:
        clf = LogisticRegression()

    n_x = len(X)
    n_xc = n_x + len(C)

    data = sp_vstack((X, C)) if issparse(X) else np.vstack((X, C))

    labels = np.full(n_xc, "ctrl")
    labels[:n_x] = "comp"
    clf.fit(data, labels)

    norm_score = clf.score(Y, np.full(len(Y), "comp")) / clf.score(X, labels[:n_x])
    norm_score = min(1.0, norm_score)

    return norm_score


def compare_knn(
    X: np.ndarray, Y: np.ndarray, C: Optional[np.ndarray] = None, n_neighbors: int = 20, use_Y_knn: bool = False
) -> tuple:
    """X - real, Y - simulated, C - control."""
    assert X.shape[1] == Y.shape[1]
    if C is not None:
        assert X.shape[1] == C.shape[1]

    n_y = len(Y)

    if C is None:
        index_data = sp_vstack((Y, X)) if issparse(X) else np.vstack((Y, X))
    else:
        datas = (Y, X, C) if use_Y_knn else (X, C)
        index_data = sp_vstack(datas) if issparse(X) else np.vstack(datas)

    y_in_index = use_Y_knn or C is None
    c_in_index = C is not None
    labels = np.full(len(index_data), "comp")
    if y_in_index:
        labels[:n_y] = "siml"
    if c_in_index:
        labels[-len(C) :] = "ctrl"

    index = pynndescent.NNDescent(index_data, n_neighbors=max(50, n_neighbors))
    indices = index.query(Y, k=n_neighbors)[0]

    uq_counts = np.unique(labels[indices], return_counts=True)

    return uq_counts[0], uq_counts[1] / uq_counts[1].sum()
