from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pynndescent
from anndata import AnnData
from scipy.sparse import issparse
from scipy.sparse import vstack as sp_vstack
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compare_de(
    adata: AnnData | None = None,
    de_key1: str = None,
    de_key2: str = None,
    de_df1: pd.DataFrame | None = None,
    de_df2: pd.DataFrame | None = None,
    shared_top: int = 100,
) -> dict:
    """Compare two differential expression analyses.

    Compare two sets of DE results and evaluate the similarity by the overlap of top DEG and
    the correlation of their scores and adjusted p-values.

    Args:
        adata: AnnData object containing DE results in `uns`. Required if `de_key1` and `de_key2` are used.
        de_key1: Key for DE results in `adata.uns`, e.g., output of `tl.rank_genes_groups`.
        de_key2: Another key for DE results in `adata.uns`, e.g., output of `tl.rank_genes_groups`.
        de_df1: DataFrame containing DE results, e.g. output from pertpy differential gene expression interface.
        de_df2: DataFrame containing DE results, e.g. output from pertpy differential gene expression interface.
        shared_top: The number of top DEG to compute the proportion of their intersection.

    """
    if (de_key1 or de_key2) and (de_df1 is not None or de_df2 is not None):
        raise ValueError(
            "Please provide either both `de_key1` and `de_key2` with `adata`, or `de_df1` and `de_df2`, but not both."
        )

    if de_df1 is None and de_df2 is None:  # use keys
        if not de_key1 or not de_key2:
            raise ValueError("Both `de_key1` and `de_key2` must be provided together if using `adata`.")

    else:  # use dfs
        if de_df1 is None or de_df2 is None:
            raise ValueError("Both `de_df1` and `de_df2` must be provided together if using dataframes.")

    if de_key1:
        if not adata:
            raise ValueError("`adata` should be provided with `de_key1` and `de_key2`. ")
        assert all(
            k in adata.uns for k in [de_key1, de_key2]
        ), "Provided `de_key1` and `de_key2` must exist in `adata.uns`."
        vars = adata.var_names

    if de_df1 is not None:
        for df in (de_df1, de_df2):
            if not {"variable", "log_fc", "adj_p_value"}.issubset(df.columns):
                raise ValueError("Each DataFrame must contain columns: 'variable', 'log_fc', and 'adj_p_value'.")

        assert set(de_df1["variable"]) == set(de_df2["variable"]), "Variables in both dataframes must match."
        vars = de_df1["variable"].sort_values()

    shared_top = min(shared_top, len(vars))
    vars_ranks = np.arange(1, len(vars) + 1)
    results = pd.DataFrame(index=vars)
    top_names = []

    if de_key1 and de_key2:
        for i, k in enumerate([de_key1, de_key2]):
            label = adata.uns[k]["names"].dtype.names[0]
            srt_idx = np.argsort(adata.uns[k]["names"][label])
            results[f"scores_{i}"] = adata.uns[k]["scores"][label][srt_idx]
            results[f"pvals_adj_{i}"] = adata.uns[k]["pvals_adj"][label][srt_idx]
            results[f"ranks_{i}"] = vars_ranks[srt_idx]
            top_names.append(adata.uns[k]["names"][label][:shared_top])
    else:
        for i, df in enumerate([de_df1, de_df2]):
            srt_idx = np.argsort(df["variable"])
            results[f"scores_{i}"] = df["log_fc"].values[srt_idx]
            results[f"pvals_adj_{i}"] = df["adj_p_value"].values[srt_idx]
            results[f"ranks_{i}"] = vars_ranks[srt_idx]
            top_names.append(df["variable"][:shared_top])

    metrics = {}
    metrics["shared_top_genes"] = len(set(top_names[0]).intersection(top_names[1])) / shared_top
    metrics["scores_corr"] = results["scores_0"].corr(results["scores_1"], method="pearson")
    metrics["pvals_adj_corr"] = results["pvals_adj_0"].corr(results["pvals_adj_1"], method="pearson")
    metrics["scores_ranks_corr"] = results["ranks_0"].corr(results["ranks_1"], method="spearman")

    return metrics


def compare_class(X: np.ndarray, Y: np.ndarray, C: np.ndarray, clf: ClassifierMixin | None = None) -> float:
    """Compare classification accuracy between real and simulated perturbations.

    Trains a classifier on the real perturbation data + the control data and reports a normalized
    classification accuracy on the simulated perturbation.

    Args:
        X: Real perturbed data.
        Y: Simulated perturbed data.
        C: Control data
        clf: sklearn classifier to use, `sklearn.linear_model.LogisticRegression` if not provided.
    """
    assert X.shape[1] == Y.shape[1] == C.shape[1]

    if clf is None:
        clf = LogisticRegression()

    n_x = X.shape[0]
    n_xc = n_x + C.shape[0]

    data = sp_vstack((X, C)) if issparse(X) else np.vstack((X, C))

    labels = np.full(n_xc, "ctrl")
    labels[:n_x] = "comp"
    clf.fit(data, labels)

    norm_score = clf.score(Y, np.full(Y.shape[0], "comp")) / clf.score(X, labels[:n_x])
    norm_score = min(1.0, norm_score)

    return norm_score


def compare_knn(
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray | None = None,
    n_neighbors: int = 20,
    use_Y_knn: bool = False,
    random_state: int = 0,
    n_jobs: int = 1,
) -> dict[str, float]:
    """Calculate proportions of real perturbed and control data points for simulated data.

    Computes proportions of real perturbed (if provided), control and simulated (if `use_Y_knn=True`)
    data points for simulated data. If control (`C`) is not provided, builds the knn graph from
    real perturbed + simulated perturbed.

    Args:
        X: Real perturbed data.
        Y: Simulated perturbed data.
        C: Control data.
        use_Y_knn: Include simulted perturbed data (`Y`) into the knn graph. Only valid when
            control (`C`) is provided.
    """
    assert X.shape[1] == Y.shape[1]
    if C is not None:
        assert X.shape[1] == C.shape[1]

    n_y = Y.shape[0]

    if C is None:
        index_data = sp_vstack((Y, X)) if issparse(X) else np.vstack((Y, X))
    else:
        datas = (Y, X, C) if use_Y_knn else (X, C)
        index_data = sp_vstack(datas) if issparse(X) else np.vstack(datas)

    y_in_index = use_Y_knn or C is None
    c_in_index = C is not None
    label_groups = ["comp"]
    labels: NDArray[np.str_] = np.full(index_data.shape[0], "comp")
    if y_in_index:
        labels[:n_y] = "siml"
        label_groups.append("siml")
    if c_in_index:
        labels[-C.shape[0] :] = "ctrl"
        label_groups.append("ctrl")

    index = pynndescent.NNDescent(
        index_data,
        n_neighbors=max(50, n_neighbors),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    indices = index.query(Y, k=n_neighbors)[0]

    uq, uq_counts = np.unique(labels[indices], return_counts=True)
    uq_counts_norm = uq_counts / uq_counts.sum()
    counts = dict(zip(label_groups, [0.0] * len(label_groups), strict=False))
    for group, count_norm in zip(uq, uq_counts_norm, strict=False):
        counts[group] = count_norm

    return counts
