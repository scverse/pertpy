from importlib.util import find_spec

import numpy as np
import pandas as pd
import pertpy as pt
import pytest
import scanpy as sc
from mudata import MuData

### NEW IMPORTS
from pertpy.utils._lazy_r_namespace import lazy_import_r_env
###


@pytest.fixture(params=["edger", "pydeseq2"])
def solver(request):
    solver_name = request.param

    if solver_name == "edger":
        try:
            from rpy2.robjects.packages import importr

            importr("edgeR")
        except Exception:  # noqa: BLE001
            pytest.skip("Required R package 'edgeR' not available")

    elif solver_name == "pydeseq2" and find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")

    return solver_name


@pytest.fixture
def adata():
    adata = sc.datasets.pbmc68k_reduced()
    return adata


@pytest.fixture
def milo():
    milo = pt.tl.Milo()
    return milo


def test_load(adata, milo):
    mdata = milo.load(adata)
    assert isinstance(mdata, MuData)
    assert "rna" in mdata.mod


def test_make_nhoods_number(adata, milo):
    adata = adata.copy()
    p = 0.1
    adata = milo.make_nhoods(adata, prop=p, copy=True)
    assert adata.obsm["nhoods"].shape[1] <= int(np.round(adata.n_obs * p))


def test_make_nhoods_missing_connectivities(adata, milo):
    adata = adata.copy()
    p = 0.1
    del adata.obsp["connectivities"]
    with pytest.raises(KeyError):
        adata = milo.make_nhoods(adata, prop=p)


def test_make_nhoods_sizes(adata, milo):
    adata = adata.copy()
    milo.make_nhoods(adata)
    knn_graph = adata.obsp["connectivities"]
    knn_graph[knn_graph != 0] = 1
    assert knn_graph.sum(0).min() <= adata.obsm["nhoods"].sum(0).min()


def test_make_nhoods_neighbors_key(adata, milo):
    adata = adata.copy()
    k = adata.uns["neighbors"]["params"]["n_neighbors"]
    test_k = 5
    sc.pp.neighbors(adata, n_neighbors=test_k, key_added="test")
    milo.make_nhoods(adata, neighbors_key="test")
    smallest_size = adata.obsm["nhoods"].toarray().sum(0).min()
    assert test_k < k
    assert smallest_size < k


def test_count_nhoods_sample_values(adata, milo):
    adata = adata.copy()
    milo.make_nhoods(adata)
    # Extract cells of one nhood
    nh = 1
    sample_col = "phase"
    milo_mdata = milo.count_nhoods(adata, sample_col=sample_col)
    nh_cells = adata.obsm["nhoods"][:, nh].nonzero()[0]

    # Value count the sample composition
    top_a = adata.obs.iloc[nh_cells].value_counts(sample_col).values.ravel()

    # Check it matches the one calculated
    sample_adata = milo_mdata["milo"]
    df = pd.DataFrame(sample_adata.X.T[nh, :].toarray()).T
    df.index = sample_adata.obs_names
    top_b = df.sort_values(0, ascending=False).values.ravel()
    assert all((top_b - top_a) == 0), 'The counts for samples in milo_mdata["milo"] does not match'


def test_count_nhoods_missing_nhoods(adata, milo):
    adata = adata.copy()
    milo.make_nhoods(adata)
    sample_col = "phase"
    del adata.obsm["nhoods"]
    with pytest.raises(KeyError):
        _ = milo.count_nhoods(adata, sample_col=sample_col)


def test_count_nhoods_sample_order(adata, milo):
    adata = adata.copy()
    milo.make_nhoods(adata)
    # Extract cells of one nhood
    nh = 1
    sample_col = "phase"
    milo_mdata = milo.count_nhoods(adata, sample_col=sample_col)
    nh_cells = adata.obsm["nhoods"][:, nh].nonzero()[0]

    # Value count the sample composition
    top_a = adata.obs.iloc[nh_cells].value_counts(sample_col).index[0]

    # Check it matches the one calculated
    sample_adata = milo_mdata["milo"]
    df = pd.DataFrame(sample_adata.X.T[nh, :].toarray()).T
    df.index = sample_adata.obs_names
    top_b = df.sort_values(0, ascending=False).index[0]

    assert top_a == top_b, 'The order of samples in milo_mdata["milo"] does not match'


@pytest.fixture
def da_nhoods_mdata(adata, milo):
    adata = adata.copy()
    milo.make_nhoods(adata)

    # Simulate experimental condition
    rng = np.random.default_rng(seed=42)
    adata.obs["condition"] = rng.choice(["ConditionA", "ConditionB"], size=adata.n_obs, p=[0.5, 0.5])
    # we simulate differential abundance in NK cells
    DA_cells = adata.obs["louvain"] == "1"
    adata.obs.loc[DA_cells, "condition"] = rng.choice(["ConditionA", "ConditionB"], size=sum(DA_cells), p=[0.2, 0.8])

    # Simulate replicates
    adata.obs["replicate"] = rng.choice(["R1", "R2", "R3"], size=adata.n_obs)
    adata.obs["sample"] = adata.obs["replicate"] + adata.obs["condition"]
    milo_mdata = milo.count_nhoods(adata, sample_col="sample")
    return milo_mdata


def test_da_nhoods_pvalues_both_solvers(da_nhoods_mdata, milo, solver):
    mdata = da_nhoods_mdata.copy()
    milo.da_nhoods(mdata, design="~condition", solver=solver)
    sample_adata = mdata["milo"].copy()
    min_p, max_p = sample_adata.var["PValue"].min(), sample_adata.var["PValue"].max()
    assert (min_p >= 0) & (max_p <= 1), "P-values are not between 0 and 1"


def test_da_nhoods_fdr_both_solvers(da_nhoods_mdata, milo, solver):
    mdata = da_nhoods_mdata.copy()
    milo.da_nhoods(mdata, design="~condition", solver=solver)
    sample_adata = mdata["milo"].copy()
    assert np.all(np.round(sample_adata.var["PValue"], 10) <= np.round(sample_adata.var["SpatialFDR"], 10)), (
        "FDR is higher than uncorrected P-values"
    )


def test_da_nhoods_missing_samples(adata, milo):
    with pytest.raises(KeyError):
        milo.da_nhoods(adata, design="~condition")


def test_da_nhoods_missing_covariate(da_nhoods_mdata, milo):
    mdata = da_nhoods_mdata.copy()
    with pytest.raises(KeyError):
        milo.da_nhoods(mdata, design="~ciaone")


def test_da_nhoods_non_unique_covariate(da_nhoods_mdata, milo):
    mdata = da_nhoods_mdata.copy()
    with pytest.raises(AssertionError):
        milo.da_nhoods(mdata, design="~phase")


def test_da_nhoods_pvalues(da_nhoods_mdata, milo):
    mdata = da_nhoods_mdata.copy()
    milo.da_nhoods(mdata, design="~condition")
    sample_adata = mdata["milo"].copy()
    min_p, max_p = sample_adata.var["PValue"].min(), sample_adata.var["PValue"].max()
    assert (min_p >= 0) & (max_p <= 1), "P-values are not between 0 and 1"


def test_da_nhoods_fdr(da_nhoods_mdata, milo):
    mdata = da_nhoods_mdata.copy()
    milo.da_nhoods(mdata, design="~condition")
    sample_adata = mdata["milo"].copy()
    assert np.all(np.round(sample_adata.var["PValue"], 10) <= np.round(sample_adata.var["SpatialFDR"], 10)), (
        "FDR is higher than uncorrected P-values"
    )


def test_da_nhoods_default_contrast(da_nhoods_mdata, milo):
    mdata = da_nhoods_mdata.copy()
    adata = mdata["rna"].copy()
    adata.obs["condition"] = (
        adata.obs["condition"].astype("category").cat.reorder_categories(["ConditionA", "ConditionB"])
    )
    milo.da_nhoods(mdata, design="~condition")
    default_results = mdata["milo"].var.copy()
    milo.da_nhoods(mdata, design="~condition", model_contrasts="conditionConditionB-conditionConditionA")
    contr_results = mdata["milo"].var.copy()

    assert np.corrcoef(contr_results["SpatialFDR"], default_results["SpatialFDR"])[0, 1] > 0.99
    assert np.corrcoef(contr_results["logFC"], default_results["logFC"])[0, 1] > 0.99


@pytest.fixture
def annotate_nhoods_mdata(adata, milo):
    adata = adata.copy()
    milo.make_nhoods(adata)

    # Simulate experimental condition
    rng = np.random.default_rng(seed=42)
    adata.obs["condition"] = rng.choice(["ConditionA", "ConditionB"], size=adata.n_obs, p=[0.5, 0.5])
    # we simulate differential abundance in NK cells
    DA_cells = adata.obs["louvain"] == "1"
    adata.obs.loc[DA_cells, "condition"] = rng.choice(["ConditionA", "ConditionB"], size=sum(DA_cells), p=[0.2, 0.8])

    # Simulate replicates
    adata.obs["replicate"] = rng.choice(["R1", "R2", "R3"], size=adata.n_obs)
    adata.obs["sample"] = adata.obs["replicate"] + adata.obs["condition"]
    milo_mdata = milo.count_nhoods(adata, sample_col="sample")
    return milo_mdata


def test_annotate_nhoods_missing_samples(annotate_nhoods_mdata, milo):
    mdata = annotate_nhoods_mdata.copy()
    del mdata.mod["milo"]
    with pytest.raises(ValueError):
        milo.annotate_nhoods_continuous(mdata, anno_col="S_score")


def test_annotate_nhoods_continuous_mean_range(annotate_nhoods_mdata, milo):
    mdata = annotate_nhoods_mdata.copy()
    milo.annotate_nhoods_continuous(mdata, anno_col="S_score")
    assert mdata["milo"].var["nhood_S_score"].max() < mdata["rna"].obs["S_score"].max()
    assert mdata["milo"].var["nhood_S_score"].min() > mdata["rna"].obs["S_score"].min()


def test_annotate_nhoods_continuous_correct_mean(annotate_nhoods_mdata, milo):
    mdata = annotate_nhoods_mdata.copy()
    milo.annotate_nhoods_continuous(mdata, anno_col="S_score")
    rng = np.random.default_rng(seed=42)
    i = rng.choice(np.arange(mdata["milo"].n_obs))
    mean_val_nhood = mdata["rna"].obs[mdata["rna"].obsm["nhoods"][:, i].toarray() == 1]["S_score"].mean()
    assert mdata["milo"].var["nhood_S_score"].iloc[i] == pytest.approx(mean_val_nhood, 0.0001)


def test_annotate_nhoods_annotation_frac_range(annotate_nhoods_mdata, milo):
    mdata = annotate_nhoods_mdata.copy()
    milo.annotate_nhoods(mdata, anno_col="louvain")
    assert mdata["milo"].var["nhood_annotation_frac"].max() <= 1.0
    assert mdata["milo"].var["nhood_annotation_frac"].min() >= 0.0


def test_annotate_nhoods_cont_gives_error(annotate_nhoods_mdata, milo):
    mdata = annotate_nhoods_mdata.copy()
    with pytest.raises(ValueError):
        milo.annotate_nhoods(mdata, anno_col="S_score")


@pytest.fixture
def add_nhood_expression_mdata(milo):
    adata = sc.datasets.pbmc3k()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    milo.make_nhoods(adata)

    # Simulate experimental condition
    rng = np.random.default_rng(seed=42)
    adata.obs["condition"] = rng.choice(["ConditionA", "ConditionB"], size=adata.n_obs, p=[0.2, 0.8])
    # we simulate differential abundance in NK cells
    DA_cells = adata.obs["leiden"] == "1"
    adata.obs.loc[DA_cells, "condition"] = rng.choice(["ConditionA", "ConditionB"], size=sum(DA_cells), p=[0.2, 0.8])

    # Simulate replicates
    adata.obs["replicate"] = rng.choice(["R1", "R2", "R3"], size=adata.n_obs)
    adata.obs["sample"] = adata.obs["replicate"] + adata.obs["condition"]
    milo_mdata = milo.count_nhoods(adata, sample_col="sample")

    return milo_mdata


def test_add_nhood_expression_nhood_mean_range(add_nhood_expression_mdata, milo):
    mdata = add_nhood_expression_mdata.copy()
    milo.add_nhood_expression(mdata)

    assert mdata["milo"].varm["expr"].shape[1] == mdata["rna"].n_vars

    mdata = add_nhood_expression_mdata.copy()
    milo.add_nhood_expression(mdata)
    nhood_ix = 10
    nhood_gex = mdata["milo"].varm["expr"][nhood_ix, :].toarray().ravel()
    nhood_cells = mdata["rna"].obs_names[mdata["rna"].obsm["nhoods"][:, nhood_ix].toarray().ravel() == 1]
    mean_gex = np.array(mdata["rna"][nhood_cells].X.mean(axis=0)).ravel()
    assert nhood_gex == pytest.approx(mean_gex, 0.0001)


### NEW TESTS


import numpy as np
import pandas as pd
import scipy.sparse as sp

import pytest

from anndata import AnnData

r, _py_to_r, _r_to_py = lazy_import_r_env()

r_string = """
.group_nhoods_from_adjacency_pycomp <- function(nhs, nhood.adj, da.res, is.da,
                                         merge.discord=FALSE,
                                         max.lfc.delta=NULL,
                                         overlap=1,
                                         subset.nhoods=NULL
                                         ){
  # Force everything into a plain base‐R matrix
  nhood.adj <- as.matrix(nhood.adj)

  if(is.null(colnames(nhs))){
    warning("No names attributed to nhoods. Converting indices to names")
    colnames(nhs) <- as.character(seq_len(ncol(nhs)))
  }

  # Subsetting logic (exactly as in miloR)
  if(!is.null(subset.nhoods)){
    if(mode(subset.nhoods) %in% c("character", "logical", "numeric")){
      if(mode(subset.nhoods) %in% c("character")){
        sub.log <- colnames(nhs) %in% subset.nhoods
      } else if (mode(subset.nhoods) %in% c("numeric")) {
        sub.log <- colnames(nhs) %in% colnames(nhs)[subset.nhoods]
      } else{
        sub.log <- subset.nhoods
      }
      nhood.adj <- nhood.adj[sub.log, sub.log]
      if(length(is.da) == ncol(nhs)){
        nhs <- nhs[sub.log]
        is.da <- is.da[sub.log]
        da.res <- da.res[sub.log, ]
      } else{
        stop("Subsetting `is.da` vector length does not equal nhoods length")
      }
    } else{
      stop("Incorrect subsetting vector provided:", class(subset.nhoods))
    }
  } else{
    if(length(is.da) != ncol(nhood.adj)){
      stop("Subsetting `is.da` vector length is not the same dimension as adjacency")
    }
  }

  # Discord‐filter
  if(isFALSE(merge.discord)){
    discord.sign <- sign(da.res[is.da, 'logFC'] %*% t(da.res[is.da, 'logFC'])) < 0
    nhood.adj[is.da, is.da][discord.sign] <- 0
  }

  # Overlap‐filter
  if(overlap > 1){
    nhood.adj[nhood.adj < overlap] <- 0
  }

  # max.lfc.delta‐filter
  if(!is.null(max.lfc.delta)){
    lfc.diff <- sapply(da.res[,"logFC"], "-", da.res[,"logFC"])
    nhood.adj[abs(lfc.diff) > max.lfc.delta] <- 0
  }

  # Binarize
  nhood.adj <- as.matrix((nhood.adj > 0) + 0)

  # Sanity checks
  if(!isSymmetric(nhood.adj)){
    stop("Overlap matrix is not symmetric")
  }
  if(nrow(nhood.adj) != ncol(nhood.adj)){
    stop("Non-square distance matrix - check nhood subsetting")
  }

  return(nhood.adj)
}
"""

r_pkg = r.STAP(r_string, "r_pkg")


def _group_nhoods_from_adjacency_rcomp(
    adjacency: sp.spmatrix,
    da_res: pd.DataFrame,
    is_da: np.ndarray,
    merge_discord: bool = False,
    overlap: int = 1,
    max_lfc_delta: float | None = None,
    subset_nhoods=None,
) -> np.ndarray:
    """
    Mirror of the R code above, returning the final binary adjacency
    (dense array) after applying Discord, Overlap, and ΔlogFC filters.
    """

    # 1) Subset if needed
    if subset_nhoods is not None:
        if isinstance(subset_nhoods, (list, np.ndarray)):
            arr = np.asarray(subset_nhoods)
            if np.issubdtype(arr.dtype, np.integer):
                mask = np.zeros(adjacency.shape[0], dtype=bool)
                mask[arr.astype(int)] = True
            else:
                names = np.array(da_res.index, dtype=str)
                mask = np.isin(names, arr.astype(str))
        elif isinstance(subset_nhoods, (pd.Series, np.ndarray)) and subset_nhoods.dtype == bool:
            if len(subset_nhoods) != adjacency.shape[0]:
                raise ValueError("Boolean subset_nhoods length must match nhood count")
            mask = np.asarray(subset_nhoods, dtype=bool)
        else:
            raise ValueError("subset_nhoods must be bool mask, index list, or name list")

        adjacency = adjacency[mask, :][:, mask]
        da_res     = da_res.loc[mask].copy()
        is_da      = is_da[mask]
    else:
        mask = np.ones(adjacency.shape[0], dtype=bool)

    M = adjacency.shape[0]
    if da_res.shape[0] != M or is_da.shape[0] != M:
        raise ValueError("da_res and is_da must match adjacency dimension after subsetting")

    # 2) Ensure CSR → COO
    if not sp.issparse(adjacency):
        adjacency = sp.csr_matrix(adjacency)
    adjacency = adjacency.tocsr()
    Acoo = adjacency.tocoo()
    rows, cols, data = (np.asarray(Acoo.row, int),
                        np.asarray(Acoo.col, int),
                        np.asarray(Acoo.data, float))

    # 3) Precompute logFC and signs
    lfc_vals = da_res["logFC"].values
    signs    = np.sign(lfc_vals)

    # 4.1) Discord filter
    if merge_discord:
        keep_discord = np.ones_like(data, dtype=bool)
    else:
        is_da_rows = is_da[rows]
        is_da_cols = is_da[cols]
        sign_rows  = signs[rows]
        sign_cols  = signs[cols]
        discord_pair = (is_da_rows & is_da_cols) & (sign_rows * sign_cols < 0)
        keep_discord = ~discord_pair

    # 4.2) Overlap filter
    if overlap <= 1:
        keep_overlap = np.ones_like(data, dtype=bool)
    else:
        keep_overlap = (data >= overlap)

    # 4.3) ΔlogFC filter
    if max_lfc_delta is None:
        keep_lfc = np.ones_like(data, dtype=bool)
    else:
        diffs = np.abs(lfc_vals[rows] - lfc_vals[cols])
        keep_lfc = (diffs <= max_lfc_delta)

    # 5) Combine masks
    keep_mask = keep_discord & keep_overlap & keep_lfc

    # 6) Reconstruct pruned adjacency, then binarize
    new_rows = rows[keep_mask]
    new_cols = cols[keep_mask]
    new_data = data[keep_mask]
    pruned = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=(M, M)).tocsr()
    pruned_bin = (pruned > 0).astype(int).toarray()

    return pruned_bin



@pytest.mark.parametrize("merge_discord_flag, overlap_val, max_lfc_val", [
    (False, 1, None),
    (True,  1, None),
    (False, 2, None),
    (False, 3, None),
    (False, 5, None),
    (False, 15, None),
    (False, 100, None),
    (False, 1, 0.5),
    (False, 1, 1.0),
    (False, 1, 2.0),
    (False, 1, 3.0),
    (False, 1, 4.0),
    (False, 1, 5.0),
])
@pytest.fixture
def test_sparse_adjacency_filters_match_R(annotate_nhoods_mdata, merge_discord_flag, overlap_val, max_lfc_val):
    """
    Compare the output of the R version (via r_pkg) versus the Python version
    for various combinations of merge_discord, overlap, and max_lfc_delta.
    """
    # 4.1) Extract inputs from mdata

    mdata = annotate_nhoods_mdata.copy()

    nhs        = mdata["rna"].obsm["nhoods"].copy()
    nhood_adj  = mdata["milo"].varp["nhood_connectivities"].copy()  # sparse
    da_res     = mdata["milo"].var.copy()
    is_da      = (da_res["SpatialFDR"].values < 0.1) & (da_res["logFC"].values > 0)

    # 4.2) Run R version (returns a dense matrix)
    r_out = r_pkg._group_nhoods_from_adjacency_pycomp(
        _py_to_r(nhs),
        _py_to_r(nhood_adj),
        _py_to_r(da_res),
        _py_to_r(is_da),
        _py_to_r(merge_discord_flag),
        _py_to_r(max_lfc_val),
        _py_to_r(overlap_val),
        subset_nhoods = r.ro.NULL
    )
    adj_R = _r_to_py(r_out)
    # adj_R is a dense NumPy matrix of shape (N, N)

    # 4.3) Run Python version
    adj_py = _group_nhoods_from_adjacency_rcomp(
        nhood_adj,
        da_res,
        is_da,
        merge_discord = merge_discord_flag,
        overlap       = overlap_val,
        max_lfc_delta = max_lfc_val,
        subset_nhoods = None
    )
    # adj_py is also a dense NumPy matrix of shape (N, N)

    # 4.4) Compare element‐by‐element
    # We convert adj_R (which may be a pandas object) explicitly to NumPy
    if not isinstance(adj_R, np.ndarray):
        adj_R = np.asarray(adj_R)
    assert adj_R.shape == adj_py.shape
    # Use a strict equality test (all entries must match)
    assert np.array_equal(adj_R, adj_py), \
        f"Difference detected (merge_discord={merge_discord_flag}, overlap={overlap_val}, max_lfc={max_lfc_val})"