from importlib.util import find_spec

import numpy as np
import pandas as pd
import pertpy as pt
import pytest
import scanpy as sc
import scipy.sparse as sp
from mudata import MuData


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

from scipy.sparse import csr_matrix


@pytest.fixture
def group_nhoods_mdata(adata, milo):
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
    milo.da_nhoods(milo_mdata, design="~condition", solver="pydeseq2")

    var = milo_mdata["milo"].var.copy()

    n = var.shape[0]
    k = max(1, int(0.1 * n))  # e.g. guarantee 10% are “significant,” at least 1

    # fdrs = np.random.rand(n)
    # New Generator interface:
    rng = np.random.default_rng()
    fdrs = rng.random(n)
    da_idx = rng.choice(n, size=k, replace=False)

    # da_idx = np.random.choice(n, size=k, replace=False)
    # fdrs[da_idx] = np.random.rand(k) * 0.1
    fdrs[da_idx] = rng.random(k) * 0.1

    # np.random.shuffle(fdrs)
    rng.shuffle(fdrs)
    milo_mdata["milo"].var["SpatialFDR"] = fdrs

    milo.build_nhood_graph(milo_mdata)
    return milo_mdata


def csr_to_r_dgCMatrix(csr: csr_matrix):
    """
    Convert a SciPy CSR matrix into an R dgCMatrix using rpy2.

    Returns an rpy2 Matrix object (class “dgCMatrix”).
    """
    import rpy2.robjects as ro
    from rpy2.robjects import FloatVector, IntVector, numpy2ri
    from rpy2.robjects.conversion import localconverter

    # 1) Ensure CSR is in COOrdinate form to extract row/col/data
    coo = csr.tocoo()
    # R is 1-based, so we must add 1 to Python’s 0-based indices:
    i_r = (coo.row + 1).astype(int)
    j_r = (coo.col + 1).astype(int)
    x_r = coo.data

    # 2) Load the Matrix package in R (only if not already loaded)
    ro.r("suppressPackageStartupMessages(library(Matrix))")

    # 3) Build the sparseMatrix(...) call in R
    #    - `sparseMatrix(i=..., j=..., x=..., dims=c(nrow, ncol))` returns a dgCMatrix by default.
    nrow, ncol = csr.shape
    r_sparse = ro.r["sparseMatrix"]

    # 4) Call sparseMatrix(i=IntVector(i_r), j=IntVector(j_r), x=FloatVector(x_r), dims=c(nrow, ncol))
    with localconverter(ro.default_converter + numpy2ri.converter):
        # Pass `dims` as an IntVector of length 2
        dims_vec = IntVector([int(nrow), int(ncol)])
        mat_r = r_sparse(
            i=IntVector(i_r.tolist()),
            j=IntVector(j_r.tolist()),
            x=FloatVector(x_r.tolist()),
            dims=dims_vec,
            index1=ro.BoolVector([True]),  # tell R that i,j are 1-based
        )

    return mat_r


# def _py_to_r(obj):
#     """
#     Convert a Python object into an R object, using only context‐managed converters.
#     - Any 2D array‐like → force to numpy.ndarray, then to R matrix
#     - scipy.sparse → dense numpy → R matrix
#     - pandas.DataFrame → R data.frame
#     - pandas.Series → logical/int/float/character vector
#     - 1D numpy array → R vector
#     - Python scalar / single‐item list → length‐1 R vector
#     - None → R NULL
#     """
#     # Import rpy2 constructs lazily
#     import rpy2.robjects as ro
#     from rpy2.robjects import numpy2ri, pandas2ri
#     from rpy2.robjects.conversion import localconverter

#     # (A) scipy.sparse → dense numpy → R matrix
#     if sp.issparse(obj):
#         arr = obj.toarray()
#         rmat = numpy2ri.py2rpy(arr)
#         return rmat

#     # (B) pandas.DataFrame → R data.frame
#     if isinstance(obj, pd.DataFrame):
#         with localconverter(ro.default_converter + pandas2ri.converter):
#             return pandas2ri.py2rpy(obj)

#     # (C) pandas.Series → logical/int/float/character R vector
#     if isinstance(obj, pd.Series):
#         if obj.dtype == bool:
#             return ro.BoolVector(obj.values.tolist())
#         elif np.issubdtype(obj.dtype, np.integer):
#             return ro.IntVector(obj.values.tolist())
#         elif np.issubdtype(obj.dtype, np.floating):
#             return ro.FloatVector(obj.values.tolist())
#         else:
#             return ro.StrVector(obj.astype(str).tolist())

#     # (D) Force anything array‐like into a NumPy array
#     try:
#         arr = np.asarray(obj)
#     except Exception:
#         arr = None

#     if isinstance(arr, np.ndarray):
#         # 2D array → R matrix
#         rmat = numpy2ri.py2rpy(arr)

#         # 1D array → R vector
#         if arr.ndim == 1:
#             if arr.dtype == bool:
#                 return ro.BoolVector(arr.tolist())
#             elif np.issubdtype(arr.dtype, np.integer):
#                 return ro.IntVector(arr.tolist())
#             elif np.issubdtype(arr.dtype, np.floating):
#                 return ro.FloatVector(arr.tolist())
#             else:
#                 return ro.StrVector(arr.astype(str).tolist())

#     # (E) Python scalar or single‐item list/tuple → length‐1 R vector
#     if isinstance(obj, bool):
#         return ro.BoolVector([obj])
#     if isinstance(obj, int | np.integer):
#         return ro.IntVector([int(obj)])
#     if isinstance(obj, float | np.floating):
#         return ro.FloatVector([float(obj)])
#     if isinstance(obj, str):
#         return ro.StrVector([obj])

#     # (F) None → R NULL
#     if obj is None:
#         return ro.NULL

#     # (G) Python list of simple types → convert to numpy array then recurse
#     if isinstance(obj, list):
#         return _py_to_r(np.asarray(obj))

#     # (H) Otherwise, cannot convert
#     raise ValueError(f"Cannot convert object of type {type(obj)} to R.")


def _py_to_r(obj):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    if isinstance(obj, np.ndarray):
        return numpy2ri.py2rpy(obj)
    if isinstance(obj, pd.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = pandas2ri.py2rpy(obj)
        return df
    if obj is None:
        return ro.NULL
    with localconverter(ro.default_converter):
        r_obj = ro.conversion.py2rpy(obj)
    return r_obj


# def _r_to_py(r_obj):
#     """
#     Convert an R object back into a NumPy array or pandas DataFrame, as appropriate.
#     """
#     import rpy2.robjects as ro
#     from rpy2.robjects import numpy2ri, pandas2ri
#     from rpy2.robjects.conversion import localconverter

#     # R matrix → NumPy array
#     if isinstance(r_obj, ro.vectors.Matrix):
#         with localconverter(ro.default_converter + numpy2ri.converter):
#             return np.asarray(r_obj)

#     # R data.frame → pandas DataFrame
#     if isinstance(r_obj, ro.vectors.DataFrame):
#         with localconverter(ro.default_converter + pandas2ri.converter):
#             return pandas2ri.rpy2py(r_obj)

#     # R vector types → NumPy arrays
#     if isinstance(r_obj, ro.vectors.BoolVector):
#         return np.asarray(r_obj)
#     if isinstance(r_obj, ro.vectors.IntVector):
#         return np.asarray(r_obj)
#     if isinstance(r_obj, ro.vectors.FloatVector):
#         return np.asarray(r_obj)
#     if isinstance(r_obj, ro.vectors.StrVector):
#         return np.asarray(r_obj)

#     # Otherwise (e.g. R NULL), return as‐is
#     return r_obj


def _group_nhoods_from_adjacency_r(
    nhs_r, nhood_adj_r, da_res_r, is_da_r, merge_discord_r, max_lfc_delta_r, overlap_r, subset_nhoods_r
):
    """
    Lazily define the R function .group_nhoods_from_adjacency_pycomp in R’s global env,
    then call it directly with arguments that are already R objects.
    Returns an R matrix of 0/1.
    """
    # Import rpy2 inside the function
    import rpy2.robjects as ro

    # Define the R function in R’s global environment:
    rcode = r"""
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

      # Subsetting logic (as in miloR)
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
        stop("Non-square distance matrix ‐ check nhood subsetting")
      }

      return(nhood.adj)
    }
    """
    # Evaluate rcode in R's global environment (defines the function):
    ro.r(rcode)

    # Now retrieve that function from R's globalenv and call it:
    f = ro.globalenv[".group_nhoods_from_adjacency_pycomp"]
    return f(nhs_r, nhood_adj_r, da_res_r, is_da_r, merge_discord_r, max_lfc_delta_r, overlap_r, subset_nhoods_r)


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
    Pure‐Python implementation.
    """
    # 1) Subset if needed
    if subset_nhoods is not None:
        if isinstance(subset_nhoods, list | np.ndarray):
            arr = np.asarray(subset_nhoods)
            if np.issubdtype(arr.dtype, np.integer):
                mask = np.zeros(adjacency.shape[0], dtype=bool)
                mask[arr.astype(int)] = True
            else:
                names = np.array(da_res.index, dtype=str)
                mask = np.isin(names, arr.astype(str))
        elif isinstance(subset_nhoods, pd.Series | np.ndarray) and getattr(subset_nhoods, "dtype", None) is bool:
            if len(subset_nhoods) != adjacency.shape[0]:
                raise ValueError("Boolean subset_nhoods length must match nhood count")
            mask = np.asarray(subset_nhoods, dtype=bool)
        else:
            raise ValueError("subset_nhoods must be bool mask, index list, or name list")

        adjacency = adjacency[mask, :][:, mask]
        da_res = da_res.loc[mask].copy()
        is_da = is_da[mask]
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
    rows, cols, data = (np.asarray(Acoo.row, int), np.asarray(Acoo.col, int), np.asarray(Acoo.data, float))

    # 3) Precompute logFC and signs
    lfc_vals = da_res["logFC"].values
    signs = np.sign(lfc_vals)

    # 4.1) Discord filter
    if merge_discord:
        keep_discord = np.ones_like(data, dtype=bool)
    else:
        is_da_rows = is_da[rows]
        is_da_cols = is_da[cols]
        sign_rows = signs[rows]
        sign_cols = signs[cols]
        discord_pair = (is_da_rows & is_da_cols) & (sign_rows * sign_cols < 0)
        keep_discord = ~discord_pair

    # 4.2) Overlap filter
    keep_overlap = np.ones_like(data, dtype=bool) if overlap <= 1 else data >= overlap

    # 4.3) ΔlogFC filter
    if max_lfc_delta is None:
        keep_lfc = np.ones_like(data, dtype=bool)
    else:
        diffs = np.abs(lfc_vals[rows] - lfc_vals[cols])
        keep_lfc = diffs <= max_lfc_delta

    # 5) Combine masks
    keep_mask = keep_discord & keep_overlap & keep_lfc

    # 6) Reconstruct pruned adjacency, then binarize
    new_rows = rows[keep_mask]
    new_cols = cols[keep_mask]
    new_data = data[keep_mask]
    pruned = sp.coo_matrix((new_data, (new_rows, new_cols)), shape=(M, M)).tocsr()
    pruned_bin = (pruned > 0).astype(int).toarray()

    return pruned_bin


@pytest.mark.parametrize(
    "merge_discord_flag, overlap_val, max_lfc_val",
    [
        (False, 1, None),
        (True, 1, None),
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
    ],
)
def test_sparse_adjacency_filters_match_R(group_nhoods_mdata, merge_discord_flag, overlap_val, max_lfc_val):
    """
    Compare the R version against the Python version for various
    settings of merge_discord, overlap, and max_lfc_delta.
    """
    # 1) Extract inputs from fixture
    mdata = group_nhoods_mdata.copy()
    nhs = mdata["rna"].obsm["nhoods"].copy()
    nhood_adj = mdata["milo"].varp["nhood_connectivities"].copy()  # sparse
    da_res = mdata["milo"].var.copy()  # DataFrame with "logFC"
    is_da = (da_res["SpatialFDR"].values < 0.1) & (da_res["logFC"].values > 0)

    # 2) Convert to R objects
    nhs_r = _py_to_r(np.zeros((nhs.shape[0], nhs.shape[1])))
    nhood_adj_r = csr_to_r_dgCMatrix(nhood_adj)
    da_res_r = _py_to_r(da_res)
    is_da_r = _py_to_r(is_da)
    merge_discord_r = _py_to_r(bool(merge_discord_flag))
    overlap_r = _py_to_r(int(overlap_val))
    max_lfc_delta_r = _py_to_r(max_lfc_val)  # None → R NULL
    subset_nhoods_r = _py_to_r(None)  # always None here

    # 3) Call the R implementation
    r_out = _group_nhoods_from_adjacency_r(
        nhs_r, nhood_adj_r, da_res_r, is_da_r, merge_discord_r, max_lfc_delta_r, overlap_r, subset_nhoods_r
    )

    # 4) Convert R output → NumPy
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    if isinstance(r_out, ro.vectors.Matrix):
        with localconverter(ro.default_converter + numpy2ri.converter):
            adj_R = np.asarray(r_out)
    else:
        adj_R = np.asarray(r_out)

    assert adj_R.shape == nhood_adj.shape

    # 5) Call the Python implementation
    adj_py = _group_nhoods_from_adjacency_rcomp(
        nhood_adj,
        da_res,
        is_da,
        merge_discord=merge_discord_flag,
        overlap=overlap_val,
        max_lfc_delta=max_lfc_val,
        subset_nhoods=None,
    )

    # 6) Compare
    assert adj_py.shape == adj_R.shape
    assert np.array_equal(adj_py, adj_R), (
        f"Mismatch for (merge_discord={merge_discord_flag}, overlap={overlap_val}, max_lfc={max_lfc_val})"
    )


@pytest.mark.parametrize(
    "nhood_group_obs, subset_nhoods, min_n_nhoods, mode",
    [
        # default obs-name, all neighborhoods, last-wins
        ("nhood_groups", None, 1, "last_wins"),
        # default obs-name, only neighborhoods "0" and "1", last-wins
        ("nhood_groups", None, 1, "last_wins"),
        # default obs-name, all neighborhoods, exclude overlaps with threshold 2
        ("nhood_groups", None, 2, "exclude_overlaps"),
        # custom obs-name, only neighborhood "2", exclude overlaps w/ threshold 3
        ("custom_lbls", None, 3, "exclude_overlaps"),
    ],
)
def test_annotate_cells_from_nhoods_various(
    group_nhoods_mdata,
    nhood_group_obs,
    subset_nhoods,
    min_n_nhoods,
    mode,
):
    from pertpy.tools._milo import Milo

    milo = Milo()
    print(milo)
    mdata = group_nhoods_mdata.copy()

    assert "SpatialFDR" in mdata["milo"].var.columns
    milo.group_nhoods(mdata)
    if nhood_group_obs == "custom_lbls":
        # Create a custom nhood group obs column
        mdata["milo"].var["custom_lbls"] = mdata["milo"].var["nhood_groups"].copy()

    # run the annotation
    milo.annotate_cells_from_nhoods(
        mdata,
        nhood_group_obs=nhood_group_obs,
        subset_nhoods=subset_nhoods,
        min_n_nhoods=min_n_nhoods,
        mode=mode,
    )

    if nhood_group_obs == "custom_lbls":
        # Create a custom nhood group obs column
        mdata["rna"].obs["custom_lbls"] = mdata["rna"].obs["nhood_groups"].copy()

    # the new column must exist
    assert nhood_group_obs in mdata["rna"].obs.columns

    col = mdata["rna"].obs[nhood_group_obs]

    # type checks
    assert col.dtype == object
    assert len(col) == mdata["rna"].n_obs

    # non-annotated cells should be NaN
    # annotated cells should only use labels from the chosen neighborhoods
    # non_null = col.dropna().astype(str).values
    # non_null = [
    #     x for x in col.astype(str).values
    #     if x.lower() not in ("nan", "<NA>")
    # ]

    # if subset_nhoods is None:
    #     # allowed = set(mdata["milo"].var[nhood_group_obs].astype(str).unique())
    #     allowed = {
    #         x for x in mdata["milo"].var[nhood_group_obs].astype(str).unique()
    #         if x.lower() != "nan"
    #     }
    # else:
    #     allowed = set(subset_nhoods)
    # 1) grab the raw Series
    ser_obs = mdata["rna"].obs[nhood_group_obs]
    ser_var = mdata["milo"].var[nhood_group_obs]

    # 2) drop real missing values
    ser_obs_non_na = ser_obs[ser_obs.notna()]
    ser_var_non_na = ser_var[ser_var.notna()]

    # 3) convert to str for comparison
    non_null = set(ser_obs_non_na.astype(str).unique())
    allowed = set(ser_var_non_na.astype(str).unique())

    assert non_null.issubset(allowed)

    # assert set(non_null).issubset(allowed)

    # For exclude_overlaps, ensure that no cell is assigned if it belongs
    # to fewer than min_n_nhoods neighborhoods
    if mode == "exclude_overlaps":
        # build a quick membership count:
        np.asarray(mdata["rna"].obs[nhood_group_obs].notna(), bool)
        counts = (mdata["rna"].obsm["nhoods"].astype(int)).sum(axis=1)
        too_few = counts < min_n_nhoods
        too_few = np.asarray(too_few).ravel()
        assert (col[too_few].isna()).all()
