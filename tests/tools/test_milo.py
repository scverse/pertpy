from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from mudata import MuData

import pertpy as pt


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


def test_da_nhoods_pvalues(da_nhoods_mdata, milo, solver):
    mdata = da_nhoods_mdata.copy()
    milo.da_nhoods(mdata, design="~condition", solver=solver)
    sample_adata = mdata["milo"].copy()
    min_p, max_p = sample_adata.var["PValue"].min(), sample_adata.var["PValue"].max()
    assert (min_p >= 0) & (max_p <= 1), "P-values are not between 0 and 1"


def test_da_nhoods_fdr(da_nhoods_mdata, milo, solver):
    mdata = da_nhoods_mdata.copy()
    milo.da_nhoods(mdata, design="~condition", solver=solver)
    sample_adata = mdata["milo"].copy()
    assert np.all(np.round(sample_adata.var["PValue"], 10) <= np.round(sample_adata.var["SpatialFDR"], 10)), (
        "FDR is higher than uncorrected P-values"
    )


def test_da_nhoods_default_contrast(da_nhoods_mdata, milo, solver):
    mdata = da_nhoods_mdata.copy()
    adata = mdata["rna"].copy()
    adata.obs["condition"] = (
        adata.obs["condition"].astype("category").cat.reorder_categories(["ConditionA", "ConditionB"])
    )
    milo.da_nhoods(mdata, design="~condition", solver=solver)
    default_results = mdata["milo"].var.copy()
    milo.da_nhoods(mdata, design="~condition", model_contrasts="conditionConditionB-conditionConditionA", solver=solver)
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


@pytest.fixture
def de_nhoods_mdata(da_nhoods_mdata):
    # da_nhoods_mdata is built from pbmc68k_reduced whose X is log-normalized.
    # de_nhoods needs an integer-count layer (for PyDESeq2) so we simulate one
    # by drawing NB counts whose mean tracks the normalized expression — plus a
    # planted DE effect on a chosen gene to make the test directional.
    mdata = da_nhoods_mdata.copy()
    rna = mdata["rna"]
    rng = np.random.default_rng(0)
    base = np.asarray(rna.X.todense() if hasattr(rna.X, "todense") else rna.X)
    mu = np.clip(np.expm1(np.clip(base, 0, 8)), 0.0, 200.0) + 0.5
    # planted DE: gene 0 up in ConditionB
    is_b = (rna.obs["condition"] == "ConditionB").to_numpy()
    mu[is_b, 0] *= 5.0
    counts = rng.negative_binomial(n=4, p=4 / (4 + mu)).astype(np.int32)
    rna.layers["counts"] = counts
    rna.uns["de_gene"] = rna.var_names[0]
    return mdata


def test_de_nhoods_shapes(de_nhoods_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    mdata = de_nhoods_mdata.copy()
    de = milo.de_nhoods(
        mdata,
        design="~condition",
        column="condition",
        baseline="ConditionA",
        group_to_compare="ConditionB",
        layer="counts",
        min_n_cells_per_sample=2,
        min_count=1,
    )
    expected = mdata["milo"].n_vars * mdata["rna"].n_vars
    assert len(de) == expected
    for c in [
        "nhood",
        "variable",
        "log_fc",
        "p_value",
        "adj_p_value",
        "pval_corrected_across_nhoods",
        "test_performed",
    ]:
        assert c in de.columns
    assert de["test_performed"].dtype == bool


def test_de_nhoods_fdr_bounds(de_nhoods_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    mdata = de_nhoods_mdata.copy()
    de = milo.de_nhoods(
        mdata,
        design="~condition",
        column="condition",
        baseline="ConditionA",
        group_to_compare="ConditionB",
        layer="counts",
        min_n_cells_per_sample=2,
        min_count=1,
    )
    valid = de.dropna(subset=["p_value"])
    assert ((valid["p_value"] >= 0) & (valid["p_value"] <= 1)).all()
    both_g = valid.dropna(subset=["adj_p_value"])
    assert (both_g["p_value"] <= both_g["adj_p_value"] + 1e-12).all()
    both_n = valid.dropna(subset=["pval_corrected_across_nhoods"])
    assert (both_n["p_value"] <= both_n["pval_corrected_across_nhoods"] + 1e-12).all()


def test_de_nhoods_planted_signal(de_nhoods_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    mdata = de_nhoods_mdata.copy()
    de = milo.de_nhoods(
        mdata,
        design="~condition",
        column="condition",
        baseline="ConditionA",
        group_to_compare="ConditionB",
        layer="counts",
        min_n_cells_per_sample=2,
        min_count=1,
    )
    g = mdata["rna"].uns["de_gene"]
    lfc = de.loc[de["variable"] == g, "log_fc"].dropna()
    assert np.median(lfc) > 0


def test_plot_de_nhood_graph(de_nhoods_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    import matplotlib

    matplotlib.use("Agg")
    mdata = de_nhoods_mdata.copy()
    milo.build_nhood_graph(mdata)
    de = milo.de_nhoods(
        mdata,
        design="~condition",
        column="condition",
        baseline="ConditionA",
        group_to_compare="ConditionB",
        layer="counts",
        min_n_cells_per_sample=2,
        min_count=1,
    )
    g = mdata["rna"].uns["de_gene"]
    fig = milo.plot_de_nhood_graph(mdata, de, gene=g, return_fig=True)
    assert fig is not None
    with pytest.raises(KeyError):
        milo.plot_de_nhood_graph(mdata, de, gene="not_a_real_gene")


def test_de_nhoods_statsmodels_runs(de_nhoods_mdata, milo):
    mdata = de_nhoods_mdata.copy()
    de = milo.de_nhoods(
        mdata,
        design="~condition",
        column="condition",
        baseline="ConditionA",
        group_to_compare="ConditionB",
        solver="statsmodels",
        layer="counts",
        min_n_cells_per_sample=2,
        min_count=1,
    )
    assert de["test_performed"].any()


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
    sc.tl.leiden(adata, flavor="igraph")
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


@pytest.fixture
def grouped_mdata(de_nhoods_mdata, milo, rng):
    """Milo object with a neighbourhood graph, synthetic DA results, groups and per-cell annotations."""
    pytest.importorskip("igraph")
    mdata = de_nhoods_mdata.copy()
    milo.build_nhood_graph(mdata)
    n = mdata["milo"].n_vars
    n_da = max(2, n // 4)
    mdata["milo"].var["logFC"] = rng.normal(0.0, 2.0, size=n)
    fdr = rng.uniform(0.2, 1.0, size=n)
    fdr[rng.choice(n, size=n_da, replace=False)] = rng.uniform(0.0, 0.05, size=n_da)
    mdata["milo"].var["SpatialFDR"] = fdr
    milo.group_nhoods(mdata)
    milo.annotate_cells_from_nhoods(mdata)
    return mdata


def test_group_nhoods_from_adjacency_filters(milo):
    pytest.importorskip("igraph")
    # two triangles ({0,1,2} and {3,4,5}) joined by a single weak bridge (2-3)
    weights = np.array(
        [
            [0, 5, 5, 0, 0, 0],
            [5, 0, 5, 0, 0, 0],
            [5, 5, 0, 1, 0, 0],
            [0, 0, 1, 0, 5, 5],
            [0, 0, 0, 5, 0, 5],
            [0, 0, 0, 5, 5, 0],
        ],
        dtype=float,
    )
    adjacency = sp.csr_matrix(weights)
    is_da = np.ones(6, dtype=bool)

    # the bridge is sign-discordant, so it is dropped and the triangles form separate groups
    discordant = pd.DataFrame({"logFC": [2, 2, 2, -2, -2, -2], "SpatialFDR": [0.01] * 6})
    labels = milo._group_nhoods_from_adjacency(adjacency, discordant, is_da, merge_discord=False)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]

    # an overlap threshold above the bridge weight also splits the triangles
    concordant = pd.DataFrame({"logFC": [2, 2, 2, 2, 2, 2], "SpatialFDR": [0.01] * 6})
    labels_overlap = milo._group_nhoods_from_adjacency(adjacency, concordant, is_da, merge_discord=True, overlap=2)
    assert labels_overlap[2] != labels_overlap[3]


def test_group_nhoods_writes_labels(grouped_mdata):
    # every neighbourhood is grouped, matching miloR which clusters all nhoods
    groups = grouped_mdata["milo"].var["nhood_groups"]
    assert groups.notna().all()
    assert groups.nunique() >= 1


def test_group_nhoods_requires_da_results(da_nhoods_mdata, milo):
    with pytest.raises(KeyError):
        milo.group_nhoods(da_nhoods_mdata)


def test_annotate_cells_from_nhoods(grouped_mdata):
    cell_labels = grouped_mdata["rna"].obs["nhood_groups"]
    nhood_labels = set(grouped_mdata["milo"].var["nhood_groups"].dropna().astype(str))
    assert cell_labels.notna().any()
    assert set(cell_labels.dropna().astype(str)) <= nhood_labels


@pytest.fixture
def markers_mdata(de_nhoods_mdata, rng):
    """Milo object with raw counts and cell-level neighbourhood groups for marker testing."""
    mdata = de_nhoods_mdata.copy()
    mdata["rna"].obs["nhood_groups"] = pd.Categorical(rng.choice(["g0", "g1"], size=mdata["rna"].n_obs))
    return mdata


def test_find_nhood_group_markers_two_group(markers_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    df = milo.find_nhood_group_markers(
        markers_mdata, group_to_compare="g1", baseline="g0", sample_col="sample", layer="counts"
    )
    assert {"variable", "log_fc", "p_value", "adj_p_value"}.issubset(df.columns)
    assert "group" not in df.columns
    assert len(df) > 0
    assert df["p_value"].dropna().between(0, 1).all()


def test_find_nhood_group_markers_one_vs_rest(markers_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    mdata = markers_mdata.copy()
    mdata["rna"].obs["nhood_groups"] = pd.Categorical(np.resize(["g0", "g1", "g2"], mdata["rna"].n_obs))
    df = milo.find_nhood_group_markers(mdata, sample_col="sample", layer="counts")
    assert "group" in df.columns
    assert df["group"].nunique() > 1


def test_find_nhood_group_markers_invalid_args(markers_mdata, milo):
    with pytest.raises(ValueError):
        milo.find_nhood_group_markers(markers_mdata, baseline="g0", sample_col="sample", layer="counts")
    with pytest.raises(ValueError):
        milo.find_nhood_group_markers(
            markers_mdata, group_to_compare="g0", baseline="g0", sample_col="sample", layer="counts"
        )
    with pytest.raises(KeyError):
        milo.find_nhood_group_markers(markers_mdata, nhood_group_key="missing", sample_col="sample", layer="counts")


# --- _nhood_subset_mask -------------------------------------------------


def test_nhood_subset_mask_boolean(milo):
    names = np.array([f"nhood_{i}" for i in range(4)])
    subset = np.array([True, False, True, False])
    mask = milo._nhood_subset_mask(names, subset)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, subset)


def test_nhood_subset_mask_integer_index(milo):
    names = np.array([f"nhood_{i}" for i in range(4)])
    mask = milo._nhood_subset_mask(names, [0, 2])
    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))


def test_nhood_subset_mask_names(milo):
    names = np.array([f"nhood_{i}" for i in range(4)])
    mask = milo._nhood_subset_mask(names, ["nhood_0", "nhood_2"])
    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))


def test_nhood_subset_mask_boolean_wrong_length(milo):
    names = np.array([f"nhood_{i}" for i in range(4)])
    with pytest.raises(ValueError):
        milo._nhood_subset_mask(names, np.array([True, False]))


# --- group_nhoods --------------------------------------------------------


@pytest.fixture
def abundance_mdata(de_nhoods_mdata, rng):
    """MuData with synthetic per-nhood DA results (SpatialFDR/logFC) but no nhood graph built yet."""
    mdata = de_nhoods_mdata.copy()
    n = mdata["milo"].n_vars
    n_da = max(2, n // 4)
    mdata["milo"].var["logFC"] = rng.normal(0.0, 2.0, size=n)
    fdr = rng.uniform(0.2, 1.0, size=n)
    fdr[rng.choice(n, size=n_da, replace=False)] = rng.uniform(0.0, 0.05, size=n_da)
    mdata["milo"].var["SpatialFDR"] = fdr
    return mdata


def test_group_nhoods_missing_nhood_connectivities(abundance_mdata, milo):
    # build_nhood_graph was never run, so `nhood_connectivities` is absent from varp
    with pytest.raises(KeyError):
        milo.group_nhoods(abundance_mdata)


def test_group_nhoods_all_fdr_nan(abundance_mdata, milo):
    mdata = abundance_mdata.copy()
    milo.build_nhood_graph(mdata)
    mdata["milo"].var["SpatialFDR"] = np.nan
    with pytest.raises(ValueError):
        milo.group_nhoods(mdata)


def test_group_nhoods_zero_da_nhoods(abundance_mdata, milo):
    mdata = abundance_mdata.copy()
    milo.build_nhood_graph(mdata)
    with pytest.raises(ValueError):
        milo.group_nhoods(mdata, da_fdr=0)


def test_group_nhoods_subset_boolean(abundance_mdata, milo):
    pytest.importorskip("igraph")
    mdata = abundance_mdata.copy()
    milo.build_nhood_graph(mdata)
    n = mdata["milo"].n_vars
    mask = np.zeros(n, dtype=bool)
    mask[: n // 2] = True
    milo.group_nhoods(mdata, subset_nhoods=mask)
    groups = mdata["milo"].var["nhood_groups"]
    assert groups[mask].notna().all()
    assert groups[~mask].isna().all()


def test_group_nhoods_subset_int_indices(abundance_mdata, milo):
    pytest.importorskip("igraph")
    mdata = abundance_mdata.copy()
    milo.build_nhood_graph(mdata)
    n = mdata["milo"].n_vars
    idx = list(range(n // 2))
    milo.group_nhoods(mdata, subset_nhoods=idx)
    groups = mdata["milo"].var["nhood_groups"]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    assert groups[mask].notna().all()
    assert groups[~mask].isna().all()


def test_group_nhoods_subset_names(abundance_mdata, milo):
    pytest.importorskip("igraph")
    mdata = abundance_mdata.copy()
    milo.build_nhood_graph(mdata)
    n = mdata["milo"].n_vars
    names = list(mdata["milo"].var_names[: n // 2])
    milo.group_nhoods(mdata, subset_nhoods=names)
    groups = mdata["milo"].var["nhood_groups"]
    assert groups.loc[names].notna().all()
    assert groups.drop(index=names).isna().all()


def test_group_nhoods_max_lfc_delta(milo):
    pytest.importorskip("igraph")
    # fully-connected graph over 6 nhoods with concordant logFC signs (all positive),
    # so neither the discordance filter nor the graph topology alone would split it;
    # only `max_lfc_delta` can prune the cross-magnitude edges into two triangles.
    weights = np.ones((6, 6)) - np.eye(6)
    adata = AnnData(np.zeros((1, 6)))
    adata.var["logFC"] = [2.0, 2.0, 2.0, 20.0, 20.0, 20.0]
    adata.var["SpatialFDR"] = 0.01
    adata.varp["nhood_connectivities"] = sp.csr_matrix(weights)

    milo.group_nhoods(adata, max_lfc_delta=None, key_added="groups_no_delta")
    milo.group_nhoods(adata, max_lfc_delta=1.0, key_added="groups_with_delta")

    no_delta = adata.var["groups_no_delta"]
    with_delta = adata.var["groups_with_delta"]
    assert no_delta.iloc[0] == no_delta.iloc[3]
    assert with_delta.iloc[0] != with_delta.iloc[3]


# --- annotate_cells_from_nhoods ------------------------------------------


def test_annotate_cells_from_nhoods_missing_groups(de_nhoods_mdata, milo):
    mdata = de_nhoods_mdata.copy()
    with pytest.raises(KeyError):
        milo.annotate_cells_from_nhoods(mdata)


def test_annotate_cells_from_nhoods_all_missing_groups(de_nhoods_mdata, milo):
    mdata = de_nhoods_mdata.copy()
    mdata["milo"].var["nhood_groups"] = pd.array([pd.NA] * mdata["milo"].n_vars, dtype="object")
    with pytest.raises(ValueError):
        milo.annotate_cells_from_nhoods(mdata)


# --- find_nhood_group_markers --------------------------------------------


def test_find_nhood_group_markers_unknown_var_names(markers_mdata, milo):
    with pytest.raises(KeyError):
        milo.find_nhood_group_markers(markers_mdata, sample_col="sample", layer="counts", var_names=["not_a_real_gene"])


def test_find_nhood_group_markers_invalid_level(markers_mdata, milo):
    with pytest.raises(ValueError):
        milo.find_nhood_group_markers(
            markers_mdata, group_to_compare="not_a_group", baseline="g0", sample_col="sample", layer="counts"
        )
    with pytest.raises(ValueError):
        milo.find_nhood_group_markers(
            markers_mdata, group_to_compare="g1", baseline="not_a_group", sample_col="sample", layer="counts"
        )


def test_find_nhood_group_markers_single_group_after_aggregation(markers_mdata, milo):
    mdata = markers_mdata.copy()
    mdata["rna"].obs["nhood_groups"] = pd.Categorical(["g0"] * mdata["rna"].n_obs)
    with pytest.raises(ValueError):
        milo.find_nhood_group_markers(mdata, sample_col="sample", layer="counts")


def test_find_nhood_group_markers_var_names_restriction(markers_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    mdata = markers_mdata.copy()
    genes = list(mdata["rna"].var_names[:5])
    df = milo.find_nhood_group_markers(
        mdata, group_to_compare="g1", baseline="g0", sample_col="sample", layer="counts", var_names=genes
    )
    assert set(df["variable"]) <= set(genes)
    assert len(df) <= len(genes)


def test_find_nhood_group_markers_n_top_genes(markers_mdata, milo):
    if find_spec("pydeseq2") is None:
        pytest.skip("pydeseq2 not available")
    mdata = markers_mdata.copy()
    n_top = 5
    df = milo.find_nhood_group_markers(
        mdata, group_to_compare="g1", baseline="g0", sample_col="sample", layer="counts", n_top_genes=n_top
    )
    assert df["variable"].nunique() <= n_top


def test_find_nhood_group_markers_edger_solver(markers_mdata, milo):
    try:
        from rpy2.robjects.packages import importr

        importr("edgeR")
    except Exception:  # noqa: BLE001
        pytest.skip("Required R package 'edgeR' not available")
    df = milo.find_nhood_group_markers(
        markers_mdata, group_to_compare="g1", baseline="g0", sample_col="sample", layer="counts", solver="edger"
    )
    assert {"variable", "log_fc", "p_value", "adj_p_value"}.issubset(df.columns)
    assert len(df) > 0


def test_find_nhood_group_markers_unknown_solver(markers_mdata, milo):
    with pytest.raises(ValueError):
        milo.find_nhood_group_markers(
            markers_mdata, group_to_compare="g1", baseline="g0", sample_col="sample", layer="counts", solver="nope"
        )
