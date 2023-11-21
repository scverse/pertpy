import numpy as np
import pandas as pd
import pertpy as pt
import pytest
import scanpy as sc
from mudata import MuData

try:
    from rpy2.robjects.packages import importr

    r_dependency = importr("edgeR")
except Exception:  # noqa: BLE001
    r_dependency = None


class TestMilopy:
    milo = pt.tl.Milo()

    @pytest.fixture
    def adata(self):
        adata = sc.datasets.pbmc68k_reduced()
        return adata

    def test_load(self, adata):
        mdata = self.milo.load(adata)
        assert isinstance(mdata, MuData)
        assert "rna" in mdata.mod

    def test_make_nhoods_number(self, adata):
        adata = adata.copy()
        p = 0.1
        adata = self.milo.make_nhoods(adata, prop=p, copy=True)
        assert adata.obsm["nhoods"].shape[1] <= int(np.round(adata.n_obs * p))

    def test_make_nhoods_missing_connectivities(self, adata):
        adata = adata.copy()
        p = 0.1
        del adata.obsp["connectivities"]
        with pytest.raises(KeyError):
            adata = self.milo.make_nhoods(adata, prop=p)

    def test_make_nhoods_sizes(self, adata):
        adata = adata.copy()
        self.milo.make_nhoods(adata)
        knn_graph = adata.obsp["connectivities"]
        knn_graph[knn_graph != 0] = 1
        assert knn_graph.sum(0).min() <= adata.obsm["nhoods"].sum(0).min()

    def test_make_nhoods_neighbors_key(self, adata):
        adata = adata.copy()
        k = adata.uns["neighbors"]["params"]["n_neighbors"]
        test_k = 5
        sc.pp.neighbors(adata, n_neighbors=test_k, key_added="test")
        self.milo.make_nhoods(adata, neighbors_key="test")
        smallest_size = adata.obsm["nhoods"].toarray().sum(0).min()
        assert test_k < k
        assert smallest_size < k

    def test_count_nhoods_sample_values(self, adata):
        adata = adata.copy()
        self.milo.make_nhoods(adata)
        # Extract cells of one nhood
        nh = 1
        sample_col = "phase"
        milo_mdata = self.milo.count_nhoods(adata, sample_col=sample_col)
        nh_cells = adata.obsm["nhoods"][:, nh].nonzero()[0]

        # Value count the sample composition
        top_a = adata.obs.iloc[nh_cells].value_counts(sample_col).values.ravel()

        # Check it matches the one calculated
        sample_adata = milo_mdata["milo"]
        df = pd.DataFrame(sample_adata.X.T[nh, :].toarray()).T
        df.index = sample_adata.obs_names
        top_b = df.sort_values(0, ascending=False).values.ravel()
        assert all((top_b - top_a) == 0), 'The counts for samples in milo_mdata["milo"] does not match'

    def test_count_nhoods_missing_nhoods(self, adata):
        adata = adata.copy()
        self.milo.make_nhoods(adata)
        sample_col = "phase"
        del adata.obsm["nhoods"]
        with pytest.raises(KeyError):
            _ = self.milo.count_nhoods(adata, sample_col=sample_col)

    def test_count_nhoods_sample_order(self, adata):
        adata = adata.copy()
        self.milo.make_nhoods(adata)
        # Extract cells of one nhood
        nh = 1
        sample_col = "phase"
        milo_mdata = self.milo.count_nhoods(adata, sample_col=sample_col)
        nh_cells = adata.obsm["nhoods"][:, nh].nonzero()[0]

        # Value count the sample composition
        top_a = adata.obs.iloc[nh_cells].value_counts(sample_col).index[0]

        # Check it matches the one calculated
        sample_adata = milo_mdata["milo"]
        df = pd.DataFrame(sample_adata.X.T[nh, :].toarray()).T
        df.index = sample_adata.obs_names
        top_b = df.sort_values(0, ascending=False).index[0]

        assert top_a == top_b, 'The order of samples in milo_mdata["milo"] does not match'

    # @pytest.mark.skipif(r_dependency is None, reason="Require R dependecy")
    @pytest.fixture
    def da_nhoods_mdata(self, adata):
        adata = adata.copy()
        self.milo.make_nhoods(adata)

        # Simulate experimental condition
        rng = np.random.default_rng(seed=42)
        adata.obs["condition"] = rng.choice(["ConditionA", "ConditionB"], size=adata.n_obs, p=[0.5, 0.5])
        # we simulate differential abundance in NK cells
        DA_cells = adata.obs["louvain"] == "1"
        adata.obs.loc[DA_cells, "condition"] = rng.choice(
            ["ConditionA", "ConditionB"], size=sum(DA_cells), p=[0.2, 0.8]
        )

        # Simulate replicates
        adata.obs["replicate"] = rng.choice(["R1", "R2", "R3"], size=adata.n_obs)
        adata.obs["sample"] = adata.obs["replicate"] + adata.obs["condition"]
        milo_mdata = self.milo.count_nhoods(adata, sample_col="sample")
        return milo_mdata

    # @pytest.mark.skipif(r_dependency is None, reason="Require R dependecy")
    def test_da_nhoods_missing_samples(self, adata):
        with pytest.raises(KeyError):
            self.milo.da_nhoods(adata, design="~condition")

    # @pytest.mark.skipif(r_dependency is None, reason="Require R dependecy")
    def test_da_nhoods_missing_covariate(self, da_nhoods_mdata):
        mdata = da_nhoods_mdata.copy()
        with pytest.raises(KeyError):
            self.milo.da_nhoods(mdata, design="~ciaone")

    # @pytest.mark.skipif(r_dependency is None, reason="Require R dependecy")
    def test_da_nhoods_non_unique_covariate(self, da_nhoods_mdata):
        mdata = da_nhoods_mdata.copy()
        with pytest.raises(AssertionError):
            self.milo.da_nhoods(mdata, design="~phase")

    @pytest.mark.skipif(r_dependency is None, reason="Require R dependecy")
    def test_da_nhoods_pvalues(self, da_nhoods_mdata):
        mdata = da_nhoods_mdata.copy()
        self.milo.da_nhoods(mdata, design="~condition")
        sample_adata = mdata["milo"].copy()
        min_p, max_p = sample_adata.var["PValue"].min(), sample_adata.var["PValue"].max()
        assert (min_p >= 0) & (max_p <= 1), "P-values are not between 0 and 1"

    @pytest.mark.skipif(r_dependency is None, reason="Require R dependecy")
    def test_da_nhoods_fdr(self, da_nhoods_mdata):
        mdata = da_nhoods_mdata.copy()
        self.milo.da_nhoods(mdata, design="~condition")
        sample_adata = mdata["milo"].copy()
        assert np.all(
            np.round(sample_adata.var["PValue"], 10) <= np.round(sample_adata.var["SpatialFDR"], 10)
        ), "FDR is higher than uncorrected P-values"

    @pytest.mark.skipif(r_dependency is None, reason="Require R dependecy")
    def test_da_nhoods_default_contrast(self, da_nhoods_mdata):
        mdata = da_nhoods_mdata.copy()
        adata = mdata["rna"].copy()
        adata.obs["condition"] = (
            adata.obs["condition"].astype("category").cat.reorder_categories(["ConditionA", "ConditionB"])
        )
        self.milo.da_nhoods(mdata, design="~condition")
        default_results = mdata["milo"].var.copy()
        self.milo.da_nhoods(mdata, design="~condition", model_contrasts="conditionConditionB-conditionConditionA")
        contr_results = mdata["milo"].var.copy()

        assert np.corrcoef(contr_results["SpatialFDR"], default_results["SpatialFDR"])[0, 1] > 0.99
        assert np.corrcoef(contr_results["logFC"], default_results["logFC"])[0, 1] > 0.99

    @pytest.fixture
    def annotate_nhoods_mdata(self, adata):
        adata = adata.copy()
        self.milo.make_nhoods(adata)

        # Simulate experimental condition
        rng = np.random.default_rng(seed=42)
        adata.obs["condition"] = rng.choice(["ConditionA", "ConditionB"], size=adata.n_obs, p=[0.5, 0.5])
        # we simulate differential abundance in NK cells
        DA_cells = adata.obs["louvain"] == "1"
        adata.obs.loc[DA_cells, "condition"] = rng.choice(
            ["ConditionA", "ConditionB"], size=sum(DA_cells), p=[0.2, 0.8]
        )

        # Simulate replicates
        adata.obs["replicate"] = rng.choice(["R1", "R2", "R3"], size=adata.n_obs)
        adata.obs["sample"] = adata.obs["replicate"] + adata.obs["condition"]
        milo_mdata = self.milo.count_nhoods(adata, sample_col="sample")
        return milo_mdata

    def test_annotate_nhoods_missing_samples(self, annotate_nhoods_mdata):
        mdata = annotate_nhoods_mdata.copy()
        del mdata.mod["milo"]
        with pytest.raises(ValueError):
            self.milo.annotate_nhoods_continuous(mdata, anno_col="S_score")

    def test_annotate_nhoods_continuous_mean_range(self, annotate_nhoods_mdata):
        mdata = annotate_nhoods_mdata.copy()
        self.milo.annotate_nhoods_continuous(mdata, anno_col="S_score")
        assert mdata["milo"].var["nhood_S_score"].max() < mdata["rna"].obs["S_score"].max()
        assert mdata["milo"].var["nhood_S_score"].min() > mdata["rna"].obs["S_score"].min()

    def test_annotate_nhoods_continuous_correct_mean(self, annotate_nhoods_mdata):
        mdata = annotate_nhoods_mdata.copy()
        self.milo.annotate_nhoods_continuous(mdata, anno_col="S_score")
        rng = np.random.default_rng(seed=42)
        i = rng.choice(np.arange(mdata["milo"].n_obs))
        mean_val_nhood = mdata["rna"].obs[mdata["rna"].obsm["nhoods"][:, i].toarray() == 1]["S_score"].mean()
        assert mdata["milo"].var["nhood_S_score"][i] == pytest.approx(mean_val_nhood, 0.0001)

    def test_annotate_nhoods_annotation_frac_range(self, annotate_nhoods_mdata):
        mdata = annotate_nhoods_mdata.copy()
        self.milo.annotate_nhoods(mdata, anno_col="louvain")
        assert mdata["milo"].var["nhood_annotation_frac"].max() <= 1.0
        assert mdata["milo"].var["nhood_annotation_frac"].min() >= 0.0

    def test_annotate_nhoods_cont_gives_error(self, annotate_nhoods_mdata):
        mdata = annotate_nhoods_mdata.copy()
        with pytest.raises(ValueError):
            self.milo.annotate_nhoods(mdata, anno_col="S_score")

    @pytest.fixture
    def add_nhood_expression_mdata(self):
        adata = sc.datasets.pbmc3k()
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        self.milo.make_nhoods(adata)

        # Simulate experimental condition
        rng = np.random.default_rng(seed=42)
        adata.obs["condition"] = rng.choice(["ConditionA", "ConditionB"], size=adata.n_obs, p=[0.2, 0.8])
        # we simulate differential abundance in NK cells
        DA_cells = adata.obs["leiden"] == "1"
        adata.obs.loc[DA_cells, "condition"] = rng.choice(
            ["ConditionA", "ConditionB"], size=sum(DA_cells), p=[0.2, 0.8]
        )

        # Simulate replicates
        adata.obs["replicate"] = rng.choice(["R1", "R2", "R3"], size=adata.n_obs)
        adata.obs["sample"] = adata.obs["replicate"] + adata.obs["condition"]
        milo_mdata = self.milo.count_nhoods(adata, sample_col="sample")
        return milo_mdata

    def test_add_nhood_expression_nhood_mean_range(self, add_nhood_expression_mdata):
        mdata = add_nhood_expression_mdata.copy()
        self.milo.add_nhood_expression(mdata)
        assert mdata["milo"].varm["expr"].shape[1] == mdata["rna"].n_vars
        mdata = add_nhood_expression_mdata.copy()
        self.milo.add_nhood_expression(mdata)
        nhood_ix = 10
        nhood_gex = mdata["milo"].varm["expr"][nhood_ix, :].toarray().ravel()
        nhood_cells = mdata["rna"].obs_names[mdata["rna"].obsm["nhoods"][:, nhood_ix].toarray().ravel() == 1]
        mean_gex = np.array(mdata["rna"][nhood_cells].X.mean(axis=0)).ravel()
        assert nhood_gex == pytest.approx(mean_gex, 0.0001)
