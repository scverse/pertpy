from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

import pertpy as pt

CWD = Path(__file__).parent.resolve()
threshold = 80


class TestMixscape:
    def make_test_adata(self):
        np.random.seed(22)
        # gene_1
        NT = np.zeros(100)
        NP = np.zeros(100)
        KO = np.zeros(100)
        gene_1 = np.concatenate((NT, NP, KO))
        gene_1 = np.expand_dims(gene_1, axis=1)

        # gene_2
        NT = np.zeros(100)
        NP = np.zeros(100)
        KO = np.zeros(100)
        gene_2 = np.concatenate((NT, NP, KO))
        gene_2 = np.expand_dims(gene_2, axis=1)

        # gene_3
        NT = np.zeros(100)
        NP = np.zeros(100)
        KO = np.zeros(100)
        gene_3 = np.concatenate((NT, NP, KO))
        gene_3 = np.expand_dims(gene_3, axis=1)

        # gene_4
        NT = np.zeros(100)
        NP = np.zeros(100)
        KO = np.zeros(100)
        gene_4 = np.concatenate((NT, NP, KO))
        gene_4 = np.expand_dims(gene_4, axis=1)

        # gene_5
        mu_NT, sigma_NT = 0.8416804, 0.6765929
        NT = np.random.normal(mu_NT, sigma_NT, 100)
        NT = np.where(NT < 0, 0, NT)

        mu_NP, sigma_NP = 0.8416804, 0.6765929
        NP = np.random.normal(mu_NP, sigma_NP, 100)
        NP = np.where(NP < 0, 0, NP)

        mu_KO, sigma_KO = 1.3398486, 0.66843104
        KO = np.random.normal(mu_KO, sigma_KO, 100)
        KO = np.where(KO < 0, 0, KO)

        gene_5 = np.concatenate((NT, NP, KO))
        gene_5 = np.expand_dims(gene_5, axis=1)

        # gene_6
        mu_NT, sigma_NT = 0.52233374, 0.5800075
        NT = np.random.normal(mu_NT, sigma_NT, 100)
        NT = np.where(NT < 0, 0, NT)

        mu_NP, sigma_NP = 0.52233374, 0.5800075
        NP = np.random.normal(mu_NP, sigma_NP, 100)
        NP = np.where(NP < 0, 0, NP)

        mu_KO, sigma_KO = 1.9704507, 0.7592313
        KO = np.random.normal(mu_KO, sigma_KO, 100)
        KO = np.where(KO < 0, 0, KO)

        gene_6 = np.concatenate((NT, NP, KO))
        gene_6 = np.expand_dims(gene_6, axis=1)

        # gene_7
        mu_NT, sigma_NT = 2.1935444, 1.0035002
        NT = np.random.normal(mu_NT, sigma_NT, 100)
        NT = np.where(NT < 0, 0, NT)

        mu_NP, sigma_NP = 2.1935444, 1.0035002
        NP = np.random.normal(mu_NP, sigma_NP, 100)
        NP = np.where(NP < 0, 0, NP)

        mu_KO, sigma_KO = 3.8569412, 0.9455437
        KO = np.random.normal(mu_KO, sigma_KO, 100)
        KO = np.where(KO < 0, 0, KO)

        gene_7 = np.concatenate((NT, NP, KO))
        gene_7 = np.expand_dims(gene_7, axis=1)

        # gene_8
        mu_NT, sigma_NT = 0.36024597, 0.5157105
        NT = np.random.normal(mu_NT, sigma_NT, 100)
        NT = np.where(NT < 0, 0, NT)

        mu_NP, sigma_NP = 0.36024597, 0.5157105
        NP = np.random.normal(mu_NP, sigma_NP, 100)
        NP = np.where(NP < 0, 0, NP)

        mu_KO, sigma_KO = 1.7390175, 0.6632239
        KO = np.random.normal(mu_KO, sigma_KO, 100)
        KO = np.where(KO < 0, 0, KO)

        gene_8 = np.concatenate((NT, NP, KO))
        gene_8 = np.expand_dims(gene_8, axis=1)

        # gene_9
        mu_NT, sigma_NT = 0.5276356, 0.62142557
        NT = np.random.normal(mu_NT, sigma_NT, 100)
        NT = np.where(NT < 0, 0, NT)

        mu_NP, sigma_NP = 0.5276356, 0.62142557
        NP = np.random.normal(mu_NP, sigma_NP, 100)
        NP = np.where(NP < 0, 0, NP)

        mu_KO, sigma_KO = 1.0565073, 0.85542923
        KO = np.random.normal(mu_KO, sigma_KO, 100)
        KO = np.where(KO < 0, 0, KO)

        gene_9 = np.concatenate((NT, NP, KO))
        gene_9 = np.expand_dims(gene_9, axis=1)

        # obs for random AnnData
        gene_target = {"gene_target": ["NT"] * 100 + ["target_gene_a"] * 200}
        gene_target = pd.DataFrame(gene_target)
        cell_type = {"cell_type": ["CellTypeA"] * 100 + ["CellTypeB"] * 100 + ["CellTypeC"] * 100}
        cell_type = pd.DataFrame(cell_type)
        label = {"label": ["control", "treatment", "treatment"] * 100}
        label = pd.DataFrame(label)

        obs = pd.concat([gene_target, cell_type, label], axis=1)
        obs = obs.set_index(np.arange(300))
        obs.index.rename("index", inplace=True)

        # var for random AnnData
        data = {"name": ["gene_1", "gene_2", "gene_3", "gene_4", "gene_5", "gene_6", "gene_7", "gene_8", "gene_9"]}
        var = pd.DataFrame(data)
        var = var.set_index("name", drop=False)
        var.index.rename("index", inplace=True)
        X = np.concatenate((gene_1, gene_2, gene_3, gene_4, gene_5, gene_6, gene_7, gene_8, gene_9), axis=1)
        X = sparse.csr_matrix(X)
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        return adata

    def test_mixscape(self):
        adata = self.make_test_adata()
        mixscape_identifier = pt.tl.Mixscape()
        mixscape_identifier.mixscape(adata=adata, control="NT", labels="gene_target")
        np_result = adata.obs["mixscape_class_global"] == "NP"
        np_result_correct = np_result[100:200]

        ko_result = adata.obs["mixscape_class_global"] == "KO"
        ko_result_correct = ko_result[200:300]

        assert "mixscape_class" in adata.obs
        assert "mixscape_class_global" in adata.obs
        assert "mixscape_class_p_ko" in adata.obs
        assert sum(np_result_correct) > threshold
        assert sum(ko_result_correct) > threshold

    def test_pert_sign(self):
        adata = self.make_test_adata()

        pt.tl.kernel_pca(adata, n_comps=50)
        mixscape_identifier = pt.tl.Mixscape()
        mixscape_identifier.pert_sign(adata, pert_key="label", control="control", use_rep="X_kpca")

        assert "X_pert" in adata.layers

    def test_lda(self):
        adata = self.make_test_adata()
        mixscape_identifier = pt.tl.Mixscape()
        mixscape_identifier.mixscape(adata=adata, control="NT", labels="gene_target")
        mixscape_identifier.lda(adata=adata, labels="gene_target", n_comps=8)

        assert "mixscape_lda" in adata.uns
