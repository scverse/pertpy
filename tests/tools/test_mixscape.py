from pathlib import Path

from anndata import read
from pandas import read_csv

import pertpy as pt

CWD = Path(__file__).parent.resolve()
threshold = 0.8


class TestMixscape:
    def test_mixscape(self):
        adata = read(f"{CWD}/mixscape.h5ad")
        r_result = read_csv(f"{CWD}/r_result.csv", index_col="index", squeeze=True)
        mixscape_identifier = pt.tl.Mixscape()
        mixscape_identifier.mixscape(adata=adata, control="NT", labels="gene_target", layer="X_pert")
        python_result = adata.obs["mixscape_class_global"]

        assert "mixscape_class" in adata.obs
        assert "mixscape_class_global" in adata.obs
        assert "mixscape_class_p_ko" in adata.obs
        assert sum(python_result.eq(r_result)) / len(python_result) > threshold
        assert (
            sum(r_result.loc[r_result == "KO"].eq(python_result)) / len(python_result.loc[python_result == "KO"])
            > threshold
        )
        assert (
            sum(r_result.loc[r_result == "NP"].eq(python_result)) / len(python_result.loc[python_result == "NP"])
            > threshold
        )
        assert (
            sum(r_result.loc[r_result == "NT"].eq(python_result)) / len(python_result.loc[python_result == "NT"])
            > threshold
        )

    def test_pert_sign(self):
        sc_sim_adata = read(f"{CWD}/sc_sim.h5ad")

        pt.tl.kernel_pca(sc_sim_adata, n_comps=50)
        mixscape_identifier = pt.tl.Mixscape()
        mixscape_identifier.pert_sign(sc_sim_adata, pert_key="label", control="control", use_rep="X_kpca")

        assert "X_pert" in sc_sim_adata.layers

    def test_lda(self):
        adata = read(f"{CWD}/mixscape.h5ad")
        mixscape_identifier = pt.tl.Mixscape()
        mixscape_identifier.mixscape(adata=adata, control="NT", labels="gene_target", layer="X_pert")
        mixscape_identifier.lda(adata=adata, labels="gene_target", layer="X_pert")

        assert "mixscape_lda" in adata.uns
