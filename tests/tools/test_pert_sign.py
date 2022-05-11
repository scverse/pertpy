from pathlib import Path

from anndata import read

import pertpy as pt

CWD = Path(__file__).parent.resolve()


def test_pert_sign():
    sc_sim_adata = read(f"{CWD}/sc_sim.h5ad")

    pt.tl.kernel_pca(sc_sim_adata, n_comps=50)

    pt.tl.pert_sign(sc_sim_adata, pert_key="label", control="control", use_rep="X_kpca")

    assert "X_pert" in sc_sim_adata.layers
