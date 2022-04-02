from pathlib import Path

import scanpy as sc

import pertpy as pt

CWD = Path(__file__).parent.resolve()

sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")


def test_load():
    """Test if load function creates anndata objects."""
    ag = pt.tl.Augurpy(estimator="random_forest_classifier")

    loaded_adata = ag.load(sc_sim_adata)
    loaded_df = ag.load(sc_sim_adata.to_df(), meta=sc_sim_adata.obs, cell_type_col="cell_type", label_col="label")

    assert loaded_adata.obs["y_"].equals(loaded_df.obs["y_"]) is True
    assert sc_sim_adata.to_df().equals(loaded_adata.to_df()) is True and sc_sim_adata.to_df().equals(loaded_df.to_df())
