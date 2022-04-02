from pathlib import Path

import numpy as np
import scanpy as sc

import pertpy.tools
from pertpy.tools._augurpy import Params

CWD = Path(__file__).parent.resolve()

sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")

ag = pertpy.tools.Augurpy("random_forest_classifier", Params(random_state=42))
ag.load(sc_sim_adata)


adata, results1 = ag.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=2)
adata, results2 = ag.predict(sc_sim_adata, n_threads=4, n_subsamples=3, random_state=42)

a, permut1 = ag.predict(sc_sim_adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=2)
a, permut2 = ag.predict(sc_sim_adata, augur_mode="permute", n_threads=4, n_subsamples=100, random_state=42)


def test_predict():
    """Test differential prioritization run."""
    delta = ag.predict_differential_prioritization(results1, results2, permut1, permut2)
    assert not np.isnan(delta["z"]).any()
