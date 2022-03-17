from pathlib import Path

import numpy as np
import scanpy as sc

from augurpy.differential_prioritization import predict_differential_prioritization
from augurpy.estimator import Params, create_estimator
from augurpy.evaluate import predict
from augurpy.read_load import load

CWD = Path(__file__).parent.resolve()

sc_sim_adata = sc.read_h5ad(f"{CWD}/sc_sim.h5ad")
sc_sim_adata = load(sc_sim_adata)

# estimators
rf_classifier = create_estimator("random_forest_classifier", Params(random_state=42))

adata, results1 = predict(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=rf_classifier, random_state=2)
adata, results2 = predict(sc_sim_adata, n_threads=4, n_subsamples=3, classifier=rf_classifier, random_state=42)

a, permut1 = predict(
    sc_sim_adata, augur_mode="permute", n_threads=4, n_subsamples=100, classifier=rf_classifier, random_state=2
)
a, permut2 = predict(
    sc_sim_adata, augur_mode="permute", n_threads=4, n_subsamples=100, classifier=rf_classifier, random_state=42
)


def test_predict():
    """Test differential prioritization run."""
    delta = predict_differential_prioritization(results1, results2, permut1, permut2)
    assert not np.isnan(delta["z"]).any()
