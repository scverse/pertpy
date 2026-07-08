import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

NUM_CELLS_PER_GROUP = 10
NUM_NOT_DE = 10
NUM_DE = 10


@pytest.fixture
def adata():
    """Synthetic screen with NT controls and a target gene split into non-perturbed and knocked-out cells."""
    rng = np.random.default_rng(seed=1)
    columns = []
    # genes that are not differentially expressed between any group
    for _ in range(NUM_NOT_DE):
        nt = np.clip(rng.normal(0, 1, NUM_CELLS_PER_GROUP), 0, None)
        non_perturbed = np.clip(rng.normal(0, 1, NUM_CELLS_PER_GROUP), 0, None)
        knockout = np.clip(rng.normal(0, 1, NUM_CELLS_PER_GROUP), 0, None)
        columns.append(np.concatenate((nt, non_perturbed, knockout))[:, None])

    # genes that are differentially expressed only in the knocked-out cells
    for i in range(NUM_DE):
        nt = np.clip(rng.normal(i + 2, 0.5 + 0.05 * i, NUM_CELLS_PER_GROUP), 0, None)
        non_perturbed = np.clip(rng.normal(i + 2, 0.5 + 0.05 * i, NUM_CELLS_PER_GROUP), 0, None)
        knockout = np.clip(rng.normal(i + 4, 0.5 + 0.1 * i, NUM_CELLS_PER_GROUP), 0, None)
        columns.append(np.concatenate((nt, non_perturbed, knockout))[:, None])

    X = np.concatenate(columns, axis=1)

    obs = pd.DataFrame(
        {
            "gene_target": ["NT"] * NUM_CELLS_PER_GROUP + ["target_gene_a"] * NUM_CELLS_PER_GROUP * 2,
            "label": ["control"] * NUM_CELLS_PER_GROUP + ["treatment"] * NUM_CELLS_PER_GROUP * 2,
        },
        index=np.arange(NUM_CELLS_PER_GROUP * 3).astype(str),
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(1, NUM_NOT_DE + NUM_DE + 1)])
    return ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
