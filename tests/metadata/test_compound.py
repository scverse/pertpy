import time

import anndata
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from anndata import AnnData
from pubchempy import PubChemHTTPError
from scipy import sparse

NUM_CELLS = 100
NUM_GENES = 100
NUM_CELLS_PER_ID = NUM_CELLS // 4


pt_compound = pt.md.Compound()


@pytest.fixture
def adata() -> AnnData:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((NUM_CELLS, NUM_GENES))
    X = np.where(X < 0, 0, X)

    obs = pd.DataFrame(
        {
            "DepMap_ID": ["ACH-000016", "ACH-000049", "ACH-001208", "ACH-000956"] * NUM_CELLS_PER_ID,
            "perturbation": ["AG-490", "Iniparib", "TAK-901", "Quercetin"] * NUM_CELLS_PER_ID,
        },
        index=[str(i) for i in range(NUM_GENES)],
    )

    var_data = {"gene_name": [f"gene{i}" for i in range(1, NUM_GENES + 1)]}
    var = pd.DataFrame(var_data).set_index("gene_name", drop=False).rename_axis("index")

    X = sparse.csr_matrix(X)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    return adata


def test_compound_annotation(adata):
    retries = 3
    attempt = 0
    while attempt < retries:
        try:
            pt_compound.annotate_compounds(adata=adata, query_id="perturbation")
            assert len(adata.obs.columns) == 5
            pubchemid = [5328779, 9796068, 16124208, 5280343] * NUM_CELLS_PER_ID
            assert pubchemid == list(adata.obs["pubchem_ID"])
            return
        except PubChemHTTPError:
            if attempt == retries - 1:
                pytest.fail("Max retries reached, PubChemHTTPError occurred")
            time.sleep(10)
            attempt += 1
