import anndata
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from anndata import AnnData
from scipy import sparse

NUM_CELLS = 100
NUM_GENES = 100
NUM_CELLS_PER_ID = NUM_CELLS // 4


class TestMetaData:
    pt_metadata = pt.tl.CompoundMetaData()

    @pytest.fixture
    def adata(self) -> AnnData:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((NUM_CELLS, NUM_GENES))
        X = np.where(X < 0, 0, X)

        cell_line = {
            "DepMap_ID": ["ACH-000016"] * NUM_CELLS_PER_ID
            + ["ACH-000049"] * NUM_CELLS_PER_ID
            + ["ACH-001208"] * NUM_CELLS_PER_ID
            + ["ACH-000956"] * NUM_CELLS_PER_ID
        }
        cell_line = pd.DataFrame(cell_line)
        obs = pd.concat([cell_line], axis=1)
        obs = obs.set_index(pd.Index([str(i) for i in range(NUM_GENES)]))
        obs.index.rename("index", inplace=True)
        obs["perturbation"] = (
            ["AG-490"] * NUM_CELLS_PER_ID
            + ["Iniparib"] * NUM_CELLS_PER_ID
            + ["TAK-901"] * NUM_CELLS_PER_ID
            + ["Quercetin"] * NUM_CELLS_PER_ID
        )

        var_data = {"gene_name": ["gene" + str(i) for i in range(1, NUM_GENES + 1)]}
        var = pd.DataFrame(var_data)
        var = var.set_index("gene_name", drop=False)
        var.index.rename("index", inplace=True)

        X = sparse.csr_matrix(X)
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        return adata

    def test_compound_annotation(self, adata):
        self.pt_metadata.annotate_compound(adata=adata, query_id="perturbation")
        assert len(adata.obs.columns) == 5
        pubchemid = (
            [5328779] * NUM_CELLS_PER_ID
            + [9796068] * NUM_CELLS_PER_ID
            + [16124208] * NUM_CELLS_PER_ID
            + [5280343] * NUM_CELLS_PER_ID
        )

        assert pubchemid == list(adata.obs["pubchem_ID"])
