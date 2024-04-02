import anndata
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from anndata import AnnData


class TestDrug:
    pt_drug = pt.md.Drug()

    @pytest.fixture
    def adata(self) -> AnnData:
        rng = np.random.default_rng()

        gene_names = ["SLC6A2", "SSTR3", "COL1A1", "RPS24", "SSTR2"]
        adata = anndata.AnnData(X=rng.standard_normal(size=(5, 5)), var=pd.DataFrame(index=gene_names))

        return adata

    def test_drug_chembl(self, adata):
        self.pt_drug.annotate(adata=adata)
        assert {"compounds"}.issubset(adata.var.columns)
        assert "CHEMBL1693" in adata.var["compounds"]["SLC6A2"]

    def test_drug_dgidb(self, adata):
        self.pt_drug.annotate(adata=adata, source="dgidb")
        assert {"compounds"}.issubset(adata.var.columns)
        assert "AMITIFADINE" in adata.var["compounds"]["SLC6A2"]

    def test_drug_pharmgkb(self, adata):
        self.pt_drug.annotate(adata=adata, source="pharmgkb")
        assert {"compounds"}.issubset(adata.var.columns)
        assert "3,4-methylenedioxymethamphetamine" in adata.var["compounds"]["SLC6A2"]
