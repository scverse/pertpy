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
    pt_metadata = pt.tl.CellLineMetaData()

    @pytest.fixture
    def adata(self) -> AnnData:
        np.random.seed(1)

        X = np.random.normal(0, 1, (NUM_CELLS, NUM_GENES))
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
        obs["perturbation"] = "Midostaurin"

        var_data = {"gene_name": ["gene" + str(i) for i in range(1, NUM_GENES + 1)]}
        var = pd.DataFrame(var_data)
        var = var.set_index("gene_name", drop=False)
        var.index.rename("index", inplace=True)

        X = sparse.csr_matrix(X)
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        return adata

    def test_cell_line_annotation(self, adata):
        self.pt_metadata.annotate_cell_lines(adata=adata)
        assert (
            len(adata.obs.columns) == len(self.pt_metadata.cell_line_meta.columns) + 1
        )  # due to the perturbation column
        assert set(self.pt_metadata.cell_line_meta.columns).issubset(adata.obs)
        stripped_cell_line_name = (
            ["SLR21"] * NUM_CELLS_PER_ID
            + ["HEKTE"] * NUM_CELLS_PER_ID
            + ["TK10"] * NUM_CELLS_PER_ID
            + ["22RV1"] * NUM_CELLS_PER_ID
        )

        assert stripped_cell_line_name == list(adata.obs["stripped_cell_line_name"])

    def test_gdsc_annotation(self, adata):
        self.pt_metadata.annotate_cell_lines(adata)
        self.pt_metadata.annotate_from_gdsc(adata, query_id="stripped_cell_line_name")
        assert "drug_name" in adata.obs
        assert "ln_ic50" in adata.obs

    def test_protein_expression_annotation(self, adata):
        self.pt_metadata.annotate_cell_lines(adata)
        self.pt_metadata.annotate_protein_expression(adata, query_id="stripped_cell_line_name")

        assert len(adata.obsm) == 1
        assert adata.obsm["proteomics_protein_intensity"].shape == (
            NUM_GENES,
            len(self.pt_metadata.proteomics_data.uniprot_id.unique()),
        )

    def test_bulk_rna_expression_annotation(self, adata):
        self.pt_metadata.annotate_cell_lines(adata)
        self.pt_metadata.annotate_bulk_rna_expression(adata, query_id="DepMap_ID", cell_line_source="broad")

        assert len(adata.obsm) == 1
        assert adata.obsm["bulk_rna_expression_broad"].shape == (
            NUM_GENES,
            self.pt_metadata.bulk_rna_broad.shape[1],
        )

        self.pt_metadata.annotate_bulk_rna_expression(adata, query_id="stripped_cell_line_name")

        assert len(adata.obsm) == 2
        assert adata.obsm["bulk_rna_expression_sanger"].shape == (
            NUM_GENES,
            self.pt_metadata.bulk_rna_sanger.shape[1],
        )
