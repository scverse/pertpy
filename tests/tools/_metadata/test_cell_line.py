import anndata
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

import pertpy as pt

NUM_CELLS = 100
NUM_GENES = 100
NUM_CELLS_PER_ID = NUM_CELLS // 4


class TestMetaData:
    def make_test_adata(self) -> AnnData:
        np.random.seed(1)

        X = np.random.normal(0, 1, (NUM_CELLS, NUM_GENES))
        X = np.where(X < 0, 0, X)

        cell_line = {
            "DepMap_ID": ["ACH-000016"] * NUM_CELLS_PER_ID
            + ["ACH-000049"] * NUM_CELLS_PER_ID
            + ["ACH-000130"] * NUM_CELLS_PER_ID
            + ["ACH-000216"] * NUM_CELLS_PER_ID
        }
        cell_line = pd.DataFrame(cell_line)
        obs = pd.concat([cell_line], axis=1)
        obs = obs.set_index(np.arange(NUM_GENES))
        obs.index.rename("index", inplace=True)

        var_data = {"gene_name": ["gene" + str(i) for i in range(1, NUM_GENES + 1)]}
        var = pd.DataFrame(var_data)
        var = var.set_index("gene_name", drop=False)
        var.index.rename("index", inplace=True)

        X = sparse.csr_matrix(X)
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        return adata

    def test_cell_line_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.CellLineMetaData()
        pt_metadata.annotate_cell_lines(adata=adata)

        assert len(adata.obs.columns) == len(pt_metadata.cell_line_meta.columns)
        assert set(pt_metadata.cell_line_meta.columns).issubset(adata.obs)
        stripped_cell_line_name = (
            ["SLR21"] * NUM_CELLS_PER_ID
            + ["HEKTE"] * NUM_CELLS_PER_ID
            + ["NALM19"] * NUM_CELLS_PER_ID
            + ["JHESOAD1"] * NUM_CELLS_PER_ID
        )

        assert stripped_cell_line_name == list(adata.obs["stripped_cell_line_name"])

    def test_ccle_expression_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.CellLineMetaData()
        pt_metadata.annotate_cell_lines(adata)
        pt_metadata.annotate_ccle_expression(adata)

        assert len(adata.obsm) == 1
        assert adata.obsm["CCLE_expression"].shape == (NUM_CELLS, len(pt_metadata.ccle_expr.columns))

    def test_protein_expression_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.CellLineMetaData()
        pt_metadata.annotate_cell_lines(adata)
        pt_metadata.annotate_protein_expression(adata)

        assert len(adata.obsm) == 1
        assert adata.obsm["proteomics_protein_intensity"].shape == (
            NUM_GENES,
            len(pt_metadata.proteomics_data.uniprot_id.unique()),
        )

    def test_bulk_rna_expression_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.CellLineMetaData()
        pt_metadata.annotate_cell_lines(adata)
        pt_metadata.annotate_bulk_rna_expression(adata)

        assert len(adata.obsm) == 1
        assert adata.obsm["bulk_rna_expression_broad"].shape == (
            NUM_GENES,
            len(pt_metadata.bulk_rna_broad.gene_id.unique()),
        )

        pt_metadata.annotate_bulk_rna_expression(adata, bulk_rna_source="sanger")

        assert len(adata.obsm) == 2
        assert adata.obsm["bulk_rna_expression_sanger"].shape == (
            NUM_GENES,
            len(pt_metadata.bulk_rna_sanger.gene_id.unique()),
        )
