import anndata
import numpy as np
import pandas as pd
from scipy import sparse

import pertpy as pt

# Generate data settings
num_cells = 100
num_genes = 100
num_cells_per_id = num_cells // 4


class TestMetaData:
    def make_test_adata(self):
        np.random.seed(1)
        # generate count matrix
        X = np.random.normal(0, 1, (num_cells, num_genes))
        X = np.where(X < 0, 0, X)

        # obs for random AnnData, which contains the cell line information DepMap_ID
        cell_line = {
            "DepMap_ID": ["ACH-000016"] * num_cells_per_id
            + ["ACH-000049"] * num_cells_per_id
            + ["ACH-000130"] * num_cells_per_id
            + ["ACH-000216"] * num_cells_per_id
        }
        cell_line = pd.DataFrame(cell_line)
        obs = pd.concat([cell_line], axis=1)
        obs = obs.set_index(np.arange(num_cells))
        obs.index.rename("index", inplace=True)

        # var for random AnnData
        var_data = {"gene_name": ["gene" + str(i) for i in range(1, num_genes + 1)]}
        var = pd.DataFrame(var_data)
        var = var.set_index("gene_name", drop=False)
        var.index.rename("index", inplace=True)

        X = sparse.csr_matrix(X)
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        return adata

    def test_cell_line_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.MetaData()
        pt_metadata.annotate_cell_lines(adata=adata)
        assert len(adata.obs.columns) == len(pt_metadata.cell_line_meta.columns)
        assert set(pt_metadata.cell_line_meta.columns).issubset(adata.obs)
        stripped_cell_line_name = (
            ["SW579"] * num_cells_per_id
            + ["BT474"] * num_cells_per_id
            + ["NALM19"] * num_cells_per_id
            + ["TE-6"] * num_cells_per_id
        )
        assert stripped_cell_line_name == list(adata.obs["stripped_cell_line_name"])

    def test_ccle_expression_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.MetaData()
        pt_metadata.annotate_cell_lines(adata)
        pt_metadata.annotate_ccle_expression(adata)
        assert len(adata.obsm) == 1
        assert adata.obsm["CCLE_expression"].shape == (num_cells, len(pt_metadata.ccle_expr.columns))

    def test_protein_expression_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.MetaData()
        pt_metadata.annotate_cell_lines(adata)
        pt_metadata.annotate_protein_expression(adata)
        assert len(adata.obsm) == 1
        assert adata.obsm["proteomics_protein_intensity"].shape == (
            num_cells,
            len(pt_metadata.proteomics_data.uniprot_id.unique()),
        )

    def test_bulk_rna_expression_annotation(self):
        adata = self.make_test_adata()
        pt_metadata = pt.tl.MetaData()
        pt_metadata.annotate_cell_lines(adata)
        pt_metadata.annotate_bulk_rna_expression(adata)
        assert len(adata.obsm) == 1
        assert adata.obsm["bulk_rna_expression_broad"].shape == (
            num_cells,
            len(pt_metadata.bulk_rna_broad.gene_id.unique()),
        )

        pt_metadata.annotate_bulk_rna_expression(adata, bulk_rna_source="sanger")
        assert len(adata.obsm) == 2
        assert adata.obsm["bulk_rna_expression_sanger"].shape == (
            num_cells,
            len(pt_metadata.bulk_rna_sanger.gene_id.unique()),
        )
