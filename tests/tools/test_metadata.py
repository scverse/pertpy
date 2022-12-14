import anndata
import numpy as np
import pandas as pd
from scipy import sparse

import pertpy as pt

# Random generate data settings
num_cells = 100
num_genes = 100
num_cells_per_id = num_cells // 4


class TestMetaData:
    def make_test_adata(self):
        np.random.seed(1)
        # generate count matrix
        np.random.seed(1)
        for i in range(num_genes):
            gene_i = np.random.normal(0, 1, num_cells)
            gene_i = np.where(gene_i < 0, 0, gene_i)
            gene_i = np.expand_dims(gene_i, axis=1)
            if i == 0:
                X = gene_i
            else:
                X = np.concatenate((X, gene_i), axis=1)

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
            ["SLR21"] * num_cells_per_id
            + ["HEKTE"] * num_cells_per_id
            + ["NALM19"] * num_cells_per_id
            + ["JHESOAD1"] * num_cells_per_id
        )
        assert stripped_cell_line_name == list(adata.obs["stripped_cell_line_name"])
