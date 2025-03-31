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


pt_metadata = pt.md.CellLine()


@pytest.fixture
def adata() -> AnnData:
    X = np.random.default_rng().normal(0, 1, (NUM_CELLS, NUM_GENES))

    obs = pd.DataFrame(
        {
            "DepMap_ID": ["ACH-000016", "ACH-000049", "ACH-001208", "ACH-000956"] * NUM_CELLS_PER_ID,
            "perturbation": ["Midostaurin"] * NUM_CELLS_PER_ID * 4,
        },
        index=[str(i) for i in range(NUM_CELLS)],
    )

    var_data = {"gene_name": [f"gene{i}" for i in range(1, NUM_GENES + 1)]}
    var = pd.DataFrame(var_data).set_index("gene_name", drop=False).rename_axis("index")

    X = sparse.csr_matrix(X)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    return adata


def test_cell_line_annotation(adata):
    pt_metadata.annotate(adata=adata)
    assert len(adata.obs.columns) == len(pt_metadata.depmap.columns) + 1  # due to the perturbation column
    stripped_cell_line_name = ["SLR21", "HEKTE", "TK10", "22RV1"] * NUM_CELLS_PER_ID
    assert stripped_cell_line_name == list(adata.obs["StrippedCellLineName"])


def test_gdsc_annotation(adata):
    pt_metadata.annotate(adata)
    pt_metadata.annotate_from_gdsc(adata, query_id="StrippedCellLineName")
    assert "ln_ic50_gdsc" in adata.obs
    assert "auc_gdsc" in adata.obs


def test_prism_annotation(adata):
    adata.obs = pd.DataFrame(
        {
            "DepMap_ID": ["ACH-000879", "ACH-000488", "ACH-000488", "ACH-000008"] * NUM_CELLS_PER_ID,
            "perturbation": ["cytarabine", "cytarabine", "secnidazole", "flutamide"] * NUM_CELLS_PER_ID,
        },
        index=[str(i) for i in range(NUM_CELLS)],
    )

    pt_metadata.annotate(adata)
    pt_metadata.annotate_from_prism(adata, query_id="DepMap_ID")
    assert "ic50_prism" in adata.obs
    assert "ec50_prism" in adata.obs
    assert "auc_prism" in adata.obs


def test_protein_expression_annotation(adata):
    pt_metadata.annotate(adata)
    pt_metadata.annotate_protein_expression(adata, query_id="StrippedCellLineName")

    assert len(adata.obsm) == 1
    assert adata.obsm["proteomics_protein_intensity"].shape == (
        NUM_GENES,
        len(pt_metadata.proteomics.uniprot_id.unique()),
    )


@pytest.mark.slow
def test_bulk_rna_expression_annotation(adata):
    pt_metadata.annotate(adata)
    pt_metadata.annotate_bulk_rna(adata, query_id="DepMap_ID", cell_line_source="broad")

    assert len(adata.obsm) == 1
    assert adata.obsm["bulk_rna_broad"].shape == (
        NUM_GENES,
        pt_metadata.bulk_rna_broad.shape[1],
    )

    pt_metadata.annotate_bulk_rna(adata, query_id="StrippedCellLineName")

    assert len(adata.obsm) == 2
    assert adata.obsm["bulk_rna_sanger"].shape == (
        NUM_GENES,
        pt_metadata.bulk_rna_sanger.shape[1],
    )
