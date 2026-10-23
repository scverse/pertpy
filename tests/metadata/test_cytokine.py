import pytest

import pertpy as pt

pt_cytokine = pt.md.Cytokine()


def test_gene_sets():
    up = pt_cytokine.gene_sets("CD14_Mono")
    down = pt_cytokine.gene_sets("CD14_Mono", direction="down")
    assert "CXCL10" in up["IFN-gamma"]
    assert not set(up["IFN-gamma"]) & set(down["IFN-gamma"])


def test_gene_sets_unknown_celltype():
    with pytest.raises(ValueError, match="not a cell type"):
        pt_cytokine.gene_sets("CD14+ Monocytes")
