from pathlib import Path

import numpy as np
import pertpy as pt
import scanpy as sc
from _pytest.fixtures import fixture

CWD = Path(__file__).parent.resolve()


@fixture
def adata():
    adata = pt.dt.cinemaot_example()
    adata = sc.pp.subsample(adata, 0.1, copy=True)

    return adata


def test_unweighted(adata):
    sc.pp.pca(adata)
    model = pt.tl.Cinemaot()
    de = model.causaleffect(
        adata,
        pert_key="perturbation",
        control="No stimulation",
        return_matching=True,
        thres=0.5,
        smoothness=1e-5,
        eps=1e-3,
        solver="Sinkhorn",
        preweight_label="cell_type0528",
    )

    eps = 1e-1
    assert "cf" in adata.obsm
    assert "ot" in de.obsm
    assert not np.isnan(np.sum(de.obsm["ot"]))
    assert not np.abs(np.sum(de.obsm["ot"]) - 1) > eps


def test_weighted(adata):
    sc.pp.pca(adata)
    model = pt.tl.Cinemaot()
    ad, de = model.causaleffect_weighted(
        adata,
        pert_key="perturbation",
        control="No stimulation",
        return_matching=True,
        thres=0.5,
        smoothness=1e-5,
        eps=1e-3,
        solver="Sinkhorn",
    )

    eps = 1e-1
    assert "cf" in ad.obsm
    assert "ot" in de.obsm
    assert not np.isnan(np.sum(de.obsm["ot"]))
    assert not np.abs(np.sum(de.obsm["ot"]) - 1) > eps


def test_pseudobulk(adata):
    sc.pp.pca(adata)
    model = pt.tl.Cinemaot()
    de = model.causaleffect(
        adata,
        pert_key="perturbation",
        control="No stimulation",
        return_matching=True,
        thres=0.5,
        smoothness=1e-5,
        eps=1e-3,
        solver="Sinkhorn",
        preweight_label="cell_type0528",
    )
    adata_pb = model.generate_pseudobulk(adata, de, pert_key="perturbation", control="No stimulation", label_list=None)

    expect_num = 9
    eps = 7
    assert "ptb" in adata_pb.obs
    assert not np.isnan(np.sum(adata_pb.X))
    print(adata_pb.shape)
    assert not np.abs(adata_pb.shape[0] - expect_num) >= eps
