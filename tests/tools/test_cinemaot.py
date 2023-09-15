from pathlib import Path

import numpy as np
import pertpy as pt
import scanpy as sc
from _pytest.fixtures import fixture

CWD = Path(__file__).parent.resolve()


class TestCinemaot:
    @fixture
    def adata(self):
        adata = pt.dt.cinemaot_example()

        return adata

    def test_unweighted(self, adata):
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
        )

        assert "cf" in adata.obsm
        assert "ot" in de.obsm
        assert not np.isnan(np.sum(de.obsm["ot"]))

    def test_weighted(self, adata):
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

        assert "cf" in ad.obsm
        assert "ot" in de.obsm
        assert not np.isnan(np.sum(de.obsm["ot"]))

    def test_pseudobulk(self, adata):
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
        )
        adata_pb = model.generate_pseudobulk(
            adata, de, pert_key="perturbation", control="No stimulation", label_list=["cell_type0528"]
        )

        assert "ptb" in adata_pb.obs
        assert not np.isnan(np.sum(adata_pb.X))
