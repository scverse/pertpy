import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt
import pytest


class TestGuideRnaProcessingAndPlotting:
    @pytest.fixture
    def adata(self):
        exp_matrix = np.array(
            [
                [9, 0, 1, 0, 1, 0, 0],
                [1, 5, 1, 7, 0, 0, 0],
                [2, 0, 1, 0, 0, 8, 0],
                [1, 1, 1, 0, 1, 1, 1],
                [0, 0, 1, 0, 0, 5, 0],
                [9, 0, 1, 7, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 6],
                [8, 0, 1, 0, 0, 0, 0],
            ]
        ).astype(np.float32)
        adata = ad.AnnData(
            exp_matrix,
            obs=pd.DataFrame(index=[f"cell_{i+1}" for i in range(exp_matrix.shape[0])]),
            var=pd.DataFrame(index=[f"guide_{i+1}" for i in range(exp_matrix.shape[1])]),
        )
        return adata

    def test_grna_threshold_assignment(self, adata):
        threshold = 5
        output_layer = "assigned_guides"
        assert output_layer not in adata.layers

        ga = pt.pp.GuideAssignment()
        ga.assign_by_threshold(adata, assignment_threshold=threshold, output_layer=output_layer)
        assert output_layer in adata.layers
        assert np.all(np.logical_xor(adata.X < threshold, adata.layers[output_layer].toarray() == 1))

    def test_grna_max_assignment(self, adata):
        threshold = 5
        output_key = "assigned_guide"
        assert output_key not in adata.obs

        ga = pt.pp.GuideAssignment()
        ga.assign_to_max_guide(adata, assignment_threshold=threshold, output_key=output_key)
        assert output_key in adata.obs
        assert tuple(adata.obs[output_key]) == tuple(
            [f"guide_{i}" if i > 0 else "NT" for i in [1, 4, 6, 0, 6, 1, 7, 1]]
        )
