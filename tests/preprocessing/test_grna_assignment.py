import anndata as ad
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from jax import random


class TestGuideRnaProcessingAndPlotting:
    @pytest.fixture
    def adata_simple(self):
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

    def test_grna_threshold_assignment(self, adata_simple):
        threshold = 5
        output_layer = "assigned_guides"
        assert output_layer not in adata_simple.layers

        ga = pt.pp.GuideAssignment()
        ga.assign_by_threshold(adata_simple, assignment_threshold=threshold, output_layer=output_layer)
        assert output_layer in adata_simple.layers
        assert np.all(np.logical_xor(adata_simple.X < threshold, adata_simple.layers[output_layer].toarray() == 1))

    def test_grna_max_assignment(self, adata_simple):
        threshold = 5
        output_key = "assigned_guide"
        assert output_key not in adata_simple.obs

        ga = pt.pp.GuideAssignment()
        ga.assign_to_max_guide(adata_simple, assignment_threshold=threshold, output_key=output_key)
        assert output_key in adata_simple.obs
        assert tuple(adata_simple.obs[output_key]) == tuple(
            [f"guide_{i}" if i > 0 else "Negative" for i in [1, 4, 6, 0, 6, 1, 7, 1]]
        )

    def test_grna_mixture_model(self, adata_simple):
        output_key = "assigned_guide"
        assert output_key not in adata_simple.obs

        ga = pt.pp.GuideAssignment()
        ga.assign_mixture_model(adata_simple)
        assert output_key in adata_simple.obs
        target = [f"guide_{i}" if i > 0 else "negative" for i in [1, 4, 6, 0, 6, 1, 7, 1, 0]]
        assert all(t in g for t, g in zip(target, adata_simple.obs[output_key], strict=False))
