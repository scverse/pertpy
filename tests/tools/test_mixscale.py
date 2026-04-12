"""Tests for Mixscape.mixscale continuous perturbation scoring."""

import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

import pertpy as pt


@pytest.fixture
def synthetic_perturbation_adata():
    """Create synthetic perturbation data with known strong/weak effects."""
    np.random.seed(42)
    n_genes = 200

    # 100 NT controls, 100 strong KO, 100 weak KO for GeneA
    # 50 cells for GeneB (moderate effect)
    n_cells = 350

    X = np.random.randn(n_cells, n_genes).astype(np.float32)

    # GeneA strong KO: large effect on first 20 genes
    X[100:200, :20] -= 3.0
    # GeneA weak KO: small effect on first 20 genes
    X[200:300, :20] -= 1.0
    # GeneB moderate: moderate effect on genes 20-40
    X[300:350, 20:40] -= 2.0

    adata = AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    labels = (
        ["NT"] * 100
        + ["GeneA"] * 100
        + ["GeneA"] * 100
        + ["GeneB"] * 50
    )
    adata.obs["gene_target"] = labels
    adata.obs["perturbation"] = [
        "NT" if x == "NT" else "targeting" for x in labels
    ]

    sc.pp.pca(adata)

    return adata


class TestMixscale:
    """Tests for the mixscale method."""

    def test_basic_scoring(self, synthetic_perturbation_adata):
        """Test that mixscale runs and produces scores."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        ms.mixscale(adata, "gene_target", "NT", layer="X_pert")

        assert "mixscale_score" in adata.obs.columns
        assert adata.obs["mixscale_score"].dtype == float

    def test_control_cells_score_zero(self, synthetic_perturbation_adata):
        """Control cells should have score 0."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        ms.mixscale(adata, "gene_target", "NT", layer="X_pert")

        nt_scores = adata.obs.loc[
            adata.obs["gene_target"] == "NT", "mixscale_score"
        ]
        assert (nt_scores == 0).all()

    def test_perturbed_cells_nonzero(self, synthetic_perturbation_adata):
        """Perturbed cells should have non-zero scores."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        ms.mixscale(adata, "gene_target", "NT", layer="X_pert")

        ko_scores = adata.obs.loc[
            adata.obs["gene_target"] == "GeneA", "mixscale_score"
        ]
        assert ko_scores.abs().mean() > 0

    def test_strong_vs_weak_perturbation(self, synthetic_perturbation_adata):
        """Strongly perturbed cells should have higher absolute scores."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        ms.mixscale(adata, "gene_target", "NT", layer="X_pert")

        scores = adata.obs["mixscale_score"].values
        # Cells 100-199 are strong KO, 200-299 are weak KO
        strong_mean = np.abs(scores[100:200]).mean()
        weak_mean = np.abs(scores[200:300]).mean()

        assert strong_mean > weak_mean, (
            f"Strong KO mean ({strong_mean:.2f}) should exceed "
            f"weak KO mean ({weak_mean:.2f})"
        )

    def test_custom_column_name(self, synthetic_perturbation_adata):
        """Test custom output column name."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        ms.mixscale(
            adata,
            "gene_target",
            "NT",
            layer="X_pert",
            new_class_name="my_score",
        )

        assert "my_score" in adata.obs.columns
        assert "mixscale_score" not in adata.obs.columns

    def test_copy_mode(self, synthetic_perturbation_adata):
        """Test that copy=True returns a new object."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        result = ms.mixscale(
            adata,
            "gene_target",
            "NT",
            layer="X_pert",
            copy=True,
        )

        assert result is not None
        assert result is not adata
        assert "mixscale_score" in result.obs.columns

    def test_no_perturbation_signature_raises(
        self, synthetic_perturbation_adata
    ):
        """Should raise KeyError if perturbation_signature hasn't been run."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()

        with pytest.raises(KeyError, match="X_pert"):
            ms.mixscale(adata, "gene_target", "NT")

    def test_multiple_perturbations(self, synthetic_perturbation_adata):
        """Test scoring with multiple perturbation groups."""
        adata = synthetic_perturbation_adata
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        ms.mixscale(adata, "gene_target", "NT", layer="X_pert")

        # Both GeneA and GeneB should have scores
        gene_a_scores = adata.obs.loc[
            adata.obs["gene_target"] == "GeneA", "mixscale_score"
        ]
        gene_b_scores = adata.obs.loc[
            adata.obs["gene_target"] == "GeneB", "mixscale_score"
        ]

        assert gene_a_scores.abs().mean() > 0
        assert gene_b_scores.abs().mean() > 0

    def test_sparse_input(self, synthetic_perturbation_adata):
        """Test that mixscale works with sparse matrices."""
        adata = synthetic_perturbation_adata
        adata.X = csr_matrix(adata.X)
        ms = pt.tl.Mixscape()
        ms.perturbation_signature(adata, "gene_target", "NT")
        ms.mixscale(adata, "gene_target", "NT", layer="X_pert")

        assert "mixscale_score" in adata.obs.columns
        assert not np.isnan(
            adata.obs.loc[
                adata.obs["gene_target"] != "NT", "mixscale_score"
            ]
        ).any()
