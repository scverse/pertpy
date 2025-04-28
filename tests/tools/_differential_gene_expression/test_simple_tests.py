import numpy as np
import pandas as pd
import pytest
from pandas.core.api import DataFrame
from pertpy.tools._differential_gene_expression import SimpleComparisonBase, TTest, WilcoxonTest


@pytest.mark.parametrize(
    "paired_by,expected",
    [
        pytest.param(
            None,
            {"gene1": {"p_value": 1.34e-14, "log_fc": -5.14}, "gene2": {"p_value": 0.54, "log_fc": -0.016}},
            id="unpaired",
        ),
        pytest.param(
            "pairing",
            {"gene1": {"p_value": 3.70e-8, "log_fc": -5.14}, "gene2": {"p_value": 0.67, "log_fc": -0.016}},
            id="paired",
        ),
    ],
)
def test_wilcoxon(test_adata_minimal, paired_by, expected):
    """Test that wilcoxon test gives the correct values.

    Reference values have been computed in R using wilcox.test
    """
    res_df = WilcoxonTest.compare_groups(
        adata=test_adata_minimal, column="condition", baseline="A", groups_to_compare="B", paired_by=paired_by
    )
    actual = res_df.loc[:, ["variable", "p_value", "log_fc"]].set_index("variable").to_dict(orient="index")
    for gene in expected:
        assert actual[gene] == pytest.approx(expected[gene], abs=0.02)


@pytest.mark.parametrize(
    "paired_by,expected",
    [
        pytest.param(
            None,
            {"gene1": {"p_value": 2.13e-26, "log_fc": -5.14}, "gene2": {"p_value": 0.96, "log_fc": -0.016}},
            id="unpaired",
        ),
        pytest.param(
            "pairing",
            {"gene1": {"p_value": 1.63e-26, "log_fc": -5.14}, "gene2": {"p_value": 0.85, "log_fc": -0.016}},
            id="paired",
        ),
    ],
)
def test_t(test_adata_minimal, paired_by, expected):
    """Test that t-test gives the correct values.

    Reference values have been computed in R using wilcox.test
    """
    res_df = TTest.compare_groups(
        adata=test_adata_minimal, column="condition", baseline="A", groups_to_compare="B", paired_by=paired_by
    )
    actual = res_df.loc[:, ["variable", "p_value", "log_fc"]].set_index("variable").to_dict(orient="index")
    for gene in expected:
        assert actual[gene] == pytest.approx(expected[gene], abs=0.02)


@pytest.mark.parametrize("seed", range(10))
def test_simple_comparison_pairing(test_adata_minimal, seed):
    """Test that paired samples are properly matched in a paired test"""

    class MockSimpleComparison(SimpleComparisonBase):
        @staticmethod
        def _test():
            return None

        def _compare_single_group(
            self, baseline_idx: np.ndarray, comparison_idx: np.ndarray, *, paired: bool = False, **kwargs
        ) -> DataFrame:
            assert paired
            x0 = self.adata[baseline_idx, :]
            x1 = self.adata[comparison_idx, :]
            assert np.all(x0.obs["condition"] == "A")
            assert np.all(x1.obs["condition"] == "B")
            assert np.all(x0.obs["pairing"].values == x1.obs["pairing"].values)
            return pd.DataFrame([{"p_value": 1}])

    rng = np.random.default_rng(seed)
    shuffle_adata_idx = rng.permutation(test_adata_minimal.obs_names)
    tmp_adata = test_adata_minimal[shuffle_adata_idx, :].copy()

    MockSimpleComparison.compare_groups(
        tmp_adata, column="condition", baseline="A", groups_to_compare=["B"], paired_by="pairing"
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {"column": "donor", "baseline": "D0", "paired_by": "pairing", "groups_to_compare": "D1"},
            id="pairing not subgroup of donor",
        ),
        pytest.param(
            {"column": "donor", "baseline": "D0", "paired_by": "condition", "groups_to_compare": "D1"},
            id="more than two per group (donor)",
        ),
        pytest.param(
            {"column": "condition", "baseline": "A", "paired_by": "donor", "groups_to_compare": "B"},
            id="more than two per group (condition)",
        ),
    ],
)
def test_invalid_pairing(test_adata_minimal, params):
    """Test that the SimpleComparisonBase class raises an error when paired analysis is requested with invalid configuration."""
    with pytest.raises(ValueError):
        TTest.compare_groups(test_adata_minimal, **params)
