from collections.abc import Sequence
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from pandas.core.api import DataFrame

if find_spec("formulaic_contrasts") is None or find_spec("formulaic") is None:
    pytestmark = pytest.mark.skip(reason="formulaic_contrasts and formulaic not available")

from pertpy.tools._differential_gene_expression import LinearModelBase, TTest


@pytest.fixture
def MockLinearModel():
    class _MockLinearModel(LinearModelBase):
        def _check_counts(self) -> None:
            pass

        def fit(self, **kwargs) -> None:
            pass

        def _test_single_contrast(self, contrast: Sequence[float], **kwargs) -> DataFrame:
            return DataFrame()

    return _MockLinearModel


@pytest.mark.parametrize(
    "formula,cond_kwargs,expected_contrast",
    [
        # single variable
        ["~ condition", {}, [1, 0]],
        ["~ condition", {"condition": "A"}, [1, 0]],
        ["~ condition", {"condition": "B"}, [1, 1]],
        ["~ condition", {"condition": "42"}, ValueError],  # non-existant category
        # no-intercept models
        ["~ 0 + condition", {"condition": "A"}, [1, 0]],
        ["~ 0 + condition", {"condition": "B"}, [0, 1]],
        # Different way of specifying dummy coding
        ["~ donor", {"donor": "D0"}, [1, 0, 0, 0]],
        ["~ C(donor)", {"donor": "D0"}, [1, 0, 0, 0]],
        ["~ C(donor, contr.treatment(base='D2'))", {"donor": "D2"}, [1, 0, 0, 0]],
        ["~ C(donor, contr.treatment(base='D2'))", {"donor": "D0"}, [1, 1, 0, 0]],
        # Handle continuous covariates
        ["~ donor + continuous", {"donor": "D1"}, [1, 1, 0, 0, 0]],
        ["~ donor + np.log1p(continuous)", {"donor": "D1"}, [1, 1, 0, 0, 0]],
        ["~ donor + continuous + np.log1p(continuous)", {"donor": "D0"}, [1, 0, 0, 0, 0, 0]],
        # Nonsense models repeating the same variable, which are nonetheless allowed by formulaic
        ["~ donor + C(donor)", {"donor": "D1"}, [1, 1, 0, 0, 1, 0, 0]],
        ["~ donor + C(donor, contr.treatment(base='D2'))", {"donor": "D0"}, [1, 0, 0, 0, 1, 0, 0]],
        [
            "~ condition + donor + C(donor, contr.treatment(base='D2'))",
            {"condition": "A"},
            ValueError,
        ],  # donor base category can't be resolved because it's ambiguous -> ValueError
        # Sum2zero coding
        ["~ C(donor, contr.sum)", {"donor": "D0"}, [1, 1, 0, 0]],
        ["~ C(donor, contr.sum)", {"donor": "D3"}, [1, -1, -1, -1]],
        # Multiple categorical variables
        ["~ condition + donor", {"condition": "A"}, [1, 0, 0, 0, 0]],
        ["~ condition + donor", {"donor": "D2"}, [1, 0, 0, 1, 0]],
        ["~ condition + donor", {"condition": "B", "donor": "D2"}, [1, 1, 0, 1, 0]],
        ["~ 0 + condition + donor", {"donor": "D1"}, [0, 0, 1, 0, 0]],
        # Interaction terms
        ["~ condition * donor", {"condition": "A"}, [1, 0, 0, 0, 0, 0, 0, 0]],
        ["~ condition + donor + condition:donor", {"condition": "A"}, [1, 0, 0, 0, 0, 0, 0, 0]],
        ["~ condition * donor", {"condition": "B", "donor": "D2"}, [1, 1, 0, 1, 0, 0, 1, 0]],
        ["~ condition * C(donor, contr.treatment(base='D2'))", {"condition": "A"}, [1, 0, 0, 0, 0, 0, 0, 0]],
        [
            "~ condition * C(donor, contr.treatment(base='D2'))",
            {"condition": "B", "donor": "D0"},
            [1, 1, 1, 0, 0, 1, 0, 0],
        ],
        [
            "~ condition:donor",
            {"condition": "A"},
            ValueError,
        ],  # Can't automatically resolve base category, because Formulaic builds a reduced-rank and full-rank factor internally
        ["~ condition:donor", {"condition": "A", "donor": "D1"}, [1, 1, 0, 0, 0, 0, 0, 0]],
        ["~ condition:C(donor)", {"condition": "A", "donor": "D1"}, [1, 1, 0, 0, 0, 0, 0, 0]],
    ],
)
def test_model_cond(test_adata_minimal, MockLinearModel, formula, cond_kwargs, expected_contrast):
    mod = MockLinearModel(test_adata_minimal, formula)
    if isinstance(expected_contrast, type):
        with pytest.raises(expected_contrast):
            mod.cond(**cond_kwargs)
    else:
        actual_contrast = mod.cond(**cond_kwargs)
        assert actual_contrast.tolist() == expected_contrast
        assert actual_contrast.index.tolist() == mod.design.columns.tolist()


def test_test_contrasts_rejects_zero_contrast(MockLinearModel, test_adata_minimal):
    mod = MockLinearModel(test_adata_minimal, "~ condition")
    with pytest.raises(ValueError, match="null space of the design matrix"):
        mod.test_contrasts(np.zeros(2))
    with pytest.raises(ValueError, match="'interaction'"):
        mod.test_contrasts({"interaction": np.zeros(2)})


def test_repr(MockLinearModel, test_adata_minimal):
    mod = MockLinearModel(test_adata_minimal, "~ condition + donor")
    assert repr(mod).splitlines() == [
        "_MockLinearModel",
        "    Data          80 obs × 2 vars",
        "    Layer         X",
        "    Design        1 + condition + donor",
        "    Variables     condition, donor",
        "    Coefficients  Intercept, condition[T.B], donor[T.D1], donor[T.D2], donor[T.D3]",
        "    Fitted        no",
    ]


def test_repr_custom_design(MockLinearModel, test_adata_minimal):
    mod = MockLinearModel(test_adata_minimal, np.ones((test_adata_minimal.n_obs, 1)))
    assert repr(mod).splitlines() == [
        "_MockLinearModel",
        "    Data    80 obs × 2 vars",
        "    Layer   X",
        "    Design  custom matrix (80 × 1)",
        "    Fitted  no",
    ]


def test_repr_without_design(test_adata_minimal):
    assert repr(TTest(test_adata_minimal)).splitlines() == [
        "TTest",
        "    Data   80 obs × 2 vars",
        "    Layer  X",
    ]


def test_repr_truncates_many_coefficients(MockLinearModel, test_adata_minimal):
    mod = MockLinearModel(test_adata_minimal, "~ 0 + pairing")
    assert "… (40 in total)" in repr(mod)


def test_repr_html(MockLinearModel, test_adata_minimal):
    html = MockLinearModel(test_adata_minimal, "~ C(donor, contr.treatment(base='D2'))")._repr_html_()
    assert html.startswith("<div") and html.endswith("</div>")
    assert ">_MockLinearModel</div>" in html
    assert ">80 obs × 2 vars</td>" in html
    assert "base=&#x27;D2&#x27;" in html
    assert "base='D2'" not in html


def test_plot_multicomparison_fc_many_genes(MockLinearModel, test_adata_minimal):
    """Test that plot_multicomparison_fc works even when heatmap hides tick labels.

    Regression test for issue #755.
    When using small figsize or many genes, seaborn heatmap hides xticklabels.
    The old code extracted labels from the rendered plot, causing ValueError.
    The fix calculates positions directly from the DataFrame.
    """
    # Create mock results with many genes to force label hiding
    results = []
    genes = [f"GENE{i}" for i in range(50)]  # 50 genes will force label hiding
    contrasts = ["contrast1", "contrast2"]

    for contrast in contrasts:
        for i, gene in enumerate(genes):
            results.append(
                {
                    "contrast": contrast,
                    "variable": gene,
                    "log_fc": 1.5 + i * 0.05,
                    "adj_p_value": 0.001 if i < 10 else 0.05,
                }
            )

    results_df = pd.DataFrame(results)

    # Create a mock model instance
    mod = MockLinearModel(test_adata_minimal, "~condition")

    # This should not raise ValueError even with small figsize
    # that causes seaborn to hide tick labels
    fig = mod.plot_multicomparison_fc(results_df, figsize=(6, 4), return_fig=True)
    assert fig is not None

    # Also test with heatmap_kwargs that explicitly hide labels
    fig = mod.plot_multicomparison_fc(results_df, xticklabels=False, return_fig=True)
    assert fig is not None


@pytest.mark.parametrize(
    "mean_col,expected_xscale", [("baseMean", "log"), ("logCPM", "linear")], ids=["pydeseq2", "edger"]
)
def test_plot_ma_detects_mean_col(MockLinearModel, test_adata_minimal, mean_col, expected_xscale):
    results_df = pd.DataFrame(
        {
            "variable": [f"GENE{i}" for i in range(4)],
            "log_fc": [2.0, -2.0, 0.1, np.nan],
            "adj_p_value": [0.001, 0.001, 0.5, 0.5],
            mean_col: [10.0, 100.0, 1000.0, 5.0],
        }
    )
    mod = MockLinearModel(test_adata_minimal, "~condition")

    fig = mod.plot_ma(results_df, return_fig=True)
    ax = fig.axes[0]

    assert ax.get_xscale() == expected_xscale
    assert ax.get_xlabel() == mean_col
    # The row with a NaN log fold change is dropped, the two low p-value rows are highlighted.
    assert sum(collection.get_offsets().shape[0] for collection in ax.collections) == 3


def test_plot_ma_without_mean_col(MockLinearModel, test_adata_minimal):
    results_df = pd.DataFrame({"variable": ["GENE0"], "log_fc": [2.0], "adj_p_value": [0.001]})
    mod = MockLinearModel(test_adata_minimal, "~condition")

    with pytest.raises(ValueError, match="Could not find a mean expression column"):
        mod.plot_ma(results_df)
    with pytest.raises(ValueError, match="does not exist"):
        mod.plot_ma(results_df, mean_col="not_found")
