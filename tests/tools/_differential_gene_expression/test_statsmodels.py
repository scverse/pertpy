from importlib.util import find_spec

import numpy as np
import pytest
import statsmodels.api as sm

if find_spec("formulaic_contrasts") is None or find_spec("formulaic") is None:
    pytestmark = pytest.mark.skip(reason="formulaic_contrasts and formulaic not available")


@pytest.mark.parametrize("kwargs", [{}, {"regression_model": sm.GLM, "family": sm.families.NegativeBinomial()}])
def test_statsmodels(test_adata, kwargs):
    """Check that the method can be initialized and fitted, and perform basic checks on the result of test_contrasts."""
    from pertpy.tools._differential_gene_expression import Statsmodels

    method = Statsmodels(adata=test_adata, design="~condition")
    method.fit(**kwargs)
    res_df = method.test_contrasts(np.array([0, 1]))
    # Check that the result has the correct number of rows
    assert len(res_df) == test_adata.n_vars


# TODO: there should be a test checking if, for a concrete example, the output p-values and effect sizes are what
# we expect (-> frozen snapshot, that way we also get a heads-up if something changes upstream)
