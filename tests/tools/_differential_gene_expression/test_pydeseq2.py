import numpy.testing as npt
from pertpy.tools._differential_gene_expression import PyDESeq2


def test_pydeseq2_simple(test_adata):
    """Check that the pyDESeq2 method can be

    1. Initialized
    2. Fitted
    3. and that test_contrast returns a DataFrame with the correct number of rows.
    """
    method = PyDESeq2(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(method.contrast("condition", "A", "B"))

    assert len(res_df) == test_adata.n_vars
    # Compare against snapshot
    npt.assert_almost_equal(
        res_df.p_value.values,
        [0.00017, 0.00033, 0.00051, 0.0286, 0.03207, 0.04723, 0.11039, 0.11452, 0.3703, 0.99625],
        decimal=4,
    )
    npt.assert_almost_equal(
        res_df.log_fc.values,
        [0.58207, 0.53855, -0.4121, 0.63281, -0.63283, -0.27066, -0.21271, 0.38601, 0.13434, 0.00146],
        decimal=4,
    )


def test_pydeseq2_complex(test_adata):
    """Check that the pyDESeq2 method can be initialized with a different covariate name and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows.
    """
    test_adata.obs["condition1"] = test_adata.obs["condition"].copy()
    method = PyDESeq2(adata=test_adata, design="~condition1+group")
    method.fit()
    res_df = method.test_contrasts(method.contrast("condition1", "A", "B"))

    assert len(res_df) == test_adata.n_vars
    # Check that the index of the result matches the var_names of the AnnData object
    assert set(test_adata.var_names) == set(res_df["variable"])
    # Compare against snapshot
    npt.assert_almost_equal(
        res_df.p_value.values,
        [7e-05, 0.00012, 0.00035, 0.01062, 0.01906, 0.03892, 0.10755, 0.11175, 0.36631, 0.94952],
        decimal=4,
    )
    npt.assert_almost_equal(
        res_df.log_fc.values,
        [-0.42347, 0.58802, 0.53528, 0.73147, -0.67374, -0.27158, -0.21402, 0.38953, 0.13511, -0.01949],
        decimal=4,
    )


def test_pydeseq2_formula(test_adata):
    """Check that the pyDESeq2 method gives consistent results when specifying contrasts, regardless of the order of covariates"""
    model1 = PyDESeq2(adata=test_adata, design="~condition+group")
    model1.fit()
    res_1 = model1.test_contrasts(model1.contrast("condition", "A", "B"))

    model2 = PyDESeq2(adata=test_adata, design="~group+condition")
    model2.fit()
    res_2 = model2.test_contrasts(model2.contrast("condition", "A", "B"))

    npt.assert_almost_equal(res_2.log_fc.values, res_1.log_fc.values)
