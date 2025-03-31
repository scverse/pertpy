import numpy.testing as npt
from pertpy.tools._differential_gene_expression import EdgeR, PyDESeq2


def test_edger_simple(test_adata):
    """Check that the EdgeR method can be

    1. Initialized
    2. Fitted
    3. That test_contrast returns a DataFrame with the correct number of rows
    """
    method = EdgeR(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(method.contrast("condition", "A", "B"))

    assert len(res_df) == test_adata.n_vars
    # Compare against snapshot
    npt.assert_almost_equal(
        res_df.p_value.values,
        [
            8.0000e-05,
            1.8000e-04,
            5.3000e-04,
            1.1800e-03,
            3.3800e-02,
            3.3820e-02,
            7.7980e-02,
            1.3715e-01,
            2.5052e-01,
            9.2485e-01,
        ],
        decimal=4,
    )
    npt.assert_almost_equal(
        res_df.log_fc.values,
        [0.61208, -0.39374, 0.57944, 0.7343, -0.58675, 0.42575, -0.23951, -0.20761, 0.17489, 0.0247],
        decimal=4,
    )


def test_edger_complex(test_adata):
    """Check that the EdgeR method can be initialized with a different covariate name and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows.
    """
    test_adata.obs["condition1"] = test_adata.obs["condition"].copy()
    method = EdgeR(adata=test_adata, design="~condition1+group")
    method.fit()
    res_df = method.test_contrasts(method.contrast("condition1", "A", "B"))

    assert len(res_df) == test_adata.n_vars
    # Check that the index of the result matches the var_names of the AnnData object
    assert set(test_adata.var_names) == set(res_df["variable"])

    # Compare ranking of genes from a different method (without design matrix handling)
    down_gene = res_df.set_index("variable").loc["gene3", "log_fc"]
    up_gene = res_df.set_index("variable").loc["gene1", "log_fc"]
    assert down_gene < up_gene

    method = PyDESeq2(adata=test_adata, design="~condition1+group")
    method.fit()
    deseq_res_df = method.test_contrasts(method.contrast("condition1", "A", "B"))
    assert all(res_df.sort_values("log_fc")["variable"].values == deseq_res_df.sort_values("log_fc")["variable"].values)
