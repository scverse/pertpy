from pertpy.tools._differential_gene_expression import EdgeR


def test_edger_simple(test_adata):
    """Check that the EdgeR method can be

    1. Initialized
    2. Fitted
    3. and that test_contrast returns a DataFrame with the correct number of rows.
    """
    method = EdgeR(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(method.contrast("condition", "A", "B"))

    assert len(res_df) == test_adata.n_vars


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


# TODO: there should be a test checking if, for a concrete example, the output p-values and effect sizes are what
# we expect (-> frozen snapshot, that way we also get a heads-up if something changes upstream)
