from pertpy.tools._differential_gene_expression import PyDESeq2


def test_pydeseq2_simple(test_adata):
    """Check that the pyDESeq2 method can be

    1. Initialized
    2. Fitted
    3. and that test_contrast returns a DataFrame with the correct number of rows.
    """
    method = PyDESeq2(adata=test_adata, design="~condition")
    method.fit()
    res_df = method.test_contrasts(["condition", "A", "B"])

    assert len(res_df) == test_adata.n_vars


def test_pydeseq2_complex(test_adata):
    """Check that the pyDESeq2 method can be initialized with a different covariate name and fitted and that the test_contrast
    method returns a dataframe with the correct number of rows.
    """
    test_adata.obs["condition1"] = test_adata.obs["condition"].copy()
    method = PyDESeq2(adata=test_adata, design="~condition1+group")
    method.fit()
    res_df = method.test_contrasts(["condition1", "A", "B"])

    assert len(res_df) == test_adata.n_vars
    # Check that the index of the result matches the var_names of the AnnData object
    assert set(test_adata.var_names) == set(res_df["variable"])


# TODO: there should be a test checking if, for a concrete example, the output p-values and effect sizes are what
# we expect (-> frozen snapshot, that way we also get a heads-up if something changes upstream)
