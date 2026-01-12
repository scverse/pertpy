import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

import pertpy as pt


@pytest.fixture
def dummy_adata():
    n_obs = 15
    n_vars = 5
    rng = np.random.default_rng()
    X = rng.random((n_obs, n_vars))
    adata = AnnData(X)
    adata.var_names = [f"gene{i}" for i in range(n_vars)]
    adata.obs["cell_type"] = ["B cell"] * 5 + ["CD8a"] * 5 + ["Mono"] * 5  # celltype column
    adata.obs["conditions"] = ["Healthy"] * 7 + ["Disease"] * 8  # contrast column
    return adata


@pytest.fixture(scope="module")
def hucira():
    return pt.tl.Hucira()


@pytest.fixture(scope="module")
def hcd():
    return pt.dt.human_cytokine_dict()


# Generic test confirming correct output.
def test_compute_ranking_statistic(dummy_adata, hucira):
    contrast_column = "conditions"
    contrasts_combo = [("Healthy", "Disease")]

    ranked_stats, _num_cells = hucira._compute_ranking_statistic(dummy_adata, contrast_column, contrasts_combo)
    assert isinstance(ranked_stats, pd.DataFrame)

    # with pytest.raises(KeyError):
    #    hucira._compute_ranking_statistic(dummy_adata, "wrong_conditions", contrasts_combo)


# Test confirming correct argument format for celltype_combo
def test_run_one_enrichment_test(dummy_adata, hcd, hucira):
    # celltype_combo_correct = ("B cell", "B_cell")
    celltype_combo_wrong = [
        ("B cell", "B_cell"),
        ("CD8a", "CD8_T_cell"),
        ("Mono", "CD14_Mono"),
    ]  # can't be a list for "run_one_enrichment_test()"

    with pytest.raises(ValueError):
        hucira.run_one_enrichment_test(
            dummy_adata, hcd, celltype_combo_wrong, "cell_type", [("Healthy", "Disease")], "conditions", "upregulated"
        )


# Smoke test run
def test_smoke_full_enrichment_test(dummy_adata, hcd, hucira):
    celltype_combo = [
        ("B cell", "B_cell"),
        ("CD8a", "CD8_T_cell"),
        ("Mono", "CD14_Mono"),
    ]  # can't be a list for "run_one_enrichment_test()"
    celltype_column = "cell_type"
    contrasts_combo = [("Healthy", "Disease")]
    contrast_column = "conditions"
    # direction

    all_enrichment_results = hucira.run_all_enrichment_test(
        dummy_adata,
        hcd,
        contrasts_combo,
        celltype_combo,
        contrast_column,
        celltype_column,
        contrasts_combo,
        contrast_column,
    )
    assert isinstance(all_enrichment_results, pd.DataFrame)

    robust_results_dict = hucira.get_robust_significant_results(
        results=all_enrichment_results,
        alphas=[0.1, 0.05, 0.01],
        threshold_valid=0.1,
        threshold_below_alpha=0.9,
    )
    assert isinstance(robust_results_dict, dict)

    cytokine_info = hucira.load_cytokine_info()
    df_senders, df_receivers = hucira.get_all_senders_and_receivers(
        dummy_adata,
        cytokine_info,
        robust_results_dict[contrasts_combo[0]][
            2
        ].cytokine.unique(),  # deep indexing, should change that soon bc not very robust.
        celltype_column,
        sender_pvalue_threshold=0.1,
        receiver_mean_X_threshold=0,
    )

    assert isinstance(df_senders, pd.DataFrame)
    assert isinstance(df_receivers, pd.DataFrame)
