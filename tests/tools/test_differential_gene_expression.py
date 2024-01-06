import pandas as pd
import pertpy as pt
import pytest


@pytest.fixture
def dummy_de_results():
    data1 = {"pvals": [0.1, 0.2, 0.3, 0.4], "pvals_adj": [0.1, 0.25, 0.35, 0.45], "logfoldchanges": [1, 2, 3, 4]}
    data2 = {"pvals": [0.1, 0.2, 0.3, 0.4], "pvals_adj": [0.15, 0.2, 0.35, 0.5], "logfoldchanges": [2, 3, 4, 5]}
    de_res_1 = pd.DataFrame(data1)
    de_res_2 = pd.DataFrame(data2)

    return de_res_1, de_res_2


@pytest.fixture
def pt_de():
    pt_de = pt.tl.DifferentialGeneExpression()
    return pt_de


def test_calculate_spearman_correlation(dummy_de_results, pt_de):
    de_res_1, de_res_2 = dummy_de_results

    result = pt_de.calculate_correlation(de_res_1, de_res_2, method="spearman")
    assert result.shape == (1, 2)
    assert all(column in result for column in ["pvals_adj", "logfoldchanges"])


def test_calculate_pearson_correlation(dummy_de_results, pt_de):
    de_res_1, de_res_2 = dummy_de_results

    result = pt_de.calculate_correlation(de_res_1, de_res_2, method="pearson")
    assert result.shape == (1, 2)
    assert all(column in result for column in ["pvals_adj", "logfoldchanges"])


def test_calculate_kendall_tau__correlation(dummy_de_results, pt_de):
    de_res_1, de_res_2 = dummy_de_results

    result = pt_de.calculate_correlation(de_res_1, de_res_2, method="kendall-tau")
    assert result.shape == (1, 2)
    assert all(column in result for column in ["pvals_adj", "logfoldchanges"])


def test_jaccard_index(dummy_de_results, pt_de):
    de_res_1, de_res_2 = dummy_de_results

    jaccard_index = pt_de.calculate_jaccard_index(de_res_1, de_res_2)
    assert 0 <= jaccard_index <= 1


def test_calculate_cohens_d(dummy_de_results, pt_de):
    de_res_1, de_res_2 = dummy_de_results

    cohens_d = pt_de.calculate_cohens_d(de_res_1, de_res_2)
    assert isinstance(cohens_d, float)
