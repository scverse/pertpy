"""Predicts the differential prioritization by performing permutation tests on samples."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from statsmodels.stats.multitest import fdrcorrection


def predict_differential_prioritization(
    augur_results1: dict[str, Any],
    augur_results2: dict[str, Any],
    permuted_results1: dict[str, Any],
    permuted_results2: dict[str, Any],
    n_subsamples: int = 50,
    n_permutations: int = 1000,
) -> DataFrame:
    """Predicts the differential prioritization by performing permutation tests on samples.

    Performs permutation tests that identifies cell types with statistically significant differences in `augur_score`
    between two conditions respectively compared to the control.

    Args:
        augur1: Augurpy results from condition 1, obtained from `predict()[1]`
        augur2: Augurpy results from condition 2, obtained from `predict()[1]`
        permuted1: permuted Augurpy results from condition 1, obtained from `predict()` with argument `augur_mode=permute`
        permuted2: permuted Augurpy results from condition 2, obtained from `predict()` with argument `augur_mode=permute`
        n_subsamples: number of subsamples to pool when calculating the mean augur score fro each permutation; defaults to 50
        n_permutations: the total number of mean augur scores to calculate from a background distribution

    Returns:
        Results object containing mean augur scores.
    """
    # compare available cell types
    cell_types = (
        set(augur_results1["summary_metrics"].columns)
        & set(augur_results2["summary_metrics"].columns)
        & set(permuted_results1["summary_metrics"].columns)
        & set(permuted_results2["summary_metrics"].columns)
    )
    # mean augur scores
    augur_score1 = (
        augur_results1["summary_metrics"]
        .loc["mean_augur_score", cell_types]
        .reset_index()
        .rename(columns={"index": "cell_type"})
    )
    augur_score2 = (
        augur_results2["summary_metrics"]
        .loc["mean_augur_score", cell_types]
        .reset_index()
        .rename(columns={"index": "cell_type"})
    )

    # mean permuted scores over cross validation runs
    permuted_cv_augur1 = (
        permuted_results1["full_results"][permuted_results1["full_results"]["cell_type"].isin(cell_types)]
        .groupby(["cell_type", "idx"], as_index=False)
        .mean()
    )
    permuted_cv_augur2 = (
        permuted_results2["full_results"][permuted_results2["full_results"]["cell_type"].isin(cell_types)]
        .groupby(["cell_type", "idx"], as_index=False)
        .mean()
    )

    sampled_permuted_cv_augur1 = []
    sampled_permuted_cv_augur2 = []

    # draw mean aucs for permute1 and permute2
    for celltype in permuted_cv_augur1["cell_type"].unique():
        df1 = permuted_cv_augur1[permuted_cv_augur1["cell_type"] == celltype]
        df2 = permuted_cv_augur2[permuted_cv_augur1["cell_type"] == celltype]
        for permutation_idx in range(n_permutations):
            # subsample
            sample1 = df1.sample(n=n_subsamples, random_state=permutation_idx, axis="index")
            sampled_permuted_cv_augur1.append(
                pd.DataFrame(
                    {
                        "cell_type": [celltype],
                        "permutation_idx": [permutation_idx],
                        "mean": [sample1["augur_score"].mean(axis=0)],
                        "std": [sample1["augur_score"].std(axis=0)],
                    }
                )
            )

            sample2 = df2.sample(n=n_subsamples, random_state=permutation_idx, axis="index")
            sampled_permuted_cv_augur2.append(
                pd.DataFrame(
                    {
                        "cell_type": [celltype],
                        "permutation_idx": [permutation_idx],
                        "mean": [sample2["augur_score"].mean(axis=0)],
                        "std": [sample2["augur_score"].std(axis=0)],
                    }
                )
            )

    permuted_samples1 = pd.concat(sampled_permuted_cv_augur1)
    permuted_samples2 = pd.concat(sampled_permuted_cv_augur2)

    # delta between augur scores
    delta = augur_score1.merge(augur_score2, on=["cell_type"], suffixes=("1", "2")).assign(
        delta_augur=lambda x: x.mean_augur_score2 - x.mean_augur_score1
    )

    # delta between permutation scores
    delta_rnd = permuted_samples1.merge(
        permuted_samples2, on=["cell_type", "permutation_idx"], suffixes=("1", "2")
    ).assign(delta_rnd=lambda x: x.mean2 - x.mean1)

    # number of values where permutations are larger than test statistic
    delta["b"] = (
        pd.merge(delta_rnd[["cell_type", "delta_rnd"]], delta[["cell_type", "delta_augur"]], on="cell_type", how="left")
        .assign(b=lambda x: (x.delta_rnd >= x.delta_augur))
        .groupby("cell_type", as_index=False)
        .sum()["b"]
    )
    delta["m"] = n_permutations
    delta["z"] = (
        delta["delta_augur"] - delta_rnd.groupby("cell_type", as_index=False).mean()["delta_rnd"]
    ) / delta_rnd.groupby("cell_type", as_index=False).std()["delta_rnd"]
    # calculate pvalues
    delta["pval"] = np.minimum(
        2 * (delta["b"] + 1) / (delta["m"] + 1), 2 * (delta["m"] - delta["b"] + 1) / (delta["m"] + 1)
    )
    delta["padj"] = fdrcorrection(delta["pval"])[1]

    return delta
