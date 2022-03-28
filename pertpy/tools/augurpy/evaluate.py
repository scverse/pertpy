"""Calculates augur score for given dataset and estimator."""
from __future__ import annotations

import random
from collections import defaultdict
from math import floor, nan
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from anndata import AnnData
from joblib import Parallel, delayed
from pandas import DataFrame
from rich import print
from rich.progress import track
from scipy import stats
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    make_scorer,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from skmisc.loess import loess


def cross_validate_subsample(
    adata: AnnData,
    estimator: RandomForestRegressor | RandomForestClassifier | LogisticRegression,
    augur_mode: str,
    subsample_size: int,
    folds: int,
    feature_perc: float,
    subsample_idx: int,
    random_state: int | None,
    zero_division: int | str,
) -> DataFrame:
    """Cross validate subsample anndata object.

    Args:
        adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        estimator: classifier to use in calculating augur metrics, either random forest or logistic regression
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        folds: number of folds to run cross validation on
        subsample_idx: index of the subsample
        random_state: set numpy random seed, sampling seed and fold seed
        zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
            set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

    Returns:
        Results for each cross validation fold.
    """
    subsample = draw_subsample(
        adata,
        augur_mode,
        subsample_size,
        feature_perc=feature_perc,
        categorical=is_classifier(estimator),
        random_state=subsample_idx,
    )
    results = run_cross_validation(
        estimator=estimator,
        subsample=subsample,
        folds=folds,
        subsample_idx=subsample_idx,
        random_state=random_state,
        zero_division=zero_division,
    )
    return results


def sample(adata: AnnData, categorical: bool, subsample_size: int, random_state: int, features: list):
    """Sample anndata observations.

    Args:
        adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        categorical: `True` if target values are categorical
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        random_state: set numpy random seed and sampling seed
        features: features returned Anndata object

    Returns:
        Subsample of anndata object of size subsample_size with given features
    """
    # export subsampling.
    random.seed(random_state)
    if categorical:
        label_subsamples = []
        y_encodings = adata.obs["y_"].unique()
        for code in y_encodings:
            label_subsamples.append(
                sc.pp.subsample(
                    adata[adata.obs["y_"] == code, features],
                    n_obs=subsample_size,
                    copy=True,
                    random_state=random_state,
                )
            )
        subsample = AnnData.concatenate(*label_subsamples, index_unique=None)
    else:
        subsample = sc.pp.subsample(adata[:, features], n_obs=subsample_size, copy=True, random_state=random_state)

    # filter features with 0 variance
    subsample.var["highly_variable"] = False
    subsample.var["means"] = np.ravel(subsample.X.mean(axis=0))
    subsample.var["var"] = np.ravel(subsample.X.power(2).mean(axis=0) - np.power(subsample.X.mean(axis=0), 2))
    # remove all features with 0 variance
    subsample.var.loc[subsample.var["var"] > 0, "highly_variable"] = True

    return subsample[:, subsample.var["highly_variable"]]


def draw_subsample(
    adata: AnnData,
    augur_mode: str,
    subsample_size: int,
    feature_perc: float,
    categorical: bool,
    random_state: int,
) -> AnnData:
    """Subsample and select random features of anndata object.

    Args:
        adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        categorical_data: `True` if target values are categorical
        random_state: set numpy random seed and sampling seed

    Returns:
        Subsample of anndata object of size subsample_size
    """
    random.seed(random_state)
    if augur_mode == "permute":
        # shuffle labels
        adata = adata.copy()
        y_columns = [col for col in adata.obs if col.startswith("y_")]
        adata.obs[y_columns] = adata.obs[y_columns].sample(frac=1, random_state=random_state).values

    if augur_mode == "velocity":
        # no feature selection, assuming this has already happenend in calculating velocity
        features = adata.var_names

    else:
        # randomly sample features from highly variable genes
        highly_variable_genes = adata.var_names[adata.var["highly_variable"]].tolist()
        features = random.sample(highly_variable_genes, floor(len(highly_variable_genes) * feature_perc))

    # randomly sample samples for each label
    return sample(
        adata=adata,
        categorical=categorical,
        subsample_size=subsample_size,
        random_state=random_state,
        features=features,
    )


def ccc_score(y_true, y_pred) -> float:
    """Implementation of Lin's Concordance correlation coefficient, based on https://gitlab.com/-/snippets/1730605.

    Args:
        y_true: array-like of shape (n_samples), ground truth (correct) target values
        y_pred: array-like of shape (n_samples), estimated target values

    Returns:
        Concordance correlation coefficient.
    """
    # covariance between y_true and y_pred
    s_xy = np.cov([y_true, y_pred])[0, 1]
    # means
    x_m = np.mean(y_true)
    y_m = np.mean(y_pred)
    # variances
    s_x_sq = np.var(y_true)
    s_y_sq = np.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2)

    return ccc


def set_scorer(
    estimator: RandomForestRegressor | RandomForestClassifier | LogisticRegression,
    multiclass: bool,
    zero_division: int | str,
) -> dict[str, Any]:
    """Set scoring fuctions for cross-validation based on estimator.

    Args:
        estimator: classifier object used to fit the model used to calculate the area under the curve
        multiclass: `True` if there are more than two target classes
        zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
            set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

    Returns:
        Dict linking name to scorer object and string name
    """
    if multiclass:
        return {
            "augur_score": make_scorer(roc_auc_score, multi_class="ovo", needs_proba=True),
            "auc": make_scorer(roc_auc_score, multi_class="ovo", needs_proba=True),
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="macro", zero_division=zero_division),
            "f1": make_scorer(f1_score, average="macro"),
            "recall": make_scorer(recall_score, average="macro"),
        }
    return (
        {
            "augur_score": make_scorer(roc_auc_score, needs_proba=True),
            "auc": make_scorer(roc_auc_score, needs_proba=True),
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="binary", zero_division=zero_division),
            "f1": make_scorer(f1_score, average="binary"),
            "recall": make_scorer(recall_score, average="binary"),
        }
        if isinstance(estimator, RandomForestClassifier) or isinstance(estimator, LogisticRegression)
        else {
            "augur_score": make_scorer(ccc_score),
            "r2": make_scorer(r2_score),
            "ccc": make_scorer(ccc_score),
            "neg_mean_squared_error": make_scorer(mean_squared_error),
            "explained_variance": make_scorer(explained_variance_score),
        }
    )


def run_cross_validation(
    subsample: AnnData,
    estimator: RandomForestRegressor | RandomForestClassifier | LogisticRegression,
    subsample_idx: int,
    folds: int,
    random_state: int | None,
    zero_division: int | str,
) -> dict:
    """Perform cross validation on given subsample.

    Args:
        subsample: subsample of gene expression matrix of size subsample_size
        estimator: classifier object to use in calculating the area under the curve
        subsample_idx: index of subsample
        folds: number of folds
        random_state: set random fold seed
        zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
            set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

    Returns:
        Dictionary containing prediction metrics and estimator for each fold.
    """
    x = subsample.to_df()
    y = subsample.obs["y_"]
    scorer = set_scorer(estimator, multiclass=True if len(y.unique()) > 2 else False, zero_division=zero_division)
    folds = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

    results = cross_validate(
        estimator=estimator,
        X=x,
        y=y.values.ravel(),
        scoring=scorer,
        cv=folds,
        return_estimator=True,
    )

    results["subsample_idx"] = subsample_idx
    for score in scorer.keys():
        results[f"mean_{score}"] = results[f"test_{score}"].mean()

    # feature importances
    feature_importances = defaultdict(list)
    if isinstance(estimator, RandomForestClassifier) or isinstance(estimator, RandomForestRegressor):
        for fold, estimator in list(zip(range(len(results["estimator"])), results["estimator"])):
            feature_importances["genes"].extend(x.columns.tolist())
            feature_importances["feature_importances"].extend(estimator.feature_importances_.tolist())
            feature_importances["subsample_idx"].extend(len(x.columns) * [subsample_idx])
            feature_importances["fold"].extend(len(x.columns) * [fold])

    # standardized coefficients with Agresti method
    # cf. https://think-lab.github.io/d/205/#3
    if isinstance(estimator, LogisticRegression):
        for fold, estimator in list(zip(range(len(results["estimator"])), results["estimator"])):
            feature_importances["genes"].extend(x.columns.tolist())
            feature_importances["feature_importances"].extend(
                (estimator.coef_ * estimator.coef_.std()).flatten().tolist()
            )
            feature_importances["subsample_idx"].extend(len(x.columns) * [subsample_idx])
            feature_importances["fold"].extend(len(x.columns) * [fold])

    results["feature_importances"] = feature_importances

    return results


def average_metrics(cell_cv_results: list[Any]) -> dict[Any, Any]:
    """Calculate average metric of cross validation runs done of one cell type.

    Args:
        cell_cv_results: list of all cross validation runs of one cell type

    Returns:
        Dict containing the average result for each metric of one cell type
    """
    metric_names = [metric for metric in [*cell_cv_results[0].keys()] if metric.startswith("mean")]
    metric_list: dict[Any, Any] = {}
    for subsample_cv_result in cell_cv_results:
        for metric in metric_names:
            metric_list[metric] = metric_list.get(metric, []) + [subsample_cv_result[metric]]

    return {metric: np.mean(values) for metric, values in metric_list.items()}


def select_highly_variable(adata: AnnData) -> AnnData:
    """Feature selection by variance using scanpy highly variable genes function.

    Args:
        adata: Anndata object containing gene expression values (cells in rows, genes in columns)

    Results:
        Anndata object with highly variable genes added as layer
    """
    min_features_for_selection = 1000

    if len(adata.var_names) - 2 > min_features_for_selection:
        try:
            sc.pp.highly_variable_genes(adata)
        except ValueError:
            print("[bold yellow]Data not normalized. Normalizing now using scanpy log1p normalize.")
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata)

    return adata


def cox_compare(loess1, loess2):
    """Cox compare test on two models.

    Based on: https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.compare_cox.html
    Info: Tests of non-nested hypothesis might not provide unambiguous answers.
    The test should be performed in both directions and it is possible
    that both or neither test rejects.

    Args:
        loess1: fitted loess regression object
        loess2: fitted loess regression object

    Returns:
        t-statistic for the test that including the fitted values of the first model in the second model
        has no effect and two-sided pvalue for the t-statistic
    """
    x = loess1.inputs.x
    z = loess2.inputs.x

    nobs = loess1.inputs.y.shape[0]

    # coxtest
    sigma2_x = np.sum(np.power(loess1.outputs.fitted_residuals, 2)) / nobs
    sigma2_z = np.sum(np.power(loess2.outputs.fitted_residuals, 2)) / nobs
    yhat_x = loess1.outputs.fitted_values
    res_dx = sm.OLS(yhat_x, z).fit()
    err_zx = res_dx.resid
    res_xzx = sm.OLS(err_zx, x).fit()
    err_xzx = res_xzx.resid

    sigma2_zx = sigma2_x + np.dot(err_zx.T, err_zx) / nobs
    c01 = nobs / 2.0 * (np.log(sigma2_z) - np.log(sigma2_zx))
    v01 = sigma2_x * np.dot(err_xzx.T, err_xzx) / sigma2_zx**2
    q = c01 / np.sqrt(v01)
    pval = 2 * stats.norm.sf(np.abs(q))

    return q, pval


def select_variance(adata: AnnData, var_quantile: float, filter_negative_residuals: bool, span: float):
    """Feature selection based on Augur implementation.

    Args:
        adata: Anndata object
        var_quantile: the quantile below which features will be filtered, based on their residuals in a loess model;
            defaults to `0.5`
        filter_negative_residuals: if `True`, filter residuals at a fixed threshold of zero, instead of `var_quantile`
        span: Smoothing factor, as a fraction of the number of points to take into account. Should be in the range
            (0, 1]. Default is 0.75

    Return:
        AnnData object with additional select_variance column in var.
    """
    adata.var["highly_variable"] = False
    adata.var["means"] = np.ravel(adata.X.mean(axis=0))
    adata.var["sds"] = np.ravel(np.sqrt(adata.X.power(2).mean(axis=0) - np.power(adata.X.mean(axis=0), 2)))
    # remove all features with 0 variance
    adata.var.loc[adata.var["sds"] > 0, "highly_variable"] = True
    cvs = adata.var.loc[adata.var["highly_variable"], "means"] / adata.var.loc[adata.var["highly_variable"], "sds"]

    # clip outliers, get best prediction model, get residuals
    lower = np.quantile(cvs, 0.01)
    upper = np.quantile(cvs, 0.99)
    keep = cvs.loc[cvs.between(lower, upper)].index

    cv0 = cvs.loc[keep]
    mean0 = adata.var.loc[keep, "means"]

    if any(mean0 < 0):
        # if there are negative values, don't bother comparing to log-transformed
        # means - fit on normalized data directly
        model = loess(mean0, cv0, span=span)

    else:
        fit1 = loess(mean0, cv0, span=span)
        fit1.fit()
        fit2 = loess(np.log(mean0), cv0, span=span)
        fit2.fit()

        # use a cox test to guess whether we should be fitting the raw or
        # log-transformed means
        cox1 = cox_compare(fit1, fit2)
        cox2 = cox_compare(fit2, fit1)

        #  compare pvalues
        if cox1[1] < cox2[1]:
            model = fit1
        else:
            model = fit2

    residuals = model.outputs.fitted_residuals

    # select features by quantile (or override w/positive residuals)
    genes = keep[residuals > 0] if filter_negative_residuals else keep[residuals > np.quantile(residuals, var_quantile)]

    adata.var["highly_variable"] = [x in genes for x in adata.var.index]

    return adata


def predict(
    adata: AnnData,
    classifier: RandomForestClassifier | RandomForestRegressor | LogisticRegression,
    n_subsamples: int = 50,
    subsample_size: int = 20,
    folds: int = 3,
    min_cells: int = None,
    feature_perc: float = 0.5,
    var_quantile: float = 0.5,
    span: float = 0.75,
    filter_negative_residuals: bool = False,
    n_threads: int = 4,
    show_progress: bool = True,
    augur_mode: Literal["permute"] | Literal["default"] | Literal["velocity"] = "default",
    select_variance_features: bool = False,
    random_state: int | None = None,
    zero_division: int | str = 0,
) -> tuple[AnnData, dict[str, Any]]:
    """Calculates the Area under the Curve using the given classifier.

    Args:
        adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
        classifier: classifier to use in calculating augur metrics, either random forest or logistic regression
        n_subsamples: number of random subsamples to draw from complete dataset for each cell type
        subsample_size: number of cells to subsample randomly per type from each experimental condition
        folds: number of folds to run cross validation on. Be careful changing this parameter without also changing
            `subsample_size`.
        min_cells: minimum number of cells for a particular cell type in each condition in order to retain that type for
            analysis (depricated..)
        feature_perc: proportion of genes that are randomly selected as features for input to the classifier in each
            subsample using the random gene filter
        var_quantile: the quantile below which features will be filtered, based on their residuals in a loess model;
            defaults to `0.5`
        span: Smoothing factor, as a fraction of the number of points to take into account. Should be in the range
            (0, 1]. Default is 0.75
        filter_negative_residuals: if `True`, filter residuals at a fixed threshold of zero, instead of `var_quantile`
        n_threads: number of threads to use for parallelization
        show_progress: if `True` display a progress bar for the analysis with estimated time remaining
        augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
            assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
            while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
            permuting the labels. Note that when setting augur_mode = "permute" n_subsample values less than 100 will be
            set to 500.
        random_state: set numpy random seed, sampling seed and fold seed
        zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
            set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

    Returns:
        A tuple with a dictionary containing the following keys

            * summary_metrics: Pandas Dataframe containing mean metrics for each cell type
            * feature_importances: Pandas Dataframe containing feature importances of genes across all cross validation runs
            * full_results: Dict containing merged results of individual cross validation runs for each cell type
            * [**cell_types]: Cross validation runs of the cell type called

        and the original anndata object with added mean_augur_score metrics in obs.

    """
    if augur_mode == "permute" and n_subsamples < 100:
        n_subsamples = 500
    if is_regressor(classifier) and len(adata.obs["y_"].unique()) <= 3:
        raise ValueError(
            f"[bold red]Regressors cannot be used on {len(adata.obs['label'].unique())} labels. Try a classifier."
        )
    if isinstance(classifier, LogisticRegression) and len(adata.obs["y_"].unique()) > 2:
        raise ValueError(
            "[Bold red]Logistic regression cannot be used for multiclass classification. "
            + "[Bold red]Use a random forest classifier or filter labels in load()."
        )
    if min_cells is None:
        min_cells = n_subsamples
    results: dict[Any, Any] = {
        "summary_metrics": {},
        "feature_importances": defaultdict(list),
        "full_results": defaultdict(list),
    }
    if select_variance_features:
        print("[bold yellow]Set smaller span value in the case of a `segmentation fault` error.")
        print("[bold yellow]Set larger span in case of svddc or other near singularities error.")
    adata.obs["augur_score"] = nan
    for cell_type in track(adata.obs["cell_type"].unique(), description="Processing data."):
        cell_type_subsample = adata[adata.obs["cell_type"] == cell_type].copy()
        if augur_mode == "default" or augur_mode == "permute":
            cell_type_subsample = (
                select_highly_variable(cell_type_subsample)
                if not select_variance_features
                else select_variance(
                    cell_type_subsample,
                    var_quantile=var_quantile,
                    filter_negative_residuals=filter_negative_residuals,
                    span=span,
                )
            )
        if len(cell_type_subsample) < min_cells:
            print(
                f"[bold red]Skipping {cell_type} cell type - {len(cell_type_subsample)} samples is less than min_cells {min_cells}."
            )
        elif (
            cell_type_subsample.obs.groupby(
                ["cell_type", "label"],
            ).y_.count()
            < subsample_size
        ).any():
            print(
                f"[bold red]Skipping {cell_type} cell type - the number of samples for at least one class type is less than "
                f"subsample size {subsample_size}."
            )
        else:
            results[cell_type] = Parallel(n_jobs=n_threads)(
                delayed(cross_validate_subsample)(
                    adata=cell_type_subsample,
                    estimator=classifier,
                    augur_mode=augur_mode,
                    subsample_size=subsample_size,
                    folds=folds,
                    feature_perc=feature_perc,
                    subsample_idx=i,
                    random_state=random_state,
                    zero_division=zero_division,
                )
                for i in range(n_subsamples)
            )
            # summarize scores for cell type
            results["summary_metrics"][cell_type] = average_metrics(results[cell_type])

            # add scores as observation to anndata
            mask = adata.obs["cell_type"].str.startswith(cell_type)
            adata.obs.loc[mask, "augur_score"] = results["summary_metrics"][cell_type]["mean_augur_score"]

            # concatenate feature importances for each subsample cv
            subsample_feature_importances_dicts = [cv["feature_importances"] for cv in results[cell_type]]

            for dictionary in subsample_feature_importances_dicts:
                for key, value in dictionary.items():
                    results["feature_importances"][key].extend(value)
            results["feature_importances"]["cell_type"].extend(
                [cell_type]
                * (len(results["feature_importances"]["genes"]) - len(results["feature_importances"]["cell_type"]))
            )

            for idx, cv in zip(range(n_subsamples), results[cell_type]):
                results["full_results"]["idx"].extend([idx] * folds)
                results["full_results"]["augur_score"].extend(cv["test_augur_score"])
                results["full_results"]["folds"].extend(range(folds))
            results["full_results"]["cell_type"].extend([cell_type] * folds * n_subsamples)
    # make sure one cell type worked
    if len(results) <= 2:
        print("[Bold red]No cells types had more than min_cells needed. Please adjust data or min_cells parameter.")

    results["summary_metrics"] = pd.DataFrame(results["summary_metrics"])
    results["feature_importances"] = pd.DataFrame(results["feature_importances"])
    results["full_results"] = pd.DataFrame(results["full_results"])
    adata.uns["summary_metrics"] = pd.DataFrame(results["summary_metrics"])

    return adata, results
