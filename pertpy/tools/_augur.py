from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from math import floor, nan
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from lamin_utils import logger
from rich.progress import track
from scipy import sparse, stats
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    make_scorer,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from skmisc.loess import loess
from statsmodels.api import OLS
from statsmodels.stats.multitest import fdrcorrection

from pertpy._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Augur:
    """Python implementation of Augur."""

    def __init__(
        self,
        estimator: Literal["random_forest_classifier", "random_forest_regressor", "logistic_regression_classifier"],
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: Literal["auto", "log2", "sqrt"] | int | float = 2,
        penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
        random_state: int | None = None,
    ):
        """Defines the Augur estimator model and parameters.

        estimator: The scikit-learn estimator model that classifies.
        n_estimators: Number of trees in the forest.
        max_depth: Maximal depth of each tree.
        max_features: Maximal number of features considered when looking at best split.

            * if int then consider max_features for each split
            * if float consider round(max_features*n_features)
            * if `auto` then max_features=n_features (default)
            * if `log2` then max_features=log2(n_features)
            * if `sqrt` then max_featuers=sqrt(n_features)

        penalty: Norm of the penalty used in logistic regression

            * if `l1` then L1 penalty is added
            * if `l2` then L2 penalty is added (default)
            * if `elasticnet` both L1 and L2 penalties are added
            * if `none` no penalty is added
        """
        self.estimator = self.create_estimator(
            classifier=estimator,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            penalty=penalty,
            random_state=random_state,
        )

    def load(
        self,
        input: AnnData | pd.DataFrame,
        *,
        meta: pd.DataFrame | None = None,
        label_col: str = "label_col",
        cell_type_col: str = "cell_type_col",
        condition_label: str | None = None,
        treatment_label: str | None = None,
    ) -> AnnData:
        """Loads the input data.

        Args:
            input: Anndata or matrix containing gene expression values (genes in rows, cells in columns)
                and optionally meta data about each cell.
            meta: Optional Pandas DataFrame containing meta data about each cell.
            label_col: column of the meta DataFrame or the Anndata or matrix containing the condition labels for each cell
                in the cell-by-gene expression matrix
            cell_type_col: column of the meta DataFrame or the Anndata or matrix containing the cell type labels for each
                cell in the cell-by-gene expression matrix
            condition_label: in the case of more than two labels, this label is used in the analysis
            treatment_label: in the case of more than two labels, this label is used in the analysis

        Returns:
            Anndata object containing gene expression values (cells in rows, genes in columns)
            and cell type, label and y dummy variables as obs

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
        """
        if isinstance(input, AnnData):
            input.obs = input.obs.rename(columns={cell_type_col: "cell_type", label_col: "label"})
            adata = input

        elif isinstance(input, pd.DataFrame):
            if meta is None:
                try:
                    _ = input[cell_type_col]
                    _ = input[label_col]
                except KeyError:
                    logger.error("No column names matching cell_type_col and label_col.")

            label = input[label_col] if meta is None else meta[label_col]
            cell_type = input[cell_type_col] if meta is None else meta[cell_type_col]
            x = input.drop([label_col, cell_type_col], axis=1) if meta is None else input
            adata = AnnData(X=x, obs=pd.DataFrame({"cell_type": cell_type, "label": label}))

        if len(adata.obs["label"].unique()) < 2:
            raise ValueError("Less than two unique labels in dataset. At least two are needed for the analysis.")
        # dummy variables for categorical data
        if adata.obs["label"].dtype.name == "category":
            # filter samples according to label
            if condition_label is not None and treatment_label is not None:
                logger.info(f"Filtering samples with {condition_label} and {treatment_label} labels.")
                adata = ad.concat(
                    [adata[adata.obs["label"] == condition_label], adata[adata.obs["label"] == treatment_label]]
                )
            label_encoder = LabelEncoder()
            adata.obs["y_"] = label_encoder.fit_transform(adata.obs["label"])
        else:
            y = adata.obs["label"].to_frame()
            y = y.rename(columns={"label": "y_"})
            adata.obs = pd.concat([adata.obs, y], axis=1)

        return adata

    def create_estimator(
        self,
        classifier: (Literal["random_forest_classifier", "random_forest_regressor", "logistic_regression_classifier"]),
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: Literal["auto", "log2", "sqrt"] | int | float = 2,
        penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
        random_state: int | None = None,
    ) -> RandomForestClassifier | RandomForestRegressor | LogisticRegression:
        """Creates a model object of the provided type and populates it with desired parameters.

        Args:
            classifier: classifier to use in calculating the area under the curve.
                        Either random forest classifier or logistic regression for categorical data
                        or random forest regressor for continous data
            n_estimators: Number of trees in the forest.
            max_depth: Maximal depth of each tree.
            max_features: Maximal number of features considered when looking at best split.

                * if int then consider max_features for each split
                * if float consider round(max_features*n_features)
                * if `auto` then max_features=n_features (default)
                * if `log2` then max_features=log2(n_features)
                * if `sqrt` then max_featuers=sqrt(n_features)

            penalty: Norm of the penalty used in logistic regression

                * if `l1` then L1 penalty is added
                * if `l2` then L2 penalty is added (default)
                * if `elasticnet` both L1 and L2 penalties are added
                * if `none` no penalty is added

            random_state: Random model seed.

        Examples:
            >>> import pertpy as pt
            >>> augur = pt.tl.Augur("random_forest_classifier")
            >>> estimator = augur.create_estimator("logistic_regression_classifier")
        """
        if classifier == "random_forest_classifier":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_state,
            )
        elif classifier == "random_forest_regressor":
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_state,
            )
        elif classifier == "logistic_regression_classifier":
            return LogisticRegression(penalty=penalty, random_state=random_state)
        else:
            raise ValueError("Invalid classifier")

    def sample(self, adata: AnnData, categorical: bool, subsample_size: int, random_state: int, features: list):
        """Sample AnnData observations.

        Args:
            adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
            categorical: `True` if target values are categorical
            subsample_size: number of cells to subsample randomly per type from each experimental condition
            random_state: set numpy random seed and sampling seed
            features: features returned Anndata object

        Returns:
            Subsample of AnnData object of size subsample_size with given features

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_highly_variable(loaded_data)
            >>> features = loaded_data.var_names
            >>> subsample = ag_rfc.sample(
            ...     loaded_data, categorical=True, subsample_size=20, random_state=42, features=loaded_data.var_names
            ... )
        """
        # export subsampling.
        random.seed(random_state)
        if categorical:
            label_subsamples = []
            y_encodings = adata.obs["y_"].unique()
            label_subsamples = [
                sc.pp.subsample(
                    adata[adata.obs["y_"] == code, features],
                    n_obs=subsample_size,
                    copy=True,
                    random_state=random_state,
                )
                for code in y_encodings
            ]

            subsample = ad.concat([*label_subsamples], index_unique=None)
        else:
            subsample = sc.pp.subsample(adata[:, features], n_obs=subsample_size, copy=True, random_state=random_state)

        # filter features with 0 variance
        subsample.var["highly_variable"] = False
        subsample.var["means"] = np.ravel(subsample.X.mean(axis=0))
        # Converting because the Syntax for power for numpy arrays is different -> use sparse Syntax as default
        if isinstance(subsample.X, np.ndarray):
            subsample.X = sparse.csc_matrix(subsample.X)
        subsample.var["var"] = np.ravel(subsample.X.power(2).mean(axis=0) - np.power(subsample.X.mean(axis=0), 2))
        # remove all features with 0 variance
        subsample.var.loc[subsample.var["var"] > 0, "highly_variable"] = True

        return subsample[:, subsample.var["highly_variable"]]

    def draw_subsample(
        self,
        adata: AnnData,
        *,
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
            feature_perc: proportion of genes that are randomly selected as features for input to the classifier in each
                          subsample using the random gene filter
            categorical: `True` if target values are categorical
            random_state: set numpy random seed and sampling seed

        Returns:
            Subsample of anndata object of size subsample_size

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_highly_variable(loaded_data)
            >>> subsample = ag_rfc.draw_subsample(adata, augur_mode="default", subsample_size=20, feature_perc=0.5, \
                categorical=True, random_state=42)
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
        return self.sample(
            adata=adata,
            categorical=categorical,
            subsample_size=subsample_size,
            random_state=random_state,
            features=features,
        )

    def cross_validate_subsample(
        self,
        adata: AnnData,
        *,
        augur_mode: str,
        subsample_size: int,
        folds: int,
        feature_perc: float,
        subsample_idx: int,
        random_state: int | None,
        zero_division: int | str,
    ) -> dict:
        """Cross validate subsample anndata object.

        Args:
            adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
            augur_mode: one of default, velocity or permute. Setting augur_mode = "velocity" disables feature selection,
                assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
                while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
                permuting the labels
            subsample_size: number of cells to subsample randomly per type from each experimental condition
            folds: number of folds to run cross validation on
            feature_perc: proportion of genes that are randomly selected as features for input to the classifier in each
                          subsample using the random gene filter
            subsample_idx: index of the subsample
            random_state: set numpy random seed, sampling seed and fold seed
            zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
                set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

        Returns:
            Results for each cross validation fold.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_highly_variable(loaded_data)
            >>> results = ag_rfc.cross_validate_subsample(loaded_data, augur_mode="default", subsample_size=20, \
                folds=3, feature_perc=0.5, subsample_idx=0, random_state=42, zero_division=0)
        """
        subsample = self.draw_subsample(
            adata,
            augur_mode=augur_mode,
            subsample_size=subsample_size,
            feature_perc=feature_perc,
            categorical=is_classifier(self.estimator),
            random_state=subsample_idx,
        )
        results = self.run_cross_validation(
            subsample=subsample,
            folds=folds,
            subsample_idx=subsample_idx,
            random_state=random_state,
            zero_division=zero_division,
        )
        return results

    def ccc_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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
        self,
        multiclass: bool,
        zero_division: int | str,
    ) -> dict[str, Any]:
        """Set scoring fuctions for cross-validation based on estimator.

        Args:
            multiclass: Whether there are more than two target classes
            zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
                set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

        Returns:
            Dict linking name to scorer object and string name

        Examples:
            >>> import pertpy as pt
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> scorer = ag_rfc.set_scorer(True, 0)
        """
        if multiclass:
            return {
                "augur_score": make_scorer(roc_auc_score, multi_class="ovo", response_method="predict_proba"),
                "auc": make_scorer(roc_auc_score, multi_class="ovo", response_method="predict_proba"),
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score, average="macro", zero_division=zero_division),
                "f1": make_scorer(f1_score, average="macro"),
                "recall": make_scorer(recall_score, average="macro"),
            }
        return (
            {
                "augur_score": make_scorer(roc_auc_score, response_method="predict_proba"),
                "auc": make_scorer(roc_auc_score, response_method="predict_proba"),
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score, average="binary", zero_division=zero_division),
                "f1": make_scorer(f1_score, average="binary"),
                "recall": make_scorer(recall_score, average="binary"),
            }
            if isinstance(self.estimator, RandomForestClassifier | LogisticRegression)
            else {
                "augur_score": make_scorer(self.ccc_score),
                "r2": make_scorer(r2_score),
                "ccc": make_scorer(self.ccc_score),
                "neg_mean_squared_error": make_scorer(root_mean_squared_error),
                "explained_variance": make_scorer(explained_variance_score),
            }
        )

    def run_cross_validation(
        self,
        subsample: AnnData,
        *,
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

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_highly_variable(loaded_data)
            >>> subsample = ag_rfc.draw_subsample(adata, augur_mode="default", subsample_size=20, feature_perc=0.5, \
                categorical=True, random_state=42)
            >>> results = ag_rfc.run_cross_validation(subsample=subsample, folds=3, subsample_idx=0, random_state=42, zero_division=0)
        """
        x = subsample.to_df()
        y = subsample.obs["y_"]
        scorer = self.set_scorer(multiclass=len(y.unique()) > 2, zero_division=zero_division)
        folds = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

        results = cross_validate(
            estimator=self.estimator,
            X=x,
            y=y.values.ravel(),
            scoring=scorer,
            cv=folds,
            return_estimator=True,
        )

        results["subsample_idx"] = subsample_idx
        for score in scorer:
            results[f"mean_{score}"] = results[f"test_{score}"].mean()

        # feature importances
        feature_importances = defaultdict(list)
        if isinstance(self.estimator, RandomForestClassifier | RandomForestRegressor):
            for fold, estimator in list(zip(range(len(results["estimator"])), results["estimator"], strict=False)):
                feature_importances["genes"].extend(x.columns.tolist())
                feature_importances["feature_importances"].extend(estimator.feature_importances_.tolist())
                feature_importances["subsample_idx"].extend(len(x.columns) * [subsample_idx])
                feature_importances["fold"].extend(len(x.columns) * [fold])

        # standardized coefficients with Agresti method
        # cf. https://think-lab.github.io/d/205/#3
        if isinstance(self.estimator, LogisticRegression):
            for fold, self.estimator in list(zip(range(len(results["estimator"])), results["estimator"], strict=False)):
                feature_importances["genes"].extend(x.columns.tolist())
                feature_importances["feature_importances"].extend(
                    (self.estimator.coef_ * self.estimator.coef_.std()).flatten().tolist()
                )
                feature_importances["subsample_idx"].extend(len(x.columns) * [subsample_idx])
                feature_importances["fold"].extend(len(x.columns) * [fold])

        results["feature_importances"] = feature_importances

        return results

    def average_metrics(self, cell_cv_results: list[Any]) -> dict[Any, Any]:
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

    def select_highly_variable(self, adata: AnnData) -> AnnData:
        """Feature selection by variance using scanpy highly variable genes function.

        Args:
            adata: Anndata object containing gene expression values (cells in rows, genes in columns)

        Results:
            Anndata object with highly variable genes added as layer

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_highly_variable(loaded_data)
        """
        min_features_for_selection = 1000

        if len(adata.var_names) - 2 > min_features_for_selection:
            try:
                sc.pp.highly_variable_genes(adata)
            except ValueError:
                logger.warn("Data not normalized. Normalizing now using scanpy log1p normalize.")
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata)

        return adata

    def cox_compare(self, loess1, loess2):
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
        res_dx = OLS(yhat_x, z).fit()
        err_zx = res_dx.resid
        res_xzx = OLS(err_zx, x).fit()
        err_xzx = res_xzx.resid

        sigma2_zx = sigma2_x + np.dot(err_zx.T, err_zx) / nobs
        c01 = nobs / 2.0 * (np.log(sigma2_z) - np.log(sigma2_zx))
        v01 = sigma2_x * np.dot(err_xzx.T, err_xzx) / sigma2_zx**2
        q = c01 / np.sqrt(v01)
        pval = 2 * stats.norm.sf(np.abs(q))

        return q, pval

    def select_variance(
        self, adata: AnnData, *, var_quantile: float, filter_negative_residuals: bool, span: float = 0.75
    ):
        """Feature selection based on Augur implementation.

        Args:
            adata: Anndata object
            var_quantile: The quantile below which features will be filtered, based on their residuals in a loess model.
            filter_negative_residuals: if `True`, filter residuals at a fixed threshold of zero, instead of `var_quantile`
            span: Smoothing factor, as a fraction of the number of points to take into account.
                  Should be in the range (0, 1].

        Return:
            AnnData object with additional select_variance column in var.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_variance(loaded_data, var_quantile=0.5, filter_negative_residuals=False, span=0.75)
        """
        adata.var["highly_variable"] = False
        adata.var["means"] = np.ravel(adata.X.mean(axis=0))
        # Converting because the Syntax for power for numpy arrays is different -> use sparse Syntax as default
        if isinstance(adata.X, np.ndarray):
            adata.X = sparse.csc_matrix(adata.X)
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
            cox1 = self.cox_compare(fit1, fit2)
            cox2 = self.cox_compare(fit2, fit1)

            # compare p values
            model = fit1 if cox1[1] < cox2[1] else fit2

        residuals = model.outputs.fitted_residuals

        # select features by quantile (or override w/positive residuals)
        genes = (
            keep[residuals > 0] if filter_negative_residuals else keep[residuals > np.quantile(residuals, var_quantile)]
        )

        adata.var["highly_variable"] = [x in genes for x in adata.var.index]

        return adata

    def predict(
        self,
        adata: AnnData,
        *,
        n_subsamples: int = 50,
        subsample_size: int = 20,
        folds: int = 3,
        min_cells: int = None,
        feature_perc: float = 0.5,
        var_quantile: float = 0.5,
        span: float = 0.75,
        filter_negative_residuals: bool = False,
        n_threads: int = 4,
        augur_mode: Literal["default", "permute", "velocity"] = "default",
        select_variance_features: bool = True,
        key_added: str = "augurpy_results",
        random_state: int | None = None,
        zero_division: int | str = 0,
    ) -> tuple[AnnData, dict[str, Any]]:
        """Calculates the Area under the Curve using the given classifier.

        Args:
            adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
            n_subsamples: number of random subsamples to draw from complete dataset for each cell type
            subsample_size: number of cells to subsample randomly per type from each experimental condition
            folds: number of folds to run cross validation on. Be careful changing this parameter without also changing `subsample_size`.
            min_cells: minimum number of cells for a particular cell type in each condition in order to retain that type for analysis (depricated..)
            feature_perc: proportion of genes that are randomly selected as features for input to the classifier in each
                          subsample using the random gene filter
            var_quantile: The quantile below which features will be filtered, based on their residuals in a loess model.
            span: Smoothing factor, as a fraction of the number of points to take into account. Should be in the range (0, 1].
            filter_negative_residuals: if `True`, filter residuals at a fixed threshold of zero, instead of `var_quantile`
            n_threads: number of threads to use for parallelization
            select_variance_features: Whether to select genes based on the original Augur implementation (True)
                                      or using scanpy's highly_variable_genes (False).
            key_added: Key to add results to in .uns
            augur_mode: One of 'default', 'velocity' or 'permute'. Setting augur_mode = "velocity" disables feature selection,
                        assuming feature selection has been performed by the RNA velocity procedure to produce the input matrix,
                        while setting augur_mode = "permute" will generate a null distribution of AUCs for each cell type by
                        permuting the labels. Note that when setting augur_mode = "permute" n_subsample values less than 100 will be set to 500.
            random_state: set numpy random seed, sampling seed and fold seed
            zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
                           set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

        Returns:
            A tuple with a dictionary containing the following keys with an updated AnnData object with mean_augur_score metrics in obs.

                * summary_metrics: Pandas Dataframe containing mean metrics for each cell type
                * feature_importances: Pandas Dataframe containing feature importances of genes across all cross validation runs
                * full_results: Dict containing merged results of individual cross validation runs for each cell type
                * [cell_types]: Cross validation runs of the cell type called

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> h_adata, h_results = ag_rfc.predict(loaded_data, subsample_size=20, n_threads=4)
        """
        adata = adata.copy()
        if augur_mode == "permute" and n_subsamples < 100:
            n_subsamples = 500
        if is_regressor(self.estimator) and len(adata.obs["y_"].unique()) <= 3:
            raise ValueError(
                f"Regressors cannot be used on {len(adata.obs['label'].unique())} labels. Try a classifier."
            )
        if isinstance(self.estimator, LogisticRegression) and len(adata.obs["y_"].unique()) > 2:
            raise ValueError(
                "Logistic regression cannot be used for multiclass classification. "
                + "Use a random forest classifier or filter labels in load()."
            )
        if min_cells is None:
            min_cells = n_subsamples
        results: dict[str, Any] = {
            "summary_metrics": {},
            "feature_importances": defaultdict(list),
            "full_results": defaultdict(list),
        }
        if select_variance_features:
            logger.warning("Set smaller span value in the case of a `segmentation fault` error.")
            logger.warning("Set larger span in case of svddc or other near singularities error.")
        adata.obs["augur_score"] = nan
        for cell_type in track(adata.obs["cell_type"].unique(), description="Processing data..."):
            cell_type_subsample = adata[adata.obs["cell_type"] == cell_type].copy()
            if augur_mode in ("default", "permute"):
                cell_type_subsample = (
                    self.select_highly_variable(cell_type_subsample)
                    if not select_variance_features
                    else self.select_variance(
                        cell_type_subsample,
                        var_quantile=var_quantile,
                        filter_negative_residuals=filter_negative_residuals,
                        span=span,
                    )
                )
            if len(cell_type_subsample) < min_cells:
                logger.warning(
                    f"Skipping {cell_type} cell type - {len(cell_type_subsample)} samples is less than min_cells {min_cells}."
                )
            elif (
                cell_type_subsample.obs.groupby(
                    ["cell_type", "label"],
                    observed=True,
                ).y_.count()
                < subsample_size
            ).any():
                logger.warning(
                    f"Skipping {cell_type} cell type - the number of samples for at least one class type is less than "
                    f"subsample size {subsample_size}."
                )
            else:
                results[cell_type] = Parallel(n_jobs=n_threads)(
                    delayed(self.cross_validate_subsample)(
                        adata=cell_type_subsample,
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
                results["summary_metrics"][cell_type] = self.average_metrics(results[cell_type])

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

                for idx, cv in zip(range(n_subsamples), results[cell_type], strict=False):
                    results["full_results"]["idx"].extend([idx] * folds)
                    results["full_results"]["augur_score"].extend(cv["test_augur_score"])
                    results["full_results"]["folds"].extend(range(folds))
                results["full_results"]["cell_type"].extend([cell_type] * folds * n_subsamples)
        # make sure one cell type worked
        if len(results) <= 2:
            logger.warning("No cells types had more than min_cells needed. Please adjust data or min_cells parameter.")

        results["summary_metrics"] = pd.DataFrame(results["summary_metrics"])
        results["feature_importances"] = pd.DataFrame(results["feature_importances"])
        results["full_results"] = pd.DataFrame(results["full_results"])
        adata.uns["summary_metrics"] = pd.DataFrame(results["summary_metrics"])
        adata.uns[key_added] = results

        return adata, results

    def predict_differential_prioritization(
        self,
        augur_results1: dict[str, Any],
        augur_results2: dict[str, Any],
        permuted_results1: dict[str, Any],
        permuted_results2: dict[str, Any],
        n_subsamples: int = 50,
        n_permutations: int = 1000,
    ) -> pd.DataFrame:
        """Predicts the differential prioritization by performing permutation tests on samples.

        Performs permutation tests that identifies cell types with statistically significant differences in `augur_score`
        between two conditions respectively compared to the control.

        Args:
            augur_results1: Augurpy results from condition 1, obtained from `predict()[1]`
            augur_results2: Augurpy results from condition 2, obtained from `predict()[1]`
            permuted_results1: permuted Augurpy results from condition 1, obtained from `predict()` with argument `augur_mode=permute`
            permuted_results2: permuted Augurpy results from condition 2, obtained from `predict()` with argument `augur_mode=permute`
            n_subsamples: number of subsamples to pool when calculating the mean augur score for each permutation.
            n_permutations: the total number of mean augur scores to calculate from a background distribution

        Returns:
            Results object containing mean augur scores.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")

            >>> data_15 = ag_rfc.load(adata, condition_label="Maintenance_Cocaine", treatment_label="withdraw_15d_Cocaine")
            >>> adata_15, results_15 = ag_rfc.predict(data_15, random_state=None, n_threads=4)
            >>> adata_15_permute, results_15_permute = ag_rfc.predict(data_15, augur_mode="permute", n_subsamples=100, random_state=None, n_threads=4)

            >>> data_48 = ag_rfc.load(adata, condition_label="Maintenance_Cocaine", treatment_label="withdraw_48h_Cocaine")
            >>> adata_48, results_48 = ag_rfc.predict(data_48, random_state=None, n_threads=4)
            >>> adata_48_permute, results_48_permute = ag_rfc.predict(data_48, augur_mode="permute", n_subsamples=100, random_state=None, n_threads=4)

            >>> pvals = ag_rfc.predict_differential_prioritization(augur_results1=results_15, augur_results2=results_48, \
                permuted_results1=results_15_permute, permuted_results2=results_48_permute)
        """
        # compare available cell types
        cell_types = (
            set(augur_results1["summary_metrics"].columns)
            & set(augur_results2["summary_metrics"].columns)
            & set(permuted_results1["summary_metrics"].columns)
            & set(permuted_results2["summary_metrics"].columns)
        )

        cell_types_list = list(cell_types)

        # mean augur scores
        augur_score1 = (
            augur_results1["summary_metrics"]
            .loc["mean_augur_score", cell_types_list]
            .reset_index()
            .rename(columns={"index": "cell_type"})
        )
        augur_score2 = (
            augur_results2["summary_metrics"]
            .loc["mean_augur_score", cell_types_list]
            .reset_index()
            .rename(columns={"index": "cell_type"})
        )

        # mean permuted scores over cross validation runs
        permuted_cv_augur1 = (
            permuted_results1["full_results"][permuted_results1["full_results"]["cell_type"].isin(cell_types_list)]
            .groupby(["cell_type", "idx"], as_index=False)
            .mean()
        )
        permuted_cv_augur2 = (
            permuted_results2["full_results"][permuted_results2["full_results"]["cell_type"].isin(cell_types_list)]
            .groupby(["cell_type", "idx"], as_index=False)
            .mean()
        )

        rng = np.random.default_rng()
        sampled_data = []

        # draw mean aucs for permute1 and permute2
        for celltype in permuted_cv_augur1["cell_type"].unique():
            df1 = permuted_cv_augur1[permuted_cv_augur1["cell_type"] == celltype]
            df2 = permuted_cv_augur2[permuted_cv_augur2["cell_type"] == celltype]

            indices1 = rng.choice(len(df1), size=(n_permutations, n_subsamples), replace=True)
            indices2 = rng.choice(len(df2), size=(n_permutations, n_subsamples), replace=True)

            scores1 = df1["augur_score"].values[indices1]
            scores2 = df2["augur_score"].values[indices2]

            means1 = scores1.mean(axis=1)
            means2 = scores2.mean(axis=1)
            stds1 = scores1.std(axis=1)
            stds2 = scores2.std(axis=1)

            sampled_data.append(
                pd.DataFrame(
                    {
                        "cell_type": np.repeat(celltype, n_permutations),
                        "permutation_idx": np.arange(n_permutations),
                        "mean1": means1,
                        "mean2": means2,
                        "std1": stds1,
                        "std2": stds2,
                    }
                )
            )

        sampled_df = pd.concat(sampled_data)

        # delta between augur scores
        delta = augur_score1.merge(augur_score2, on=["cell_type"], suffixes=("1", "2")).assign(
            delta_augur=lambda x: x.mean_augur_score2 - x.mean_augur_score1
        )

        # delta between permutation scores
        delta_rnd = sampled_df.assign(delta_rnd=lambda x: x.mean2 - x.mean1)

        # number of values where permutations are larger than test statistic
        delta["b"] = (
            pd.merge(
                delta_rnd[["cell_type", "delta_rnd"]], delta[["cell_type", "delta_augur"]], on="cell_type", how="left"
            )
            .assign(b=lambda x: (x.delta_rnd >= x.delta_augur))
            .groupby("cell_type", as_index=False)
            .sum()["b"]
        )
        delta["m"] = n_permutations
        delta["z"] = (
            delta["delta_augur"] - delta_rnd.groupby("cell_type", as_index=False).mean()["delta_rnd"]
        ) / delta_rnd.groupby("cell_type", as_index=False).std()["delta_rnd"]

        delta["pval"] = np.minimum(
            2 * (delta["b"] + 1) / (delta["m"] + 1), 2 * (delta["m"] - delta["b"] + 1) / (delta["m"] + 1)
        )
        delta["padj"] = fdrcorrection(delta["pval"])[1]

        return delta

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_dp_scatter(  # pragma: no cover # noqa: D417
        self,
        results: pd.DataFrame,
        *,
        top_n: int = None,
        ax: Axes = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot scatterplot of differential prioritization.

        Args:
            results: Results after running differential prioritization.
            top_n: optionally, the number of top prioritized cell types to label in the plot
            ax: optionally, axes used to draw plot
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")

            >>> data_15 = ag_rfc.load(adata, condition_label="Maintenance_Cocaine", treatment_label="withdraw_15d_Cocaine")
            >>> adata_15, results_15 = ag_rfc.predict(data_15, random_state=None, n_threads=4)
            >>> adata_15_permute, results_15_permute = ag_rfc.predict(data_15, augur_mode="permute", n_subsamples=100, random_state=None, n_threads=4)

            >>> data_48 = ag_rfc.load(adata, condition_label="Maintenance_Cocaine", treatment_label="withdraw_48h_Cocaine")
            >>> adata_48, results_48 = ag_rfc.predict(data_48, random_state=None, n_threads=4)
            >>> adata_48_permute, results_48_permute = ag_rfc.predict(data_48, augur_mode="permute", n_subsamples=100, random_state=None, n_threads=4)

            >>> pvals = ag_rfc.predict_differential_prioritization(augur_results1=results_15, augur_results2=results_48, \
                permuted_results1=results_15_permute, permuted_results2=results_48_permute)
            >>> ag_rfc.plot_dp_scatter(pvals)

        Preview:
            .. image:: /_static/docstring_previews/augur_dp_scatter.png
        """
        x = results["mean_augur_score1"]
        y = results["mean_augur_score2"]

        if ax is None:
            fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=results.z, cmap="Greens")

        # adding optional labels
        top_n_index = results.sort_values(by="pval").index[:top_n]
        for idx in top_n_index:
            ax.annotate(
                results.loc[idx, "cell_type"],
                (results.loc[idx, "mean_augur_score1"], results.loc[idx, "mean_augur_score2"]),
            )

        # add diagonal
        limits = max(ax.get_xlim(), ax.get_ylim())
        (_,) = ax.plot(limits, limits, ls="--", c=".3")

        # formatting and details
        plt.xlabel("Augur scores 1")
        plt.ylabel("Augur scores 2")
        legend1 = ax.legend(*scatter.legend_elements(), loc="center left", title="z-scores", bbox_to_anchor=(1, 0.5))
        ax.add_artist(legend1)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_important_features(  # pragma: no cover # noqa: D417
        self,
        data: dict[str, Any],
        *,
        key: str = "augurpy_results",
        top_n: int = 10,
        ax: Axes = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot a lollipop plot of the n features with largest feature importances.

        Args:
            data: results after running `predict()` as dictionary or the AnnData object.
            key: Key in the AnnData object of the results
            top_n: n number feature importance values to plot. Default is 10.
            ax: optionally, axes used to draw plot
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> v_adata, v_results = ag_rfc.predict(
            ...     loaded_data, subsample_size=20, select_variance_features=True, n_threads=4
            ... )
            >>> ag_rfc.plot_important_features(v_results)

        Preview:
            .. image:: /_static/docstring_previews/augur_important_features.png
        """
        results = data.uns[key] if isinstance(data, AnnData) else data
        n_features = (
            results["feature_importances"]
            .groupby("genes", as_index=False)
            .feature_importances.mean()
            .sort_values(by="feature_importances")[-top_n:]
        )

        if ax is None:
            fig, ax = plt.subplots()
        y_axes_range = range(1, top_n + 1)
        ax.hlines(
            y_axes_range,
            xmin=0,
            xmax=n_features["feature_importances"],
        )

        ax.plot(n_features["feature_importances"], y_axes_range, "o")

        plt.xlabel("Mean Feature Importance")
        plt.ylabel("Gene")
        plt.yticks(y_axes_range, n_features["genes"])

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_lollipop(  # pragma: no cover # noqa: D417
        self,
        data: dict[str, Any] | AnnData,
        *,
        key: str = "augurpy_results",
        ax: Axes = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot a lollipop plot of the mean augur values.

        Args:
            data: results after running `predict()` as dictionary or the AnnData object.
            key: .uns key in the results AnnData object.
            ax: optionally, axes used to draw plot.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> v_adata, v_results = ag_rfc.predict(
            ...     loaded_data, subsample_size=20, select_variance_features=True, n_threads=4
            ... )
            >>> ag_rfc.plot_lollipop(v_results)

        Preview:
            .. image:: /_static/docstring_previews/augur_lollipop.png
        """
        results = data.uns[key] if isinstance(data, AnnData) else data
        if ax is None:
            fig, ax = plt.subplots()
        y_axes_range = range(1, len(results["summary_metrics"].columns) + 1)
        ax.hlines(
            y_axes_range,
            xmin=0,
            xmax=results["summary_metrics"].sort_values("mean_augur_score", axis=1).loc["mean_augur_score"],
        )

        ax.plot(
            results["summary_metrics"].sort_values("mean_augur_score", axis=1).loc["mean_augur_score"],
            y_axes_range,
            "o",
        )

        plt.xlabel("Mean Augur Score")
        plt.ylabel("Cell Type")
        plt.yticks(y_axes_range, results["summary_metrics"].sort_values("mean_augur_score", axis=1).columns)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_scatterplot(  # pragma: no cover # noqa: D417
        self,
        results1: dict[str, Any],
        results2: dict[str, Any],
        *,
        top_n: int = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Create scatterplot with two augur results.

        Args:
            results1: results after running `predict()`
            results2: results after running `predict()`
            top_n: optionally, the number of top prioritized cell types to label in the plot
            {common_plot_args}

        Returns:
            Axes of the plot.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> h_adata, h_results = ag_rfc.predict(loaded_data, subsample_size=20, n_threads=4)
            >>> v_adata, v_results = ag_rfc.predict(
            ...     loaded_data, subsample_size=20, select_variance_features=True, n_threads=4
            ... )
            >>> ag_rfc.plot_scatterplot(v_results, h_results)

        Preview:
            .. image:: /_static/docstring_previews/augur_scatterplot.png
        """
        cell_types = results1["summary_metrics"].columns

        fig, ax = plt.subplots()
        ax.scatter(
            results1["summary_metrics"].loc["mean_augur_score", cell_types],
            results2["summary_metrics"].loc["mean_augur_score", cell_types],
        )

        # adding optional labels
        top_n_cell_types = (
            (results1["summary_metrics"].loc["mean_augur_score"] - results2["summary_metrics"].loc["mean_augur_score"])
            .sort_values(ascending=False)
            .index[:top_n]
        )
        for txt in top_n_cell_types:
            ax.annotate(
                txt,
                (
                    results1["summary_metrics"].loc["mean_augur_score", txt],
                    results2["summary_metrics"].loc["mean_augur_score", txt],
                ),
            )

        # adding diagonal
        limits = max(ax.get_xlim(), ax.get_ylim())
        (diag_line,) = ax.plot(limits, limits, ls="--", c=".3")

        plt.xlabel("Augur scores 1")
        plt.ylabel("Augur scores 2")

        if return_fig:
            return plt.gcf()
        plt.show()
        return None
