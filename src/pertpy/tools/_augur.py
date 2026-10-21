from __future__ import annotations

import random
import time
import warnings
from collections import defaultdict
from functools import partial
from math import floor, nan
from typing import TYPE_CHECKING, Any, Literal, cast

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from fast_array_utils.conv import to_dense
from joblib import Parallel, delayed
from rich.progress import track
from scipy import sparse, stats
from sklearn import config_context
from sklearn.base import clone, is_classifier, is_regressor
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from skmisc.loess import loess
from statsmodels.api import OLS
from statsmodels.stats.multitest import fdrcorrection

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy._logger import logger
from pertpy._types import CSBase, RandomStateLike, _sklearn_random_state, cast_frame, cast_matrix
from pertpy.tools.core import _is_raw_counts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _fit_forest(forest: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> list[DecisionTreeClassifier]:
    """Fit the trees of ``forest`` on float32 ``X`` one by one, bit for bit as ``RandomForestClassifier.fit`` does but without its per-tree cloning and validation."""
    random_state = check_random_state(forest.random_state)
    params = {name: getattr(forest, name) for name in forest.estimator_params if name != "random_state"}
    n_samples = X.shape[0]
    trees = []
    for _ in range(forest.n_estimators):
        seed = random_state.randint(np.iinfo(np.int32).max)
        indices = np.random.RandomState(seed).randint(0, n_samples, n_samples, dtype=np.int32)
        tree = DecisionTreeClassifier(random_state=seed, **params)
        tree.fit(X, y, sample_weight=np.bincount(indices, minlength=n_samples).astype(np.float64), check_input=False)
        trees.append(tree)
    return trees


def _forest_proba(trees: list[DecisionTreeClassifier], X: np.ndarray) -> np.ndarray:
    """Class probabilities of a forest on float32 ``X``, accumulated like ``RandomForestClassifier.predict_proba``."""
    proba = np.zeros((X.shape[0], trees[0].n_classes_))
    for tree in trees:
        proba += tree.predict_proba(X, check_input=False)
    return proba / len(trees)


def _forest_importances(trees: list[DecisionTreeClassifier]) -> np.ndarray:
    """Impurity importances of a forest, like ``RandomForestClassifier.feature_importances_``."""
    importances = [tree.feature_importances_ for tree in trees if tree.tree_.node_count > 1]
    if not importances:
        return np.zeros(trees[0].n_features_in_)
    mean = np.mean(importances, axis=0)
    return mean / np.sum(mean)


class Augur:
    """Python implementation of Augur."""

    def __init__(
        self,
        estimator: Literal["random_forest_classifier", "random_forest_regressor", "logistic_regression_classifier"],
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: Literal["auto", "log2", "sqrt"] | int | float = 2,
        penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
        random_state: RandomStateLike = None,
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
        layer: str | None = None,
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
            layer: Layer in AnnData to use for expression data. If None, uses .X
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
            >>> adata = pt.ds.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> augur_adata = ag_rfc.load(adata)
        """
        if isinstance(input, AnnData):
            adata = input
            obs_renamed = cast_frame(adata.obs).rename(columns={cell_type_col: "cell_type", label_col: "label"})

        elif isinstance(input, pd.DataFrame):
            if meta is None:
                try:
                    _ = input[cell_type_col]
                    _ = input[label_col]
                except KeyError:
                    logger.error("No column names matching cell_type_col and label_col.")

            label = input[label_col] if meta is None else meta[label_col]
            cell_type = input[cell_type_col] if meta is None else meta[cell_type_col]
            X = input.drop([label_col, cell_type_col], axis=1) if meta is None else input
            adata = AnnData(X=X, obs=pd.DataFrame({"cell_type": cell_type, "label": label}))
            obs_renamed = cast_frame(adata.obs)

        if len(obs_renamed["label"].unique()) < 2:
            raise ValueError("Less than two unique labels in dataset. At least two are needed for the analysis.")

        if isinstance(input, AnnData):
            final_adata = AnnData(
                X=adata.X,
                obs=obs_renamed,
                var=cast_frame(adata.var),
                layers=cast("Mapping[str, np.ndarray | CSBase]", adata.layers),
            )
        else:
            final_adata = adata

        # dummy variables for categorical data
        if final_adata.obs["label"].dtype.name == "category":
            if condition_label is not None and treatment_label is not None:
                logger.info(f"Filtering samples with {condition_label} and {treatment_label} labels.")
                final_adata = ad.concat(
                    [
                        final_adata[final_adata.obs["label"] == condition_label],
                        final_adata[final_adata.obs["label"] == treatment_label],
                    ]
                )
            final_adata.obs["y_"] = LabelEncoder().fit_transform(final_adata.obs["label"])
        else:
            y = final_adata.obs["label"].to_frame()
            y = y.rename(columns={"label": "y_"})
            final_adata.obs = pd.concat([final_adata.obs, y], axis=1)

        if layer is not None:
            if layer not in final_adata.layers:
                raise ValueError(f"Layer '{layer}' not found in AnnData object")
            counts = cast_matrix(final_adata.layers[layer])
        else:
            counts = cast_matrix(final_adata.X)

        if not _is_raw_counts(counts):
            logger.warning("Data does not appear to be raw counts. Augur developers recommend using raw counts.")

        return final_adata

    def create_estimator(
        self,
        classifier: (Literal["random_forest_classifier", "random_forest_regressor", "logistic_regression_classifier"]),
        *,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: Literal["auto", "log2", "sqrt"] | int | float = 2,
        penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
        random_state: RandomStateLike = None,
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
        random_state = _sklearn_random_state(random_state)
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
            # scikit-learn 1.8 deprecated `penalty` in favor of `l1_ratio` (the L1/L2 mix) and `C` (no penalty via C=inf).
            if penalty in (None, "none"):
                return LogisticRegression(C=np.inf, random_state=random_state)
            l1_ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": 0.5}[penalty]
            solver = "saga" if l1_ratio > 0.0 else "lbfgs"
            return LogisticRegression(l1_ratio=l1_ratio, solver=solver, random_state=random_state)
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
            >>> adata = pt.ds.sc_sim_augur()
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
            y = adata.obs["y_"]
            pools = [np.flatnonzero(y == code) for code in y.unique()]
        else:
            pools = [np.arange(adata.n_obs)]
        obs_idx = np.concatenate([sc.pp.sample(pool, n=subsample_size, rng=random_state)[0] for pool in pools])
        subsample = adata[obs_idx, features].copy()

        # filter features with 0 variance
        var = cast_frame(subsample.var)
        var["highly_variable"] = False
        var["means"] = np.ravel(cast_matrix(subsample.X).mean(axis=0))
        # Converting because the Syntax for power for numpy arrays is different -> use sparse Syntax as default
        X = cast_matrix(subsample.X)
        if isinstance(X, np.ndarray):
            X = sparse.csc_matrix(X)
            subsample.X = X
        var["var"] = np.ravel(X.power(2).mean(axis=0) - np.power(X.mean(axis=0), 2))
        # remove all features with 0 variance
        var.loc[var["var"] > 0, "highly_variable"] = True

        return subsample[:, var["highly_variable"]]

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
            >>> adata = pt.ds.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_highly_variable(loaded_data)
            >>> subsample = ag_rfc.draw_subsample(adata, augur_mode="default", subsample_size=20, feature_perc=0.5, \
                categorical=True, random_state=42)
        """
        random.seed(random_state)
        if augur_mode == "permute":
            # shuffle labels
            obs = cast_frame(adata.obs).copy()
            y_columns = obs.columns[obs.columns.str.startswith("y_")]
            obs[y_columns] = obs[y_columns].sample(frac=1, random_state=random_state).values
            adata = AnnData(X=adata.X, obs=obs, var=cast_frame(adata.var))

        if augur_mode == "velocity":
            features = adata.var_names.tolist()
        else:
            features = adata.var_names[cast_frame(adata.var)["highly_variable"]].tolist()
            if len(features) >= 1000:
                features = random.sample(features, floor(len(features) * feature_perc))

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
        random_state: RandomStateLike,
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
            >>> adata = pt.ds.sc_sim_augur()
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

    def _metric_functions(
        self, multiclass: bool, zero_division: int | str
    ) -> dict[str, tuple[Callable[..., float], dict[str, Any], str]]:
        """Metric name to (function, keyword arguments, estimator response method) for the estimator.

        Binary metrics take the first class as the positive one, like yardstick's ``event_level = "first"`` in R Augur.
        """
        if is_regressor(self.estimator):
            return {
                "augur_score": (self.ccc_score, {}, "predict"),
                "r2": (r2_score, {}, "predict"),
                "ccc": (self.ccc_score, {}, "predict"),
                "neg_mean_squared_error": (root_mean_squared_error, {}, "predict"),
                "explained_variance": (explained_variance_score, {}, "predict"),
            }
        auc_kwargs: dict[str, Any] = {"multi_class": "ovo"} if multiclass else {}
        label_kwargs: dict[str, Any] = {"average": "macro"} if multiclass else {"average": "binary", "pos_label": 0}
        return {
            "augur_score": (roc_auc_score, auc_kwargs, "predict_proba"),
            "auc": (roc_auc_score, auc_kwargs, "predict_proba"),
            "accuracy": (accuracy_score, {}, "predict"),
            "precision": (precision_score, {**label_kwargs, "zero_division": zero_division}, "predict"),
            "f1": (f1_score, label_kwargs, "predict"),
            "recall": (recall_score, label_kwargs, "predict"),
        }

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
        return {
            name: make_scorer(function, response_method=method, **kwargs)
            for name, (function, kwargs, method) in self._metric_functions(multiclass, zero_division).items()
        }

    def _score_once(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Mapping[str, tuple[Callable[..., float], dict[str, Any], str]],
    ) -> dict[str, float]:
        """Evaluate every metric from one pass of the estimator over ``X``, giving the same values as scikit-learn's scorers one by one.

        A metric that raises scores NaN, as scikit-learn's ``error_score`` default does.
        """
        if any(method == "predict_proba" for _, _, method in metrics.values()):
            responses = self._responses_from_proba(estimator.classes_, self._predict_proba(estimator, X))
        else:
            responses = {"predict": estimator.predict(X)}
        return self._evaluate_metrics(y, responses, metrics)

    @staticmethod
    def _responses_from_proba(classes: np.ndarray, proba: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "predict_proba": proba if proba.shape[1] > 2 else proba[:, 1],
            "predict": classes.take(np.argmax(proba, axis=1), axis=0),
        }

    @staticmethod
    def _evaluate_metrics(
        y: np.ndarray,
        responses: Mapping[str, np.ndarray],
        metrics: Mapping[str, tuple[Callable[..., float], dict[str, Any], str]],
    ) -> dict[str, float]:
        scores = {}
        for name, (function, kwargs, method) in metrics.items():
            try:
                scores[name] = function(y, responses[method], **kwargs)
            except Exception as error:  # noqa: BLE001
                warnings.warn(f"Scoring {name} failed, setting it to NaN: {error}", UserWarning, stacklevel=2)
                scores[name] = np.nan
        return scores

    @staticmethod
    def _predict_proba(estimator: Any, X: np.ndarray) -> np.ndarray:
        """Class probabilities of ``estimator`` on ``X``.

        A random forest is accumulated tree by tree without per-tree input validation, exactly as its ``predict_proba`` does it.
        """
        if not isinstance(estimator, RandomForestClassifier):
            return estimator.predict_proba(X)
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        proba = np.zeros((X.shape[0], estimator.n_classes_))
        for tree in estimator.estimators_:
            proba += tree.predict_proba(X32, check_input=False)
        proba /= len(estimator.estimators_)
        return proba

    def run_cross_validation(
        self,
        subsample: AnnData,
        *,
        subsample_idx: int,
        folds: int,
        random_state: RandomStateLike,
        zero_division: int | str,
    ) -> dict:
        """Perform cross validation on given subsample.

        Binary precision, recall and F1 take class ``0`` as the positive class, so ``y_`` must be encoded from ``0`` as :meth:`load` does.

        Args:
            subsample: subsample of gene expression matrix of size subsample_size
            estimator: classifier object to use in calculating the area under the curve
            subsample_idx: index of subsample
            folds: number of folds
            random_state: set random fold seed; ``None`` seeds the folds and an unseeded estimator by ``subsample_idx``, so results are reproducible like R Augur's
            zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
                set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

        Returns:
            Dictionary containing prediction metrics and estimator for each fold.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_highly_variable(loaded_data)
            >>> subsample = ag_rfc.draw_subsample(adata, augur_mode="default", subsample_size=20, feature_perc=0.5, \
                categorical=True, random_state=42)
            >>> results = ag_rfc.run_cross_validation(subsample=subsample, folds=3, subsample_idx=0, random_state=42, zero_division=0)
        """
        x = np.asarray(to_dense(subsample.X))
        genes = np.asarray(subsample.var_names, dtype=object)
        y = subsample.obs["y_"].to_numpy().ravel()
        metrics = self._metric_functions(multiclass=len(np.unique(y)) > 2, zero_division=zero_division)
        seed = subsample_idx if random_state is None else _sklearn_random_state(random_state)
        fold_class = KFold if is_regressor(self.estimator) else StratifiedKFold
        splitter = fold_class(n_splits=folds, random_state=seed, shuffle=True)
        estimator = self.estimator
        if estimator.random_state is None:
            estimator = clone(estimator).set_params(random_state=subsample_idx)
        with config_context(skip_parameter_validation=True, assume_finite=True):
            if (
                isinstance(estimator, RandomForestClassifier)
                and estimator.bootstrap
                and estimator.max_samples is None
                and estimator.class_weight is None
                and not estimator.oob_score
            ):
                results, importances = self._cross_validate_forest(estimator, x, y, splitter, metrics)
            else:
                results = cross_validate(
                    estimator=estimator,
                    X=x,
                    y=y,
                    scoring=partial(self._score_once, metrics=metrics),
                    cv=splitter,
                    return_estimator=True,
                )
                estimators = results.pop("estimator")
                if isinstance(estimator, RandomForestClassifier | RandomForestRegressor):
                    importances = [fitted.feature_importances_ for fitted in estimators]
                elif isinstance(estimator, LogisticRegression):
                    # standardized coefficients with Agresti method, cf. https://think-lab.github.io/d/205/#3
                    importances = [(fitted.coef_ * fitted.coef_.std()).ravel() for fitted in estimators]
                else:
                    importances = []
        results["subsample_idx"] = subsample_idx
        for name in metrics:
            results[f"mean_{name}"] = results[f"test_{name}"].mean()
        n_folds = len(importances)
        results["feature_importances"] = {
            "genes": np.tile(genes, n_folds),
            "feature_importances": np.concatenate(importances) if importances else np.empty(0),
            "subsample_idx": np.full(n_folds * genes.size, subsample_idx),
            "fold": np.repeat(np.arange(n_folds), genes.size),
        }

        return results

    def _cross_validate_forest(
        self,
        forest: RandomForestClassifier,
        x: np.ndarray,
        y: np.ndarray,
        splitter: KFold | StratifiedKFold,
        metrics: Mapping[str, tuple[Callable[..., float], dict[str, Any], str]],
    ) -> tuple[dict[str, Any], list[np.ndarray]]:
        """Cross-validate a random forest classifier tree by tree, with the same scores and importances as ``cross_validate`` on the forest."""
        x32 = np.ascontiguousarray(x, dtype=np.float32)
        scores: dict[str, list[float]] = {name: [] for name in metrics}
        fit_times, score_times, importances = [], [], []
        for train, test in splitter.split(x32, y):
            start = time.perf_counter()
            trees = _fit_forest(forest, x32[train], y[train])
            fitted = time.perf_counter()
            fold_scores = self._evaluate_metrics(
                y[test], self._responses_from_proba(trees[0].classes_, _forest_proba(trees, x32[test])), metrics
            )
            for name, value in fold_scores.items():
                scores[name].append(value)
            fit_times.append(fitted - start)
            score_times.append(time.perf_counter() - fitted)
            importances.append(_forest_importances(trees))
        results: dict[str, Any] = {"fit_time": np.array(fit_times), "score_time": np.array(score_times)}
        results.update({f"test_{name}": np.array(values) for name, values in scores.items()})
        return results, importances

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
            >>> adata = pt.ds.sc_sim_augur()
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
            >>> adata = pt.ds.sc_sim_augur()
            >>> ag_rfc = pt.tl.Augur("random_forest_classifier")
            >>> loaded_data = ag_rfc.load(adata)
            >>> ag_rfc.select_variance(loaded_data, var_quantile=0.5, filter_negative_residuals=False, span=0.75)
        """
        var = cast_frame(adata.var)
        var["highly_variable"] = False
        var["means"] = np.ravel(cast_matrix(adata.X).mean(axis=0))
        # Converting because the Syntax for power for numpy arrays is different -> use sparse Syntax as default
        X = cast_matrix(adata.X)
        if isinstance(X, np.ndarray):
            X = sparse.csc_matrix(X)
            adata.X = X
        var["sds"] = np.ravel(np.sqrt(X.power(2).mean(axis=0) - np.power(X.mean(axis=0), 2)))
        # remove all features with 0 variance
        var.loc[var["sds"] > 0, "highly_variable"] = True
        cvs = var.loc[var["highly_variable"], "means"] / var.loc[var["highly_variable"], "sds"]

        # clip outliers, get best prediction model, get residuals
        lower = np.quantile(cvs, 0.01)
        upper = np.quantile(cvs, 0.99)
        keep = cvs.loc[cvs.between(lower, upper)].index

        cv0 = cvs.loc[keep]
        mean0 = var.loc[keep, "means"]

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
        min_cells: int | None = None,
        feature_perc: float = 0.5,
        var_quantile: float = 0.5,
        span: float = 0.75,
        filter_negative_residuals: bool = False,
        n_threads: int = 4,
        augur_mode: Literal["default", "permute", "velocity"] = "default",
        select_variance_features: bool = True,
        key_added: str = "augurpy_results",
        random_state: RandomStateLike = None,
        zero_division: int | str = 0,
    ) -> tuple[AnnData, dict[str, Any]]:
        """Calculates the Area under the Curve using the given classifier.

        Scikit-learn's random forest keeps searching for a split when the ``max_features`` drawn for a node cannot split it, where R's ``randomForest`` stops, so AUCs run higher than R Augur's on sparse data while the ranking of cell types agrees.

        Args:
            adata: Anndata with obs `label` and `cell_type` for label and cell type and dummie variable `y_` columns used as target
            n_subsamples: number of random subsamples to draw from complete dataset for each cell type
            subsample_size: number of cells to subsample randomly per type from each experimental condition
            folds: number of folds to run cross validation on. Be careful changing this parameter without also changing `subsample_size`.
            min_cells: minimum number of cells of a cell type in each condition to analyze that cell type; defaults to ``subsample_size`` like R Augur
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
            random_state: set numpy random seed, sampling seed and fold seed; ``None`` seeds every subsample's folds and an unseeded estimator by the subsample index, so results are reproducible like R Augur's
            zero_division: 0 or 1 or `warn`; Sets the value to return when there is a zero division. If
                           set to “warn”, this acts as 0, but warnings are also raised. Precision metric parameter.

        Returns:
            A tuple with a dictionary containing the following keys with an updated AnnData object with mean_augur_score metrics in obs.

                * summary_metrics: Pandas Dataframe containing mean metrics for each cell type
                * feature_importances: Pandas Dataframe containing feature importances of genes across all cross validation runs (impurity importance of the random forest or standardized coefficients of the logistic regression, where R Augur reports permutation importance)
                * full_results: Dict containing merged results of individual cross validation runs for each cell type
                * [cell_types]: Cross validation runs of the cell type called

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.sc_sim_augur()
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
            min_cells = subsample_size
        results: dict[str, Any] = {
            "summary_metrics": {},
            "feature_importances": defaultdict(list),
            "full_results": defaultdict(list),
        }
        if select_variance_features:
            logger.warning("Set smaller span value in the case of a `segmentation fault` error.")
            logger.warning("Set larger span in case of svddc or other near singularities error.")
        adata.obs["augur_score"] = nan

        eligible_cell_types = []
        for cell_type in adata.obs["cell_type"].unique():
            cell_type_subsample = adata[adata.obs["cell_type"] == cell_type].copy()
            if is_classifier(self.estimator):
                smallest = int(cast_frame(cell_type_subsample.obs).groupby("label", observed=True)["y_"].count().min())
            else:
                smallest = cell_type_subsample.n_obs
            if smallest < max(min_cells, subsample_size):
                logger.warning(
                    f"Skipping {cell_type} cell type - {smallest} cells in the smallest condition is less than "
                    f"min_cells {min_cells} or subsample_size {subsample_size}."
                )
                continue
            if augur_mode in ("default", "permute"):
                if cell_type_subsample.n_vars < 1000:
                    cell_type_subsample.var["highly_variable"] = True
                elif select_variance_features:
                    cell_type_subsample = self.select_variance(
                        cell_type_subsample,
                        var_quantile=var_quantile,
                        filter_negative_residuals=filter_negative_residuals,
                        span=span,
                    )
                else:
                    cell_type_subsample = self.select_highly_variable(cell_type_subsample)
            eligible_cell_types.append((cell_type, cell_type_subsample))

        # Fan subsamples out across all eligible cell types in one pool so workers don't idle at each cell-type boundary; submission order is preserved for regrouping.
        flat_tasks = [(cell_type, sub, i) for cell_type, sub in eligible_cell_types for i in range(n_subsamples)]
        cv_results = Parallel(n_jobs=n_threads, return_as="generator")(
            delayed(self.cross_validate_subsample)(
                adata=sub,
                augur_mode=augur_mode,
                subsample_size=subsample_size,
                folds=folds,
                feature_perc=feature_perc,
                subsample_idx=i,
                random_state=random_state,
                zero_division=zero_division,
            )
            for _, sub, i in flat_tasks
        )
        cv_results = list(track(cv_results, total=len(flat_tasks), description="Processing data..."))

        for offset, (cell_type, _) in enumerate(eligible_cell_types):
            results[cell_type] = cv_results[offset * n_subsamples : (offset + 1) * n_subsamples]

            # summarize scores for cell type
            results["summary_metrics"][cell_type] = self.average_metrics(results[cell_type])

            # add scores as observation to anndata
            obs = cast_frame(adata.obs)
            obs.loc[obs["cell_type"] == cell_type, "augur_score"] = results["summary_metrics"][cell_type][
                "mean_augur_score"
            ]

            importances = [cv["feature_importances"] for cv in results[cell_type]]
            for key in ("genes", "feature_importances", "subsample_idx", "fold"):
                results["feature_importances"][key].append(np.concatenate([fi[key] for fi in importances]))
            results["feature_importances"]["cell_type"].append(
                np.repeat(cell_type, sum(fi["genes"].size for fi in importances))
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
        results["feature_importances"] = pd.DataFrame(
            {key: np.concatenate(parts) for key, parts in results["feature_importances"].items()}
        )
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
        *,
        n_subsamples: int = 50,
        n_permutations: int = 1000,
        random_state: RandomStateLike = 0,
    ) -> pd.DataFrame:
        """Predicts the differential prioritization by performing permutation tests on samples.

        Performs permutation tests that identifies cell types with statistically significant differences in `augur_score`
        between two conditions respectively compared to the control.
        Each permutation averages ``n_subsamples`` permuted scores drawn without replacement, as R Augur does.

        Args:
            augur_results1: Augurpy results from condition 1, obtained from `predict()[1]`
            augur_results2: Augurpy results from condition 2, obtained from `predict()[1]`
            permuted_results1: permuted Augurpy results from condition 1, obtained from `predict()` with argument `augur_mode=permute`
            permuted_results2: permuted Augurpy results from condition 2, obtained from `predict()` with argument `augur_mode=permute`
            n_subsamples: number of subsamples to pool when calculating the mean augur score for each permutation.
            n_permutations: the total number of mean augur scores to calculate from a background distribution
            random_state: seed of the permutation draws

        Returns:
            Results object containing mean augur scores.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.bhattacherjee()
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

        rng = np.random.default_rng(random_state)
        sampled_data = []

        # draw mean aucs for permute1 and permute2
        for celltype in permuted_cv_augur1["cell_type"].unique():
            df1 = permuted_cv_augur1[permuted_cv_augur1["cell_type"] == celltype]
            df2 = permuted_cv_augur2[permuted_cv_augur2["cell_type"] == celltype]

            indices1 = np.argsort(rng.random((n_permutations, len(df1))), axis=1)[:, : min(n_subsamples, len(df1))]
            indices2 = np.argsort(rng.random((n_permutations, len(df2))), axis=1)[:, : min(n_subsamples, len(df2))]

            scores1 = df1["augur_score"].to_numpy()[indices1]
            scores2 = df2["augur_score"].to_numpy()[indices2]

            means1 = scores1.mean(axis=1)
            means2 = scores2.mean(axis=1)
            stds1 = scores1.std(axis=1, ddof=1)
            stds2 = scores2.std(axis=1, ddof=1)

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
        # delta between permutation scores, summarized per cell type
        permuted = sampled_df.assign(delta_rnd=lambda x: x.mean2 - x.mean1).merge(
            delta[["cell_type", "delta_augur"]], on="cell_type"
        )
        permuted["exceeds"] = permuted["delta_rnd"] >= permuted["delta_augur"]
        null = permuted.groupby("cell_type").agg(
            b=("exceeds", "sum"), mean_rnd=("delta_rnd", "mean"), sd_rnd=("delta_rnd", "std")
        )
        delta = delta.merge(null, left_on="cell_type", right_index=True)
        delta["m"] = n_permutations
        delta["z"] = (delta["delta_augur"] - delta["mean_rnd"]) / delta["sd_rnd"]
        delta["pval"] = np.minimum(
            2 * (delta["b"] + 1) / (delta["m"] + 1), 2 * (delta["m"] - delta["b"] + 1) / (delta["m"] + 1)
        )
        delta["padj"] = fdrcorrection(delta["pval"])[1]
        return delta.drop(columns=["mean_rnd", "sd_rnd"])

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_dp_scatter(  # pragma: no cover # noqa: D417
        self,
        results: pd.DataFrame,
        *,
        top_n: int | None = None,
        ax: Axes | None = None,
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
            >>> adata = pt.ds.bhattacherjee()
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
        ax: Axes | None = None,
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
            >>> adata = pt.ds.sc_sim_augur()
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
        ax: Axes | None = None,
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
            >>> adata = pt.ds.sc_sim_augur()
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
        top_n: int | None = None,
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
            >>> adata = pt.ds.sc_sim_augur()
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
