"""Creates model object of desired type and populates it with desired parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from switchlang import switch


@dataclass
class Params:
    """Type signature for random forest and logistic regression parameters."""

    n_estimators: int = 100
    max_depth: int | None = None
    max_features: Literal["auto"] | Literal["log2"] | Literal["sqrt"] | int | float = 2
    penalty: Literal["l1"] | Literal["l2"] | Literal["elasticnet"] | Literal["none"] = "l2"
    random_state: int | None = None


def _raise_exception(exception_message: str):
    """Raise exception for invalid classifier input."""
    raise Exception(exception_message)


def create_estimator(
    classifier: (
        Literal["random_forest_classifier"]
        | Literal["random_forest_regressor"]
        | Literal["logistic_regression_classifier"]
    ),
    params: Params | None = None,
) -> RandomForestClassifier | RandomForestRegressor | LogisticRegression:
    """Creates a model object of the provided type and populates it with desired parameters.

    Args:
        classifier: classifier to use in calculating the area under the curve.
                    Either random forest classifier or logistic regression for categorical data
                    or random forest regressor for continous data
        params: parameters used to populate the model object.
                n_estimators defines the number of trees in the forest;
                max_depth specifies the maximal depth of each tree;
                max_features specifies the maximal number of features considered when looking at best split,
                    if int then consider max_features for each split
                    if float consider round(max_features*n_features)
                    if `auto` then max_features=n_features (default)
                    if `log2` then max_features=log2(n_features)
                    if `sqrt` then max_featuers=sqrt(n_features)
                penalty defines the norm of the penalty used in logistic regression
                    if `l1` then L1 penalty is added
                    if `l2` then L2 penalty is added (default)
                    if `elasticnet` both L1 and L2 penalties are added
                    if `none` no penalty is added
                random_state sets random model seed

    Returns:
        Estimator object.
    """
    if params is None:
        params = Params()
    with switch(classifier) as c:
        c.case(
            "random_forest_classifier",
            lambda: RandomForestClassifier(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                max_features=params.max_features,
                random_state=params.random_state,
            ),
        )
        c.case(
            "random_forest_regressor",
            lambda: RandomForestRegressor(
                n_estimators=params.n_estimators,
                max_depth=params.max_depth,
                max_features=params.max_features,
                random_state=params.random_state,
            ),
        )
        c.case(
            "logistic_regression_classifier",
            lambda: LogisticRegression(penalty=params.penalty, random_state=params.random_state),
        )
        c.default(lambda: _raise_exception("Missing valid input."))

    return c.result
