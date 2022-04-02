import pytest
from sklearn.ensemble import RandomForestRegressor

import pertpy.tools
from pertpy.tools._augurpy import Params

ag = pertpy.tools.Augurpy(estimator="random_forest_regressor")


def test_creation():
    """Test output of create_estimator."""
    assert isinstance(ag.estimator, RandomForestRegressor)


def test_missing_value():
    """Test raising missing value error."""
    with pytest.raises(Exception):
        ag.create_estimator("this is no estimator")


def test_params():
    """Test parameters."""
    rf_estimator = ag.create_estimator("random_forest_classifier", Params(n_estimators=9, max_depth=10, penalty=13))
    lr_estimator = ag.create_estimator("logistic_regression_classifier", Params(penalty="elasticnet"))
    assert rf_estimator.get_params()["n_estimators"] == 9
    assert rf_estimator.get_params()["max_depth"] == 10
    assert lr_estimator.get_params()["penalty"] == "elasticnet"

    with pytest.raises(TypeError):
        ag.create_estimator("random_forest_regressor", Params(unvalid=10))
