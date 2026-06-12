"""Smoke tests for yield_model (sklearn-based, no GEE required)."""
import numpy as np
import pandas as pd
import pytest
from src.yield_model import build_models, cross_validate, fit_and_predict, evaluate


@pytest.fixture
def yield_cfg():
    return {
        "models": {
            "random_forest": {"n_estimators": 20, "max_depth": 5},
            "gradient_boosting": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
        },
        "cv_folds": 3,
    }


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(0, 1, (n, 4))
    y = 3.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(0, 0.2, n)
    return pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"]), pd.Series(y)


class TestBuildModels:

    def test_returns_expected_models(self, yield_cfg):
        models = build_models(yield_cfg)
        assert "random_forest" in models
        assert "gradient_boosting" in models


class TestCrossValidate:

    def test_returns_dataframe(self, yield_cfg, synthetic_data):
        X, y = synthetic_data
        models = build_models(yield_cfg)
        cv = cross_validate(models, X, y, folds=yield_cfg["cv_folds"])
        assert isinstance(cv, pd.DataFrame)
        assert "r2_mean" in cv.columns
        assert len(cv) == 2


class TestFitAndPredict:

    def test_predictions_shape(self, yield_cfg, synthetic_data):
        X, y = synthetic_data
        models = build_models(yield_cfg)
        pred = fit_and_predict(models["random_forest"], X[:80], y[:80], X[80:])
        assert len(pred) == 20


class TestEvaluate:

    def test_metrics_keys(self):
        y_true = np.array([3.0, 3.5, 4.0, 2.8])
        y_pred = np.array([3.1, 3.4, 4.2, 2.7])
        m = evaluate(y_true, y_pred)
        assert "r2" in m
        assert "rmse" in m
        assert "bias" in m

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = evaluate(y, y)
        assert abs(m["r2"] - 1.0) < 1e-10
        assert abs(m["rmse"]) < 1e-10
        assert abs(m["bias"]) < 1e-10
