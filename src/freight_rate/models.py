"""Model definitions and training utilities."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from freight_rate.config import CONFIG

logger = logging.getLogger(__name__)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, handling zeros."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Percentage Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all evaluation metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "medape": median_ape(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


class BaseModel:
    """Abstract base model interface."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> BaseModel:
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        raise NotImplementedError


class LightGBMModel(BaseModel):
    """LightGBM regressor wrapper."""

    def __init__(self, params: dict[str, Any] | None = None, num_boost_round: int = 5000):
        self.params = params or CONFIG.models.lgbm_params.copy()
        self.num_boost_round = num_boost_round
        self.model: lgb.Booster | None = None
        self.best_iteration: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 200,
    ) -> LightGBMModel:
        train_data = lgb.Dataset(X, label=y, free_raw_data=False)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
            valid_sets.append(val_data)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self.best_iteration = self.model.best_iteration
        logger.info("LightGBM trained: best_iteration=%d", self.best_iteration)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X, num_iteration=self.best_iteration)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save_model(str(path))
        logger.info("LightGBM saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        model = cls()
        model.model = lgb.Booster(model_file=str(path))
        model.best_iteration = model.model.best_iteration
        return model

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return pd.Series(
            self.model.feature_importance(importance_type=importance_type),
            index=self.model.feature_name(),
        ).sort_values(ascending=False)


class XGBoostModel(BaseModel):
    """XGBoost regressor wrapper."""

    def __init__(self, params: dict[str, Any] | None = None, num_boost_round: int = 5000):
        self.params = params or CONFIG.models.xgb_params.copy()
        self.num_boost_round = num_boost_round
        self.model: xgb.Booster | None = None
        self.best_iteration: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 200,
    ) -> XGBoostModel:
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_iteration = self.model.best_iteration
        logger.info("XGBoost trained: best_iteration=%d", self.best_iteration)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save_model(str(path))
        logger.info("XGBoost saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> XGBoostModel:
        model = cls()
        model.model = xgb.Booster()
        model.model.load_model(str(path))
        model.best_iteration = model.model.best_iteration
        return model


class RidgeModel(BaseModel):
    """Ridge regression baseline."""

    def __init__(self, params: dict[str, Any] | None = None):
        alpha = params.get("alpha", 1.0) if params else 1.0
        self.model = Ridge(alpha=alpha, random_state=CONFIG.models.random_state)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> RidgeModel:
        self.model.fit(X, y)
        logger.info("Ridge fitted: alpha=%s", self.model.alpha)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: Path) -> RidgeModel:
        model = cls()
        with open(path, "rb") as f:
            model.model = pickle.load(f)
        return model


class EnsembleModel(BaseModel):
    """Weighted ensemble of multiple models."""

    def __init__(self, models: list[tuple[BaseModel, float]]):
        """
        Args:
            models: List of (model, weight) tuples. Weights should sum to 1.
        """
        self.models = models
        self.weights = np.array([w for _, w in models])
        assert abs(self.weights.sum() - 1.0) < 1e-6, "Weights must sum to 1"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m, _ in self.models])
        return preds @ self.weights

    def save(self, path: Path) -> None:
        # Save each sub-model
        path.parent.mkdir(parents=True, exist_ok=True)
        for i, (m, _w) in enumerate(self.models):
            m.save(path.with_suffix(f".model{i}{path.suffix}"))
        # Save weights
        np.save(path.with_suffix(".weights.npy"), self.weights)

    @classmethod
    def load(cls, path: Path, model_types: list[type]) -> EnsembleModel:
        models = []
        for i, mtype in enumerate(model_types):
            m = mtype.load(path.with_suffix(f".model{i}{path.suffix}"))
            w = np.load(path.with_suffix(".weights.npy"))[i]
            models.append((m, float(w)))
        return cls(models)


def train_model_cv(
    model_cls: type,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    params: dict[str, Any] | None = None,
) -> tuple[list[BaseModel], list[dict[str, float]]]:
    """
    Train model using cross-validation splits.

    Returns:
        List of fitted models (one per fold)
        List of validation metrics per fold (on dollar scale)
    """
    models = []
    fold_metrics = []

    for split in splits:
        X_train = X.iloc[split.train_idx]
        y_train = y.iloc[split.train_idx]
        X_val = X.iloc[split.val_idx]
        y_val = y.iloc[split.val_idx]

        model = model_cls(params=params)
        model.fit(X_train, y_train, X_val, y_val)
        models.append(model)

        y_pred_log = model.predict(X_val)
        # Convert from log scale to dollar scale for metrics
        y_val_dollar = np.expm1(y_val.values)
        y_pred_dollar = np.expm1(y_pred_log)
        metrics = compute_metrics(y_val_dollar, y_pred_dollar)
        fold_metrics.append(metrics)
        logger.info(
            "Fold %s: MAPE=%.2f%%, MAE=%.2f, RMSE=%.2f, R2=%.4f",
            split.fold_name,
            metrics["mape"],
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

    return models, fold_metrics


def aggregate_cv_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate CV metrics: mean ± std."""
    keys = fold_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg
