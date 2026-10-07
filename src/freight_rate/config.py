"""Configuration management for the freight rate prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml or .git."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback: assume standard src layout
    return current.parents[3]


@dataclass(frozen=True)
class Paths:
    """All filesystem paths used by the pipeline."""

    root: Path = field(default_factory=_find_project_root)
    data_dir: Path = field(init=False)
    train_path: Path = field(init=False)
    validation_path: Path = field(init=False)
    validation_template_path: Path = field(init=False)
    december_chart_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    predictions_path: Path = field(init=False)
    december_filled_path: Path = field(init=False)
    model_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        r = self.root
        object.__setattr__(self, "data_dir", r / "data")
        object.__setattr__(self, "train_path", self.data_dir / "train_test.csv")
        object.__setattr__(self, "validation_path", self.data_dir / "validation.csv")
        object.__setattr__(
            self, "validation_template_path", self.data_dir / "validation_predictions_template.csv"
        )
        object.__setattr__(self, "december_chart_path", self.data_dir / "december_chart_inputs.csv")
        object.__setattr__(self, "output_dir", r / "output")
        object.__setattr__(self, "predictions_path", self.output_dir / "validation_predictions.csv")
        object.__setattr__(
            self, "december_filled_path", self.data_dir / "december_chart_inputs.csv"
        )  # overwrite in place
        object.__setattr__(self, "model_dir", r / "models")
        object.__setattr__(self, "reports_dir", r / "reports")


@dataclass(frozen=True)
class SplitConfig:
    """Time-based cross-validation split configuration."""

    n_splits: int = 4
    test_months: int = 1  # each fold tests on 1 month
    gap_days: int = 0
    # Expanding window: train on Jan-Jun -> test Jul, Jan-Jul -> test Aug, etc.
    first_train_end_month: int = 6  # June (month 6)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""

    target_encoding_smoothing: float = 10.0
    target_encoding_min_samples: int = 5
    # Columns for target/frequency encoding (computed out-of-fold only)
    te_columns: tuple[str, ...] = (
        "pickup",
        "delivery",
        "lane",
        "equipment",
        "pickup_region",
        "delivery_region",
    )
    # Cyclical encoding periods
    cyclical_features: dict[str, int] = field(
        default_factory=lambda: {"month": 12, "week": 52, "dow": 7, "day_of_year": 365}
    )


@dataclass(frozen=True)
class ModelConfig:
    """Model training configuration."""

    random_state: int = 42
    n_jobs: int = -1
    # LightGBM
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "mape",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 255,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "force_col_wise": True,
        }
    )
    # XGBoost
    xgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "eval_metric": "mape",
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "tree_method": "hist",
            "verbosity": 0,
        }
    )
    # Optuna
    optuna_n_trials: int = 30
    optuna_timeout: int = 1800  # seconds


@dataclass(frozen=True)
class Config:
    """Root configuration aggregator."""

    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            paths=Paths(),
            split=SplitConfig(**data.get("split", {})),
            features=FeatureConfig(**data.get("features", {})),
            models=ModelConfig(**data.get("models", {})),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "split": self.split.__dict__,
            "features": self.features.__dict__,
            "models": {
                "random_state": self.models.random_state,
                "n_jobs": self.models.n_jobs,
                "lgbm_params": self.models.lgbm_params,
                "xgb_params": self.models.xgb_params,
                "optuna_n_trials": self.models.optuna_n_trials,
                "optuna_timeout": self.models.optuna_timeout,
            },
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


# Global config instance
CONFIG = Config()
