"""Tests for the freight rate prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate.config import CONFIG
from freight_rate.data import add_base_features, get_feature_columns, load_train, load_validation
from freight_rate.features import apply_target_encodings, compute_oof_target_encodings
from freight_rate.models import compute_metrics
from freight_rate.validation import make_time_splits, validate_no_leakage


class TestDataLoading:
    """Test data loading and basic structure."""

    def test_train_loads(self):
        df = load_train()
        assert len(df) == 48000
        assert "posted_rate" in df.columns
        assert df["load_id"].nunique() == 48000
        assert np.issubdtype(df["date"].dtype, np.datetime64)

    def test_validation_loads(self):
        df = load_validation()
        assert len(df) == 12000
        assert "posted_rate" not in df.columns
        assert df["load_id"].nunique() == 12000

    def test_date_ranges(self):
        train = load_train()
        val = load_validation()
        assert train["date"].max() < val["date"].min(), "Train must be before validation"

    def test_no_duplicate_ids(self):
        train = load_train()
        val = load_validation()
        assert train["load_id"].duplicated().sum() == 0
        assert val["load_id"].duplicated().sum() == 0


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_base_features(self):
        train = load_train()
        train_fe = add_base_features(train, is_train=True)
        assert "haversine" in train_fe.columns
        assert "road_ratio" in train_fe.columns
        assert "lane" in train_fe.columns
        assert "month_sin" in train_fe.columns
        assert "weight_missing" in train_fe.columns

    def test_feature_columns_excludes_target(self):
        train = load_train()
        train_fe = add_base_features(train, is_train=True)
        feat_cols = get_feature_columns(train_fe)
        assert "posted_rate" not in feat_cols
        assert "load_id" not in feat_cols
        assert "date" not in feat_cols
        assert "pickup" not in feat_cols

    def test_target_encoding_oof(self):
        train = load_train()
        train_fe = add_base_features(train, is_train=True)
        # Use small subset for speed
        sub = train_fe.sample(1000, random_state=42).copy()
        sub["te_target"] = sub["posted_rate"]  # dummy
        encoded, encodings = compute_oof_target_encodings(
            sub, "posted_rate", ["equipment"], n_splits=3, random_state=42
        )
        assert "te_equipment" in encoded.columns
        assert encoded["te_equipment"].notna().all()
        assert "equipment" in encodings

    def test_apply_target_encodings(self):
        train = load_train()
        train_fe = add_base_features(train, is_train=True)
        sub = train_fe.sample(500, random_state=42).copy()
        encoded, encodings = compute_oof_target_encodings(
            sub, "posted_rate", ["equipment"], n_splits=3, random_state=42
        )
        # Apply to new data
        new_data = sub.copy()
        new_data = apply_target_encodings(
            new_data, encodings, ["equipment"], sub["posted_rate"].mean()
        )
        assert "te_equipment" in new_data.columns
        assert new_data["te_equipment"].notna().all()


class TestValidationSplits:
    """Test time-based CV splits."""

    def test_time_splits_created(self):
        train = load_train()
        train_fe = add_base_features(train, is_train=True)
        splits = make_time_splits(train_fe)
        assert len(splits) >= 3
        for split in splits:
            assert len(split.train_idx) > 0
            assert len(split.val_idx) > 0
            assert validate_no_leakage(split.train_idx, split.val_idx, train_fe)

    def test_no_leakage(self):
        train = load_train()
        train_fe = add_base_features(train, is_train=True)
        splits = make_time_splits(train_fe)
        for split in splits:
            assert validate_no_leakage(split.train_idx, split.val_idx, train_fe)

    def test_expanding_window(self):
        train = load_train()
        train_fe = add_base_features(train, is_train=True)
        splits = make_time_splits(train_fe)
        # Each fold should have more training data
        train_sizes = [len(s.train_idx) for s in splits]
        assert all(train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1))


class TestMetrics:
    """Test metric computations."""

    def test_mape(self):
        y_true = [100, 200, 300]
        y_pred = [110, 190, 330]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mape"] > 0
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["r2"] <= 1.0

    def test_mape_with_zeros(self):
        y_true = [0, 100, 200]
        y_pred = [10, 110, 190]
        metrics = compute_metrics(y_true, y_pred)
        # Should not crash, just ignore zeros
        assert metrics["mape"] >= 0


class TestSubmissionFormat:
    """Test submission file format requirements."""

    def test_validation_template_structure(self):
        template = pd.read_csv(CONFIG.paths.validation_template_path)
        assert list(template.columns) == ["load_id", "predicted_rate"]
        assert len(template) == 12000
        assert template["load_id"].str.match(r"TE-\d{6}").all()

    def test_december_chart_structure(self):
        dec = pd.read_csv(CONFIG.paths.december_chart_path)
        expected_cols = [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
        assert list(dec.columns) == expected_cols
        assert len(dec) == 31
        assert dec["date"].nunique() == 31


class TestConfig:
    """Test configuration loading."""

    def test_config_defaults(self):
        assert CONFIG.split.n_splits == 4
        assert CONFIG.models.random_state == 42
        assert "equipment" in CONFIG.features.te_columns

    def test_config_yaml_roundtrip(self, tmp_path):
        yaml_path = tmp_path / "test_config.yaml"
        CONFIG.to_yaml(yaml_path)
        loaded = CONFIG.__class__.from_yaml(yaml_path)
        assert loaded.split.n_splits == CONFIG.split.n_splits
        assert loaded.models.random_state == CONFIG.models.random_state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
