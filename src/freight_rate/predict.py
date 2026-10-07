"""Prediction pipeline for validation and December chart."""

from __future__ import annotations

import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from freight_rate.config import CONFIG
from freight_rate.data import (
    add_base_features,
    load_december_chart,
    load_validation,
    load_validation_template,
)
from freight_rate.features import (
    apply_frequency_encodings,
    apply_target_encodings,
)
from freight_rate.models import LightGBMModel, RidgeModel, XGBoostModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_artifacts() -> tuple:
    """Load all trained models and preprocessing artifacts."""
    model_dir = CONFIG.paths.model_dir

    # Load multiple LGBM seed models
    lgbm_models = []
    n_seeds = 5
    n_seeds_path = model_dir / "n_seeds.json"
    if n_seeds_path.exists():
        with open(n_seeds_path) as f:
            n_seeds = json.load(f)

    for i in range(n_seeds):
        seed_model_path = model_dir / f"lgbm_final_seed{i}.txt"
        if seed_model_path.exists():
            lgbm_models.append(LightGBMModel.load(seed_model_path))
        else:
            # Fallback to main model
            lgbm_models.append(LightGBMModel.load(model_dir / "lgbm_final.txt"))
            break

    xgb = XGBoostModel.load(model_dir / "xgb_final.json")
    ridge = RidgeModel.load(model_dir / "ridge_final.pkl")

    with open(model_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)

    te_encodings = joblib.load(model_dir / "target_encodings.joblib")
    freq_encodings = joblib.load(model_dir / "freq_encodings.joblib")

    # Load ensemble weights (or recompute)
    weights_path = model_dir / "ensemble_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights = json.load(f)
    else:
        # Default equal weights
        weights = {"lgbm": 1 / 3, "xgb": 1 / 3, "ridge": 1 / 3}

    return lgbm_models, xgb, ridge, feature_cols, te_encodings, freq_encodings, weights


def prepare_validation_features(
    val: pd.DataFrame,
    feature_cols: list,
    te_encodings: dict,
    freq_encodings: dict,
) -> pd.DataFrame:
    """Prepare validation features with all engineering steps."""
    # Add base features
    val = add_base_features(val, is_train=False)

    # Apply target encodings
    global_mean = 2373.98  # train mean posted_rate (from EDA)
    val = apply_target_encodings(val, te_encodings, list(CONFIG.features.te_columns), global_mean)

    # Apply frequency encodings
    val = apply_frequency_encodings(val, freq_encodings, list(CONFIG.features.te_columns))

    # Lane features (need train stats - load from saved or recompute)
    # For now, we'll need to compute these from train data again
    # TODO: save lane/city stats during training
    return val


def prepare_december_features(
    dec: pd.DataFrame,
    val: pd.DataFrame,  # needed for daily mi/qs
    feature_cols: list,
    te_encodings: dict,
    freq_encodings: dict,
) -> pd.DataFrame:
    """Prepare December chart features."""
    # Add base features (includes merging daily mi/qs from val)
    from freight_rate.data import add_base_features

    dec = add_base_features(dec, is_train=False)

    # Apply encodings
    global_mean = 2373.98
    dec = apply_target_encodings(dec, te_encodings, list(CONFIG.features.te_columns), global_mean)
    dec = apply_frequency_encodings(dec, freq_encodings, list(CONFIG.features.te_columns))

    return dec


def predict_all(
    val: pd.DataFrame,
    dec: pd.DataFrame,
    lgbm_models: list,
    xgb: XGBoostModel,
    ridge: RidgeModel,
    feature_cols: list,
    weights: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions for validation and December chart."""
    X_val = val[feature_cols]
    X_dec = dec[feature_cols]

    # Ridge needs numeric features only
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns.tolist()
    X_val_numeric = val[numeric_cols]
    X_dec_numeric = dec[numeric_cols]

    # Average predictions across LGBM seeds (convert from log scale with expm1)
    lgbm_val_preds = np.column_stack([np.expm1(m.predict(X_val)) for m in lgbm_models])
    lgbm_val = np.mean(lgbm_val_preds, axis=1)
    xgb_val = np.expm1(xgb.predict(X_val))
    ridge_val = np.expm1(ridge.predict(X_val_numeric))

    lgbm_dec_preds = np.column_stack([np.expm1(m.predict(X_dec)) for m in lgbm_models])
    lgbm_dec = np.mean(lgbm_dec_preds, axis=1)
    xgb_dec = np.expm1(xgb.predict(X_dec))
    ridge_dec = np.expm1(ridge.predict(X_dec_numeric))

    # Ensemble
    val_pred = weights["lgbm"] * lgbm_val + weights["xgb"] * xgb_val + weights["ridge"] * ridge_val
    dec_pred = weights["lgbm"] * lgbm_dec + weights["xgb"] * xgb_dec + weights["ridge"] * ridge_dec

    # Ensure positive (redundant but safe)
    val_pred = np.maximum(val_pred, 1.0)
    dec_pred = np.maximum(dec_pred, 1.0)

    return val_pred, dec_pred


def save_predictions(
    val: pd.DataFrame,
    val_pred: np.ndarray,
    dec: pd.DataFrame,
    dec_pred: np.ndarray,
) -> None:
    """Save predictions in required submission format."""
    # Ensure output directory exists
    CONFIG.paths.output_dir.mkdir(parents=True, exist_ok=True)

    # Validation predictions
    template = load_validation_template()
    pred_df = pd.DataFrame({"load_id": val["load_id"].values, "predicted_rate": val_pred})
    # Ensure order matches template
    pred_df = pred_df.set_index("load_id").loc[template["load_id"]].reset_index()
    pred_df.to_csv(CONFIG.paths.predictions_path, index=False)
    logger.info(
        "Saved validation predictions to %s (%d rows)", CONFIG.paths.predictions_path, len(pred_df)
    )

    # December chart - overwrite the input file with predicted_rate filled
    dec_filled = dec.copy()
    dec_filled["predicted_rate"] = dec_pred
    # Keep original 7 columns in order
    output_cols = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
    dec_filled[output_cols].to_csv(CONFIG.paths.december_filled_path, index=False)
    logger.info(
        "Saved December chart predictions to %s (%d rows)",
        CONFIG.paths.december_filled_path,
        len(dec_filled),
    )


def main() -> None:
    logger.info("=" * 60)
    logger.info("FREIGHT RATE PREDICTION - INFERENCE PIPELINE")
    logger.info("=" * 60)

    # Load artifacts
    lgbm_models, xgb, ridge, feature_cols, te_encodings, freq_encodings, weights = load_artifacts()
    logger.info("Loaded models and artifacts. Ensemble weights: %s", weights)
    logger.info("Loaded %d LGBM seed models", len(lgbm_models))

    # Load raw data
    val = load_validation()
    dec = load_december_chart()

    # Prepare features for validation
    logger.info("Preparing validation features...")
    val = add_base_features(val, is_train=False)

    # December chart is missing lat/lon - add from train data
    from freight_rate.data import load_train

    train_for_coords = load_train()
    city_coords = train_for_coords.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    city_coords = (
        pd.concat(
            [
                city_coords,
                train_for_coords.groupby("delivery")[["delivery_lat", "delivery_lon"]]
                .first()
                .rename(columns={"delivery_lat": "pickup_lat", "delivery_lon": "pickup_lon"}),
            ]
        )
        .groupby(level=0)
        .first()
    )

    for city_col, lat_col, lon_col in [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]:
        coord_map = city_coords.reset_index().set_index("index")
        dec[lat_col] = dec[city_col].map(coord_map["pickup_lat"])
        dec[lon_col] = dec[city_col].map(coord_map["pickup_lon"])

    # Add daily mi/qs to December chart from validation (INPUTS, no leakage)
    dec_daily = (
        val.groupby("date")
        .agg(
            market_index_daily=("market_index", "mean"),
            quote_signal_daily=("quote_signal", "mean"),
        )
        .reset_index()
    )
    dec = dec.merge(dec_daily, on="date", how="left")
    dec["market_index"] = dec["market_index_daily"].fillna(val["market_index"].mean())
    dec["quote_signal"] = dec["quote_signal_daily"].fillna(val["quote_signal"].mean())
    dec = dec.drop(columns=["market_index_daily", "quote_signal_daily"])

    dec = add_base_features(dec, is_train=False)

    global_mean = 2373.98
    val = apply_target_encodings(val, te_encodings, list(CONFIG.features.te_columns), global_mean)
    val = apply_frequency_encodings(val, freq_encodings, list(CONFIG.features.te_columns))
    dec = apply_target_encodings(dec, te_encodings, list(CONFIG.features.te_columns), global_mean)
    dec = apply_frequency_encodings(dec, freq_encodings, list(CONFIG.features.te_columns))

    # For lane/city features, we need train stats
    # Load train and compute (in production, these would be saved artifacts)
    from freight_rate.data import load_train

    train = load_train()
    train = add_base_features(train, is_train=True)

    # Lane features
    lane_stats = (
        train.groupby("lane")
        .agg(
            lane_mean_rate=("posted_rate", "mean"),
            lane_median_rate=("posted_rate", "median"),
            lane_std_rate=("posted_rate", "std"),
            lane_count=("posted_rate", "size"),
            lane_mean_rpm=("rpm", "mean"),
            lane_median_rpm=("rpm", "median"),
        )
        .reset_index()
    )

    pu_stats = (
        train.groupby("pickup")
        .agg(
            pu_mean_rate=("posted_rate", "mean"),
            pu_count=("posted_rate", "size"),
            pu_mean_rpm=("rpm", "mean"),
        )
        .reset_index()
    )

    de_stats = (
        train.groupby("delivery")
        .agg(
            de_mean_rate=("posted_rate", "mean"),
            de_count=("posted_rate", "size"),
            de_mean_rpm=("rpm", "mean"),
        )
        .reset_index()
    )

    for df in [val, dec]:
        df = df.merge(lane_stats, on="lane", how="left")
        df = df.merge(pu_stats, on="pickup", how="left")
        df = df.merge(de_stats, on="delivery", how="left")
        for c in [
            "lane_mean_rate",
            "lane_median_rate",
            "lane_std_rate",
            "lane_mean_rpm",
            "lane_median_rpm",
        ]:
            df[c] = df[c].fillna(global_mean if "rate" in c else train["rpm"].mean())
        df["lane_count"] = df["lane_count"].fillna(0)
        df["lane_seen"] = (df["lane_count"] > 0).astype(int)
        for c in ["pu_mean_rate", "de_mean_rate"]:
            df[c] = df[c].fillna(global_mean)
        for c in ["pu_mean_rpm", "de_mean_rpm"]:
            df[c] = df[c].fillna(train["rpm"].mean())
        df["pu_count"] = df["pu_count"].fillna(0)
        df["de_count"] = df["de_count"].fillna(0)
        df["pu_seen"] = (df["pu_count"] > 0).astype(int)
        df["de_seen"] = (df["de_count"] > 0).astype(int)

    # Merge daily mi/qs from val
    dec_daily = (
        val.groupby("date")
        .agg(
            market_index_daily=("market_index", "mean"),
            quote_signal_daily=("quote_signal", "mean"),
        )
        .reset_index()
    )
    dec = dec.merge(dec_daily, on="date", how="left")
    dec["market_index"] = dec["market_index_daily"].fillna(val["market_index"].mean())
    dec["quote_signal"] = dec["quote_signal_daily"].fillna(val["quote_signal"].mean())
    dec = dec.drop(columns=["market_index_daily", "quote_signal_daily"])

    # Generate predictions
    logger.info("Generating predictions...")
    val_pred, dec_pred = predict_all(val, dec, lgbm_models, xgb, ridge, feature_cols, weights)

    # Save
    save_predictions(val, val_pred, dec, dec_pred)

    logger.info("Prediction complete!")


if __name__ == "__main__":
    main()
