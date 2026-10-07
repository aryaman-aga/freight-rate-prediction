"""Data loading and preprocessing utilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from freight_rate.config import CONFIG

logger = logging.getLogger(__name__)


def load_train() -> pd.DataFrame:
    """Load training data with target column."""
    df = pd.read_csv(CONFIG.paths.train_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded train: %d rows, %d columns", len(df), df.shape[1])
    return df


def load_validation() -> pd.DataFrame:
    """Load validation data (no target)."""
    df = pd.read_csv(CONFIG.paths.validation_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded validation: %d rows", len(df))
    return df


def load_validation_template() -> pd.DataFrame:
    """Load validation predictions template."""
    df = pd.read_csv(CONFIG.paths.validation_template_path)
    logger.info("Loaded validation template: %d rows", len(df))
    return df


def load_december_chart() -> pd.DataFrame:
    """Load December chart inputs."""
    df = pd.read_csv(CONFIG.paths.december_chart_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded December chart: %d rows", len(df))
    return df


def _haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine distance in miles."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_base_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Add base features that don't require target information."""
    df = df.copy()

    # Distance features
    df["haversine"] = _haversine(
        df["pickup_lat"].values,
        df["pickup_lon"].values,
        df["delivery_lat"].values,
        df["delivery_lon"].values,
    )
    df["road_ratio"] = df["distance"] / df["haversine"]

    # Lane identifier
    df["lane"] = df["pickup"] + ">" + df["delivery"]

    # Region buckets (simple geographic grouping by lat/lon)
    # Using rough US census regions
    def region(lat: float, lon: float) -> str:
        if lat >= 40:
            return "North" if lon <= -100 else "Northeast"
        elif lat >= 34:
            return "Central" if lon <= -100 else "Mid-Atlantic"
        else:
            return "South" if lon <= -100 else "Southeast"

    df["pickup_region"] = df.apply(lambda r: region(r["pickup_lat"], r["pickup_lon"]), axis=1)
    df["delivery_region"] = df.apply(lambda r: region(r["delivery_lat"], r["delivery_lon"]), axis=1)

    # Date features
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["is_month_end"] = (df["date"].dt.is_month_end).astype(int)
    df["is_quarter_end"] = (df["date"].dt.is_quarter_end).astype(int)

    # Holiday distance features (days to nearest holiday)
    # Thanksgiving: 4th Thursday of November
    # Christmas: Dec 25
    # New Year: Jan 1
    year = df["date"].dt.year
    # Thanksgiving (4th Thursday of Nov)
    nov1 = pd.to_datetime(year.astype(str) + "-11-01")
    first_thu = nov1 + pd.to_timedelta((3 - nov1.dt.dayofweek) % 7, unit="D")
    thanksgiving = first_thu + pd.to_timedelta(21, unit="D")  # 4th Thursday
    # Christmas
    christmas = pd.to_datetime(year.astype(str) + "-12-25")
    # New Year
    new_year = pd.to_datetime((year + 1).astype(str) + "-01-01")

    df["days_to_thanksgiving"] = (thanksgiving - df["date"]).dt.days
    df["days_to_christmas"] = (christmas - df["date"]).dt.days
    df["days_to_new_year"] = (new_year - df["date"]).dt.days

    # Cyclical encoding (only for features with full training coverage)
    for col, period in CONFIG.features.cyclical_features.items():
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    # Equipment encoding
    df["equipment"] = df["equipment"].astype("category")

    # Weight features - fix negative weights (typos), clip at 0
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].abs()  # negative weights are typos
    df["weight"] = df["weight"].clip(lower=1000)  # minimum reasonable weight
    df["weight"] = df["weight"].fillna(df["weight"].median())

    # Short distance flag (extrapolation risk)
    df["is_short_distance"] = (df["distance"] < 150).astype(int)

    # Market index / quote signal missing flags
    df["mi_missing"] = df["market_index"].isna().astype(int)
    df["qs_missing"] = df["quote_signal"].isna().astype(int)
    df["market_index"] = df["market_index"].fillna(df["market_index"].median())
    df["quote_signal"] = df["quote_signal"].fillna(df["quote_signal"].median())

    # Interactions
    df["dist_x_equip"] = df["distance"] * df["equipment"].cat.codes
    # Dampen holiday effects for short distances (extrapolation)
    for h in ["days_to_thanksgiving", "days_to_christmas", "days_to_new_year"]:
        df[f"{h}_x_short"] = df[h] * df["is_short_distance"]

    if is_train:
        # Target - use log(rate) for better MAPE optimization
        df["posted_rate"] = df["posted_rate"].astype(float)
        df["rpm"] = df["posted_rate"] / df["distance"]
        df["log_rate"] = np.log1p(df["posted_rate"])
        df["log_rpm"] = np.log1p(df["rpm"])
        # Target for training is log_rate
        df["target"] = df["log_rate"]

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "target") -> list[str]:
    """Get list of feature column names (excluding target and identifiers)."""
    exclude = {
        target_col,
        "load_id",
        "date",
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "lane",
        "pickup_region",
        "delivery_region",
        "rpm",
        "log_rate",
        "log_rpm",
        "target",
        "posted_rate",
    }
    return [
        c
        for c in df.columns
        if c not in exclude and not c.startswith("te_") and not c.startswith("freq_")
    ]


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare all datasets with base features."""
    train = load_train()
    val = load_validation()
    dec = load_december_chart()

    # December chart is missing lat/lon - add from train data (Lexington, Fort Wayne coords are consistent)
    city_coords = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    city_coords = (
        pd.concat(
            [
                city_coords,
                train.groupby("delivery")[["delivery_lat", "delivery_lon"]]
                .first()
                .rename(columns={"delivery_lat": "pickup_lat", "delivery_lon": "pickup_lon"}),
            ]
        )
        .groupby(level=0)
        .first()
    )

    # Add coords to December chart
    for city_col, lat_col, lon_col in [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]:
        coord_map = city_coords.reset_index().set_index("index")
        dec[lat_col] = dec[city_col].map(coord_map["pickup_lat"])
        dec[lon_col] = dec[city_col].map(coord_map["pickup_lon"])

    # Add daily market_index/quote_signal to December chart from validation (INPUTS, no leakage)
    dec_daily = (
        val.groupby("date")
        .agg(
            market_index_daily=("market_index", "mean"),
            quote_signal_daily=("quote_signal", "mean"),
        )
        .reset_index()
    )
    dec = dec.merge(dec_daily, on="date", how="left")
    # Fill with validation-period overall means
    dec["market_index"] = dec["market_index_daily"].fillna(val["market_index"].mean())
    dec["quote_signal"] = dec["quote_signal_daily"].fillna(val["quote_signal"].mean())
    dec = dec.drop(columns=["market_index_daily", "quote_signal_daily"])

    train = add_base_features(train, is_train=True)
    val = add_base_features(val, is_train=False)
    dec = add_base_features(dec, is_train=False)

    return train, val, dec
