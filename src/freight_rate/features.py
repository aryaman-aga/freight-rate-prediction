"""Target encoding and advanced feature engineering (out-of-fold safe)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from freight_rate.config import CONFIG

logger = logging.getLogger(__name__)


def _smooth_target_encode(
    train_series: pd.Series,
    target: pd.Series,
    test_series: pd.Series,
    smoothing: float,
    min_samples: int,
    global_mean: float,
) -> tuple[pd.Series, pd.Series]:
    """Compute smoothed target encoding for a categorical feature."""
    # Group stats on train
    stats = target.groupby(train_series).agg(["mean", "count"])
    stats.columns = ["mean", "count"]

    # Smoothing weight
    weight = stats["count"] / (stats["count"] + smoothing)

    # Smoothed mean
    smoothed = weight * stats["mean"] + (1 - weight) * global_mean

    # For categories with too few samples, use global mean
    smoothed[stats["count"] < min_samples] = global_mean

    # Map to train and test
    train_encoded = train_series.map(smoothed).fillna(global_mean)
    test_encoded = test_series.map(smoothed).fillna(global_mean)

    return train_encoded, test_encoded


def compute_oof_target_encodings(
    df: pd.DataFrame,
    target_col: str,
    cat_cols: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Compute out-of-fold target encodings for categorical columns.

    Returns:
        - DataFrame with added te_<col> columns
        - Dict mapping column -> full-train encoding Series (for test-time transform)
    """
    df = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = df[target_col].mean()

    # Store full-train encodings for later use on validation/test
    full_encodings = {}

    for col in cat_cols:
        if col not in df.columns:
            logger.warning("Column %s not found, skipping target encoding", col)
            continue

        te_col = f"te_{col}"
        df[te_col] = np.nan

        fold_encodings = []
        for train_idx, val_idx in kf.split(df):
            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]

            train_enc, val_enc = _smooth_target_encode(
                train_fold[col],
                train_fold[target_col],
                val_fold[col],
                CONFIG.features.target_encoding_smoothing,
                CONFIG.features.target_encoding_min_samples,
                global_mean,
            )

            df.loc[val_fold.index, te_col] = val_enc.values
            fold_encodings.append((train_fold.index, train_enc))

        # Compute full-train encoding for this column (used at inference)
        full_enc, _ = _smooth_target_encode(
            df[col],
            df[target_col],
            df[col],
            CONFIG.features.target_encoding_smoothing,
            CONFIG.features.target_encoding_min_samples,
            global_mean,
        )
        full_encodings[col] = full_enc

        # Sanity check
        nan_count = df[te_col].isna().sum()
        if nan_count:
            logger.warning(
                "OOF encoding for %s has %d NaNs, filling with global mean", col, nan_count
            )
            df[te_col] = df[te_col].fillna(global_mean)

    return df, full_encodings


def apply_target_encodings(
    df: pd.DataFrame,
    encodings: dict[str, pd.Series],
    cat_cols: list[str],
    global_mean: float,
) -> pd.DataFrame:
    """Apply precomputed target encodings to new data (validation/test)."""
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns or col not in encodings:
            continue
        te_col = f"te_{col}"
        mapped = df[col].map(encodings[col])
        # Handle categorical columns by converting to float first
        if mapped.dtype.name == "category":
            mapped = mapped.astype(float)
        df[te_col] = mapped.fillna(global_mean)
    return df


def compute_frequency_encodings(
    df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Compute frequency encodings (count / total) for categorical columns."""
    df = df.copy()
    full_encodings = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        freq = df[col].value_counts() / len(df)
        full_encodings[col] = freq
        df[f"freq_{col}"] = df[col].map(freq).fillna(0)

    return df, full_encodings


def apply_frequency_encodings(
    df: pd.DataFrame,
    encodings: dict[str, pd.Series],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Apply precomputed frequency encodings."""
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns or col not in encodings:
            continue
        df[f"freq_{col}"] = df[col].map(encodings[col]).fillna(0)
    return df


def add_lane_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    dec: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add lane-level aggregate features computed only on train."""
    # Lane statistics from train
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

    # Smooth with global
    global_mean = train["posted_rate"].mean()
    global_rpm = train["rpm"].mean()

    for df, name in [(train, "train"), (val, "val"), (dec, "dec")]:
        df = df.merge(lane_stats, on="lane", how="left", suffixes=("", "_lane"))
        # Fill unseen lanes with global
        for c in [
            "lane_mean_rate",
            "lane_median_rate",
            "lane_std_rate",
            "lane_mean_rpm",
            "lane_median_rpm",
        ]:
            df[c] = df[c].fillna(global_mean if "rate" in c else global_rpm)
        df["lane_count"] = df["lane_count"].fillna(0)

        # Lane familiarity
        df["lane_seen"] = (df["lane_count"] > 0).astype(int)
        logger.info("%s: %d/%d lanes seen in train", name, df["lane_seen"].sum(), len(df))

    return train, val, dec


def add_city_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    dec: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add city-level aggregate features."""
    # Pickup city stats
    pu_stats = (
        train.groupby("pickup")
        .agg(
            pu_mean_rate=("posted_rate", "mean"),
            pu_count=("posted_rate", "size"),
            pu_mean_rpm=("rpm", "mean"),
        )
        .reset_index()
    )

    # Delivery city stats
    de_stats = (
        train.groupby("delivery")
        .agg(
            de_mean_rate=("posted_rate", "mean"),
            de_count=("posted_rate", "size"),
            de_mean_rpm=("rpm", "mean"),
        )
        .reset_index()
    )

    global_mean = train["posted_rate"].mean()
    global_rpm = train["rpm"].mean()

    for df, _name in [(train, "train"), (val, "val"), (dec, "dec")]:
        df = df.merge(pu_stats, on="pickup", how="left")
        df = df.merge(de_stats, on="delivery", how="left")
        for c in ["pu_mean_rate", "de_mean_rate"]:
            df[c] = df[c].fillna(global_mean)
        for c in ["pu_mean_rpm", "de_mean_rpm"]:
            df[c] = df[c].fillna(global_rpm)
        df["pu_count"] = df["pu_count"].fillna(0)
        df["de_count"] = df["de_count"].fillna(0)
        df["pu_seen"] = (df["pu_count"] > 0).astype(int)
        df["de_seen"] = (df["de_count"] > 0).astype(int)

    return train, val, dec


def engineer_all_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    dec: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Full feature engineering pipeline.

    Returns:
        train, val, dec DataFrames with all features
        target_encodings dict for inference
        freq_encodings dict for inference
    """
    logger.info("Starting feature engineering...")

    # Target encodings (OOF on train only)
    cat_cols = list(CONFIG.features.te_columns)
    train, te_encodings = compute_oof_target_encodings(
        train, "posted_rate", cat_cols, n_splits=5, random_state=CONFIG.models.random_state
    )

    # Frequency encodings on full train
    train, freq_encodings = compute_frequency_encodings(train, cat_cols)

    # Apply to val and dec
    global_mean = train["posted_rate"].mean()
    val = apply_target_encodings(val, te_encodings, cat_cols, global_mean)
    dec = apply_target_encodings(dec, te_encodings, cat_cols, global_mean)

    val = apply_frequency_encodings(val, freq_encodings, cat_cols)
    dec = apply_frequency_encodings(dec, freq_encodings, cat_cols)

    # Lane features
    train, val, dec = add_lane_features(train, val, dec)

    # City features
    train, val, dec = add_city_features(train, val, dec)

    logger.info(
        "Feature engineering complete. Train shape: %s, Val: %s, Dec: %s",
        train.shape,
        val.shape,
        dec.shape,
    )
    return train, val, dec, te_encodings, freq_encodings
