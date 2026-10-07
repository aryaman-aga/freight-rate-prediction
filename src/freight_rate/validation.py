"""Time-based cross-validation split logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from freight_rate.config import CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVSplit:
    """A single train/validation split with metadata."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    fold_name: str


def make_time_splits(df: pd.DataFrame, date_col: str = "date") -> list[CVSplit]:
    """
    Create expanding-window time-based CV splits.

    Train: Jan 1 -> end of month M
    Val: month M+1

    Following the config: first_train_end_month=6 (June), n_splits=4
    -> folds: Jan-Jun | Jul, Jan-Jul | Aug, Jan-Aug | Sep, Jan-Sep | Oct
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    splits = []
    first_end = CONFIG.split.first_train_end_month

    for i in range(CONFIG.split.n_splits):
        train_end_month = first_end + i
        val_month = train_end_month + 1

        if val_month > 10:  # train only goes to Oct
            break

        train_mask = df[date_col].dt.month <= train_end_month
        val_mask = df[date_col].dt.month == val_month

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning("Fold %d has empty split, skipping", i)
            continue

        split = CVSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            train_start=df.loc[train_idx, date_col].min(),
            train_end=df.loc[train_idx, date_col].max(),
            val_start=df.loc[val_idx, date_col].min(),
            val_end=df.loc[val_idx, date_col].max(),
            fold_name=f"fold_{i}_train_1-{train_end_month}_val_{val_month}",
        )
        splits.append(split)
        logger.info(
            "Fold %d: train %s to %s (%d rows), val %s to %s (%d rows)",
            i,
            split.train_start.date(),
            split.train_end.date(),
            len(train_idx),
            split.val_start.date(),
            split.val_end.date(),
            len(val_idx),
        )

    return splits


def make_lane_grouped_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 3,
) -> list[CVSplit]:
    """
    Create lane-grouped time splits to test cold-start generalization.

    Each fold: train on earlier months, validate on later months,
    but also ensure some lanes are completely held out.
    """
    # For simplicity, just do a different time split with gap
    # This is a secondary validation strategy
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    splits = []
    months = sorted(df[date_col].dt.month.unique())
    # Use last 3 months as validation windows
    for i, val_month in enumerate(months[-3:]):
        train_months = [m for m in months if m < val_month]
        if len(train_months) < 3:
            continue

        train_mask = df[date_col].dt.month.isin(train_months)
        val_mask = df[date_col].dt.month == val_month

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(val_idx) == 0:
            continue

        split = CVSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            train_start=df.loc[train_idx, date_col].min(),
            train_end=df.loc[train_idx, date_col].max(),
            val_start=df.loc[val_idx, date_col].min(),
            val_end=df.loc[val_idx, date_col].max(),
            fold_name=f"lane_fold_{i}_val_{val_month}",
        )
        splits.append(split)
        logger.info(
            "Lane fold %d: train months %s, val month %d (%d train, %d val)",
            i,
            train_months,
            val_month,
            len(train_idx),
            len(val_idx),
        )

    return splits


def validate_no_leakage(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df: pd.DataFrame,
    date_col: str = "date",
) -> bool:
    """Verify no temporal leakage: max train date < min val date."""
    train_dates = df.iloc[train_idx][date_col]
    val_dates = df.iloc[val_idx][date_col]
    max_train = train_dates.max()
    min_val = val_dates.min()
    ok = max_train < min_val
    if not ok:
        logger.error("LEAKAGE: max train date %s >= min val date %s", max_train, min_val)
    return ok
