"""Evaluation and analysis utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_rate.models import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute and log all metrics."""
    metrics = compute_metrics(y_true, y_pred)
    logger.info("%sMetrics:", prefix)
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k.upper(), v)
    return metrics


def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate residual analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred
    abs_pct_error = np.abs(residuals / y_true) * 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 2. Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=5)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")

    # 3. Absolute % Error distribution
    ax = axes[0, 2]
    ax.hist(abs_pct_error, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(
        abs_pct_error.mean(), color="r", linestyle="--", label=f"Mean: {abs_pct_error.mean():.1f}%"
    )
    ax.axvline(
        np.median(abs_pct_error),
        color="g",
        linestyle="--",
        label=f"Median: {np.median(abs_pct_error):.1f}%",
    )
    ax.set_xlabel("Absolute % Error")
    ax.set_ylabel("Frequency")
    ax.set_title("APE Distribution")
    ax.legend()

    # 4. Residuals by month (if date available)
    ax = axes[1, 0]
    if "month" in X.columns:
        monthly_mape = X.groupby("month").apply(
            lambda g: np.mean(np.abs((y_true[g.index] - y_pred[g.index]) / y_true[g.index])) * 100
        )
        monthly_mape.plot(kind="bar", ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("MAPE (%)")
        ax.set_title("MAPE by Month")
    else:
        ax.text(0.5, 0.5, "No month column", ha="center", va="center")

    # 5. Residuals by equipment
    ax = axes[1, 1]
    if "equipment" in X.columns:
        equip_mape = X.groupby("equipment").apply(
            lambda g: np.mean(np.abs((y_true[g.index] - y_pred[g.index]) / y_true[g.index])) * 100
        )
        equip_mape.plot(kind="bar", ax=ax)
        ax.set_xlabel("Equipment")
        ax.set_ylabel("MAPE (%)")
        ax.set_title("MAPE by Equipment")
    else:
        ax.text(0.5, 0.5, "No equipment column", ha="center", va="center")

    # 6. Error by rate bucket
    ax = axes[1, 2]
    rate_buckets = pd.qcut(y_true, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    bucket_mape = pd.Series(abs_pct_error).groupby(rate_buckets).mean()
    bucket_mape.plot(kind="bar", ax=ax)
    ax.set_xlabel("Rate Quintile")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by Rate Bucket")

    plt.tight_layout()
    plt.savefig(output_dir / "residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved residual analysis to %s", output_dir / "residual_analysis.png")


def plot_feature_importance(
    model,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 30,
) -> None:
    """Plot feature importance for tree-based models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importance"):
        imp = model.feature_importance(importance_type="gain")
    elif hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
    else:
        logger.warning("Model does not have feature importance")
        return

    imp = imp.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(10, 8))
    imp.plot(kind="barh")
    plt.xlabel("Gain")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature importance to %s", output_dir / "feature_importance.png")


def plot_december_chart(
    dec_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create the December chart matching score.py style."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = dec_df.copy().sort_values("date")

    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    color = "#064A56"
    ax.plot(
        df["date"],
        df["predicted_rate"],
        color=color,
        linewidth=2.6,
        marker="o",
        markersize=3.2,
    )
    floor = float(df["predicted_rate"].min())
    ax.fill_between(
        df["date"],
        df["predicted_rate"],
        floor - max(10.0, floor * 0.02),
        color=color,
        alpha=0.08,
    )
    ax.set_title(
        "Candidate: December 2025 Predicted Load Rate",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Predicted rate ($)")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9DAFB3")
    ax.tick_params(axis="x", rotation=35)
    ax.text(
        0,
        -0.40,
        "Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#455A60",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_dir / "candidate_december.png", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved December chart to %s", output_dir / "candidate_december.png")


def compare_models_cv(
    model_names: list[str],
    cv_results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    """Create comparison table and plot for CV results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table
    rows = []
    for name in model_names:
        row = {"Model": name}
        for metric in ["mape", "mae", "rmse", "r2"]:
            row[f"{metric}_mean"] = cv_results[name][metric]["mean"]
            row[f"{metric}_std"] = cv_results[name][metric]["std"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "cv_comparison.csv", index=False)
    logger.info("\nCV Comparison:\n%s", df.to_string())

    # Plot
    metrics_to_plot = ["mape", "mae", "rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, metrics_to_plot, strict=False):
        means = [cv_results[name][metric]["mean"] for name in model_names]
        stds = [cv_results[name][metric]["std"] for name in model_names]
        x = np.arange(len(model_names))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"CV {metric.upper()} (mean ± std)")
    plt.tight_layout()
    plt.savefig(output_dir / "cv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved CV comparison to %s", output_dir / "cv_comparison.png")


def generate_report_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cv_results: dict,
    ensemble_weights: dict,
    output_dir: Path,
) -> dict:
    """Collect all data needed for the final report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "train_shape": train.shape,
        "val_shape": val.shape,
        "train_date_range": f"{train['date'].min().date()} to {train['date'].max().date()}",
        "val_date_range": f"{val['date'].min().date()} to {val['date'].max().date()}",
        "target_stats": train["posted_rate"].describe().to_dict(),
        "cv_results": cv_results,
        "ensemble_weights": ensemble_weights,
        "feature_count": len(
            [c for c in train.columns if not c.startswith("te_") and not c.startswith("freq_")]
        ),
    }

    import json

    with open(output_dir / "report_data.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    return report_data
