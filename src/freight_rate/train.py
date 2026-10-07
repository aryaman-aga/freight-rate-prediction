"""Training pipeline entry point."""

from __future__ import annotations

import logging
import sys

import numpy as np

from freight_rate.config import CONFIG
from freight_rate.data import get_feature_columns, prepare_data
from freight_rate.features import engineer_all_features
from freight_rate.models import (
    LightGBMModel,
    RidgeModel,
    XGBoostModel,
    aggregate_cv_metrics,
    compute_metrics,
    train_model_cv,
)
from freight_rate.validation import make_time_splits, validate_no_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("FREIGHT RATE PREDICTION - TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load and prepare data
    logger.info("Loading data...")
    train, val, dec = prepare_data()

    # 2. Feature engineering
    logger.info("Engineering features...")
    train, val, dec, te_encodings, freq_encodings = engineer_all_features(train, val, dec)

    # 3. Prepare features and target (log scale)
    feature_cols = get_feature_columns(train)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    X = train[feature_cols]
    y = train["target"]  # log(rate)

    # 4. Time-based CV splits
    logger.info("Creating time-based CV splits...")
    splits = make_time_splits(train)
    for split in splits:
        assert validate_no_leakage(split.train_idx, split.val_idx, train), "Leakage detected!"

    # 5. Train models with CV
    logger.info("Training LightGBM with CV...")
    lgbm_models, lgbm_metrics = train_model_cv(LightGBMModel, X, y, splits)
    lgbm_agg = aggregate_cv_metrics(lgbm_metrics)
    logger.info("LightGBM CV: %s", lgbm_agg)

    logger.info("Training XGBoost with CV...")
    xgb_models, xgb_metrics = train_model_cv(XGBoostModel, X, y, splits)
    xgb_agg = aggregate_cv_metrics(xgb_metrics)
    logger.info("XGBoost CV: %s", xgb_agg)

    logger.info("Training Ridge baseline with CV...")
    # Ridge needs numeric features only - select numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    ridge_models, ridge_metrics = train_model_cv(
        RidgeModel, X_numeric, y, splits, params={"alpha": 10.0}
    )
    ridge_agg = aggregate_cv_metrics(ridge_metrics)
    logger.info("Ridge CV: %s", ridge_agg)

    # 6. Select best model and ensemble weights
    # Simple strategy: weight inversely proportional to MAPE
    mape_scores = {
        "lgbm": lgbm_agg["mape"]["mean"],
        "xgb": xgb_agg["mape"]["mean"],
        "ridge": ridge_agg["mape"]["mean"],
    }
    inv_mape = {k: 1.0 / v for k, v in mape_scores.items()}
    total = sum(inv_mape.values())
    weights = {k: v / total for k, v in inv_mape.items()}
    logger.info("Ensemble weights (inv MAPE): %s", weights)

    # 6b. Evaluate ensemble on CV folds
    logger.info("Evaluating ensemble on CV folds...")
    ensemble_fold_metrics = []
    for split in splits:
        X_train = X.iloc[split.train_idx]
        y_train = y.iloc[split.train_idx]
        X_val = X.iloc[split.val_idx]
        y_val = y.iloc[split.val_idx]
        X_val_numeric = X_numeric.iloc[split.val_idx]

        # Quick retrain on this fold for ensemble eval
        m_lgbm = LightGBMModel().fit(X_train, y_train, X_val, y_val)
        m_xgb = XGBoostModel().fit(X_train, y_train, X_val, y_val)
        m_ridge = RidgeModel(params={"alpha": 10.0}).fit(
            X_train.select_dtypes(include=[np.number]), y_train
        )

        # Predict on log scale, convert back with expm1
        p_lgbm = np.expm1(m_lgbm.predict(X_val))
        p_xgb = np.expm1(m_xgb.predict(X_val))
        p_ridge = np.expm1(m_ridge.predict(X_val_numeric))

        ensemble_pred = (
            weights["lgbm"] * p_lgbm + weights["xgb"] * p_xgb + weights["ridge"] * p_ridge
        )
        # y_val is on log scale, convert to dollar scale for metrics
        metrics = compute_metrics(np.expm1(y_val.values), ensemble_pred)
        ensemble_fold_metrics.append(metrics)
        logger.info(
            "Ensemble Fold %s: MAPE=%.2f%%, MAE=%.2f, RMSE=%.2f, R2=%.4f",
            split.fold_name,
            metrics["mape"],
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

    ensemble_agg = aggregate_cv_metrics(ensemble_fold_metrics)
    logger.info("Ensemble CV: %s", ensemble_agg)

    # Use best single model if it beats ensemble
    best_single = min(mape_scores, key=mape_scores.get)
    if mape_scores[best_single] < ensemble_agg["mape"]["mean"]:
        logger.info(
            "Best single model (%s MAPE=%.2f%%) beats ensemble (MAPE=%.2f%%), using single model",
            best_single,
            mape_scores[best_single],
            ensemble_agg["mape"]["mean"],
        )
        weights = {best_single: 1.0}
        for k in mape_scores:
            if k != best_single:
                weights[k] = 0.0
        logger.info("Adjusted weights: %s", weights)

    # 7. Retrain on ALL data with fixed params (multiple seeds, average predictions)
    logger.info("Retraining final LightGBM on all training data with multiple seeds...")
    seeds = [42, 123, 456, 789, 999]
    lgbm_models = []
    for seed in seeds:
        params = CONFIG.models.lgbm_params.copy()
        params["random_state"] = seed
        params["seed"] = seed
        m = LightGBMModel(params=params).fit(X, y)
        lgbm_models.append(m)

    final_xgb = XGBoostModel().fit(X, y)
    final_ridge = RidgeModel(params={"alpha": 10.0}).fit(X_numeric, y)

    # 8. Save models
    CONFIG.paths.model_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(lgbm_models):
        m.save(CONFIG.paths.model_dir / f"lgbm_final_seed{i}.txt")
    # Save the first one as the "main" model for compatibility
    lgbm_models[0].save(CONFIG.paths.model_dir / "lgbm_final.txt")
    final_xgb.save(CONFIG.paths.model_dir / "xgb_final.json")
    final_ridge.save(CONFIG.paths.model_dir / "ridge_final.pkl")

    # Save feature columns for inference
    import json

    with open(CONFIG.paths.model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    # Save ensemble weights
    with open(CONFIG.paths.model_dir / "ensemble_weights.json", "w") as f:
        json.dump(weights, f)

    # Save number of seeds for inference
    with open(CONFIG.paths.model_dir / "n_seeds.json", "w") as f:
        json.dump(len(seeds), f)

    # Save encodings
    import joblib

    joblib.dump(te_encodings, CONFIG.paths.model_dir / "target_encodings.joblib")
    joblib.dump(freq_encodings, CONFIG.paths.model_dir / "freq_encodings.joblib")

    # 9. Quick validation on validation set
    X_val = val[feature_cols]
    X_val_numeric = val[numeric_cols]

    # Average predictions across seeds (convert from log scale with expm1)
    lgbm_preds = np.column_stack([np.expm1(m.predict(X_val)) for m in lgbm_models])
    lgbm_pred = np.mean(lgbm_preds, axis=1)
    xgb_pred = np.expm1(final_xgb.predict(X_val))
    ridge_pred = np.expm1(final_ridge.predict(X_val_numeric))

    # Weighted ensemble
    ensemble_pred = (
        weights["lgbm"] * lgbm_pred + weights["xgb"] * xgb_pred + weights["ridge"] * ridge_pred
    )

    # Note: validation set has no ground truth, so we can't compute metrics here
    # Just log predictions summary
    logger.info("Validation predictions summary:")
    logger.info(
        "  LGBM (avg): mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        lgbm_pred.mean(),
        lgbm_pred.std(),
        lgbm_pred.min(),
        lgbm_pred.max(),
    )
    logger.info(
        "  XGB:    mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        xgb_pred.mean(),
        xgb_pred.std(),
        xgb_pred.min(),
        xgb_pred.max(),
    )
    logger.info(
        "  Ridge:  mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        ridge_pred.mean(),
        ridge_pred.std(),
        ridge_pred.min(),
        ridge_pred.max(),
    )
    logger.info(
        "  Ens:    mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        ensemble_pred.mean(),
        ensemble_pred.std(),
        ensemble_pred.min(),
        ensemble_pred.max(),
    )

    # Sanity check: Lexington>Fort Wayne December chart predictions
    X_dec = dec[feature_cols]
    X_dec_numeric = dec[numeric_cols]

    lgbm_dec_preds = np.column_stack([np.expm1(m.predict(X_dec)) for m in lgbm_models])
    lgbm_dec = np.mean(lgbm_dec_preds, axis=1)
    xgb_dec = np.expm1(final_xgb.predict(X_dec))
    ridge_dec = np.expm1(final_ridge.predict(X_dec_numeric))

    dec_ens = weights["lgbm"] * lgbm_dec + weights["xgb"] * xgb_dec + weights["ridge"] * ridge_dec
    logger.info("December chart predictions (ensemble):")
    for _i, (d, p) in enumerate(zip(dec["date"], dec_ens, strict=False)):
        logger.info("  %s: $%.2f", d.date(), p)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
