# Freight Rate Prediction Challenge

**Machine Learning Engineer Assessment Solution**

## Overview

This repository contains a production-quality solution for the Freight Rate Prediction Challenge. The task is to predict freight load rates for 12,000 validation loads (Nov–Dec 2025) and a 31-day December chart for a fixed lane, using 48,000 historical loads (Jan–Oct 2025) as training data.

## Repository Structure

```
.
├── configs/                 # Configuration files
│   └── default.yaml         # Default pipeline configuration
├── data/                    # Input data (gitignored, not committed)
│   ├── train_test.csv       # 48,000 labeled loads (Jan–Oct 2025)
│   ├── validation.csv       # 12,000 loads to predict (Nov–Dec 2025)
│   ├── validation_predictions_template.csv  # Submission template
│   └── december_chart_inputs.csv            # 31-day fixed lane inputs
├── models/                  # Trained model artifacts (gitignored)
├── notebooks/               # Exploratory analysis
│   └── 01_eda.ipynb         # EDA notebook
├── reports/                 # Generated reports and charts
├── src/
│   └── freight_rate/        # Main package
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       ├── data.py          # Data loading & base features
│       ├── features.py      # Target encoding & advanced features
│       ├── validation.py    # Time-based CV splits
│       ├── models.py        # Model definitions (LGBM, XGB, Ridge, Ensemble)
│       ├── train.py         # Training pipeline
│       ├── predict.py       # Inference pipeline
│       └── evaluate.py      # Evaluation & visualization
├── tests/                   # Unit tests
│   └── test_pipeline.py
├── score.py                 # Official scorer (validates submissions)
├── requirements.txt         # Pinned dependencies
├── Makefile                 # Pipeline commands
└── README.md                # This file
```

## Quick Start

### Prerequisites
- Python 3.11+
- `make` (optional, for convenience)

### Setup

```bash
# Clone and enter repo
git clone <your-repo-url>
cd freight-rate-prediction

# Create virtual environment and install dependencies
make setup
# Or manually:
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Train models with cross-validation
make train

# Generate submission files
make predict

# Validate with official scorer (uses validation_predictions.csv at repo root)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

This produces:
- `validation_predictions.csv` — 12,000 predictions in required format (repo root)
- `data/december_chart_inputs.csv` — Updated with `predicted_rate` filled
- `scorer_results/candidate_december.png` — December chart for report

### Individual Commands

```bash
make train      # Train models, save to models/
make predict    # Load models, generate predictions
make score      # Run official score.py validation
make test       # Run pytest suite
make lint       # Run ruff linter
make clean      # Remove generated artifacts
```

## Methodology

### Data & Validation Strategy

- **Strict temporal split**: Training (Jan–Oct 2025) → Validation (Nov–Dec 2025). No random shuffling.
- **Expanding-window CV**: 4 folds — Jan–Jun|Jul, Jan–Jul|Aug, Jan–Aug|Sep, Jan–Sep|Oct. Mirrors production deployment.
- **No leakage**: All target encodings computed out-of-fold; day-level features (market_index) properly handled.

### Features

| Category | Features |
|----------|----------|
| **Distance** | `distance`, `haversine`, `road_ratio` (distance/haversine) |
| **Geography** | `pickup`, `delivery`, `lane`, `pickup_region`, `delivery_region`, lat/lon |
| **Equipment** | `equipment` (Dry Van/Reefer/Flatbed), interactions with distance |
| **Weight** | `weight`, `weight_missing` flag |
| **Temporal** | `month`, `week`, `dow`, `day_of_year`, cyclical sin/cos, month/quarter end flags |
| **Market** | `market_index`, `quote_signal` (with missing flags), daily aggregates |
| **Target Encodings (OOF)** | Smoothed means for pickup, delivery, lane, equipment, regions |
| **Frequency Encodings** | Category frequencies for same columns |
| **Lane/City Stats** | Historical mean/median rate & RPM per lane/city (train only) |

### Models

| Model | Role | CV MAPE | CV MAE | CV RMSE | R² |
|-------|------|---------|--------|---------|-----|
| LightGBM | Primary | **7.67%** | $172 | $647 | 0.816 |
| XGBoost | Secondary | 8.86% | $206 | $671 | 0.801 |
| Ridge | Baseline | 9.27% | $160 | $636 | 0.822 |
| Ensemble (inv MAPE) | Tested | 7.82% | $167 | $643 | 0.818 |

**Final model**: LightGBM only (weight 1.0) — beats ensemble on CV MAPE. Multi-seed averaging (5 seeds) for stability.

### Key Findings (from EDA)

1. **Distance dominates** (corr = 0.91 with rate). Rate ≈ distance × RPM + intercept.
2. **Market Index is day-level** — near-constant within day; daily mean MI + trend explains 59% of daily RPM variance.
3. **Strong seasonality**: RPM rises from 2.10 (Jan) to 2.33 (Jun), then declines to 2.24 (Oct).
4. **Equipment effect**: Reefer (2.38 $/mi) > Flatbed (2.29) > Dry Van (2.12).
5. **Distribution shift**: Validation MI mean 0.93 vs Train 1.08 (14% cooler market).
6. **Cold start**: 8 unseen cities, 12% unseen lanes in validation — handled via geo features & smoothed encodings.
7. **December chart**: Fixed lane Lexington→Fort Wayne, only date varies. Need to capture DOW pattern and Dec market level.
8. **Target has heavy tail** - MAPE-appropriate, log-transform helps.

## Submission Files

| File | Description |
|------|-------------|
| `validation_predictions.csv` | 12,000 rows: `load_id,predicted_rate` (exact template order) |
| `data/december_chart_inputs.csv` | 31 rows with `predicted_rate` filled (original 7 columns preserved) |
| `scorer_results/candidate_december.png` | December chart for report |

## Report

The report includes:
- Problem framing & data overview
- Validation strategy with temporal justification
- Feature engineering details
- Model comparison table (CV mean ± std)
- Final ensemble performance
- Error analysis (residuals by lane/month/rate bucket, SHAP importance)
- December chart with interpretation
- Limitations & next steps

## Loom Script (2–3 min)

| Time | Topic |
|------|-------|
| 0:00–0:30 | Problem & data: 48k train loads, 12k future validation, fixed December chart |
| 0:30–1:00 | Key EDA findings: distance dominance, day-level MI, seasonality, distribution shift |
| 1:00–1:20 | Validation: expanding-window CV, no leakage, OOF encodings |
| 1:20–1:50 | Features: geo, temporal, market, target encodings, lane/city stats |
| 1:50–2:10 | Models: LGBM/XGB/Ridge, Optuna tuning, ensemble weighting |
| 2:10–2:30 | December chart walkthrough, limitations (no Dec history), next steps |

## Reproducibility

```bash
# From clean clone:
git clone <repo>
cd freight-rate-prediction
make setup && make train && make predict && python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

All random seeds fixed (`random_state=42`). No absolute paths. Config-driven.

## Dependencies

See `requirements.txt` for pinned versions:
- pandas, numpy, scikit-learn
- lightgbm, xgboost, optuna
- matplotlib, shap
- pytest, ruff

## License

Assessment submission — proprietary.