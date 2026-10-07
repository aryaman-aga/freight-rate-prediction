# Loom Script — Freight Rate Prediction Challenge (2–3 minutes)

## 0:00–0:30 — Problem & Data
- **Task**: Predict rates for 12,000 loads (Nov–Dec 2025) + 31-day December chart for fixed lane Lexington→Fort Wayne (360 mi, Dry Van, 32k lbs)
- **Training**: 48,000 loads, Jan–Oct 2025
- **Key constraint**: Validation is strictly future — no random splits allowed

## 0:30–1:00 — Key EDA Findings
1. **Distance dominates** (corr 0.91): Rate ≈ distance × RPM + $270 intercept. Equipment RPM: Reefer 2.38 > Flatbed 2.29 > Dry Van 2.12 $/mi
2. **Market Index is day-level**: Near-constant within day; daily MI + trend explains 59% of daily RPM variance (R²=0.59). Row-level corr ~0.03
3. **Seasonality**: RPM 2.10 (Jan) → 2.33 (Jun) → 2.24 (Oct). No Nov/Dec history
4. **Distribution shift**: Val MI mean 0.93 vs Train 1.08 (14% cooler market). QS range truncated
5. **Cold start**: 8 new cities, 17.5% unseen lanes in validation — handled via geo features + smoothed encodings

## 1:00–1:20 — Validation Strategy
- **Expanding-window time CV**: Jan–Jun|Jul, Jan–Jul|Aug, Jan–Aug|Sep, Jan–Sep|Oct (4 folds)
- Mirrors production: retrain on all history, predict next month
- **No leakage**: All target encodings OOF (5-fold), smoothing=10, min_samples=5. Programmatically verified max(train_date) < min(val_date) per fold
- **Why not K-fold?**: Would leak day-level MI and temporal trends — fatal for this data

## 1:20–1:50 — Features & Models
- **Base**: distance, haversine, road_ratio, equipment, weight (fixed negative typos), cyclical temporal (month/dow), daily MI/QS, holiday-distance (days_to_christmas, days_to_new_year, days_to_thanksgiving)
- **Advanced (no leakage)**: OOF target encodings (pickup, delivery, lane, equipment, regions), frequency encodings, lane/city stats
- **Models**: LightGBM (primary), XGBoost, Ridge (numeric only)
- **CV Results**:
  - LightGBM: 6.06% MAPE ± 0.47
  - XGBoost: 6.37% MAPE ± 0.47
  - Ridge: 19.7% MAPE ± 0.9
  - Ensemble (inv-MAPE weighted): 6.86% MAPE ± 0.48 → **LightGBM alone wins**

## 1:50–2:10 — Final Selection & December Chart
- **Final**: LightGBM only (weight 1.0) — best CV MAPE, simplest, most robust
- **Target**: log(rate) — no floor needed, handles positive constraint naturally
- **December chart**: 31 predictions, $735–$859 (mean $795) vs historical $758–$974 (mean $857)
  - Captures DOW pattern (weekend dips Dec 7, 14, 21, 28)
  - Consistent with cooler market (MI ~0.93)
  - No Dec history — this is extrapolation

## 2:10–2:30 — Limitations & Next Steps
- **Limitations**: No Dec history (extrapolation), MI shift non-linearity, single-model risk, feature store not versioned, limited HPO, weight sign-fix assumption
- **Next steps**: Collect Nov/Dec data, Optuna tuning (30+ trials), fuel/weather/holiday features, feature store, quantile regression

---

## Key Code Walkthrough (if asked)
- `src/freight_rate/train.py` — single entry point, CV, ensemble eval, model selection, multi-seed retrain
- `src/freight_rate/features.py` — OOF target encodings, frequency encodings, lane/city stats
- `src/freight_rate/validation.py` — expanding-window CV, leakage check
- `src/freight_rate/predict.py` — inference pipeline, handles missing Dec coords via train lookup, multi-seed averaging
- `Makefile` — `make setup && make train && make predict && python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv` reproduces end-to-end