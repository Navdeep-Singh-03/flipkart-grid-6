# Flipkart Gridlock Hackathon 2.0 — Approach Document
**Team Submission | OOF R² = 0.9974 | Score = max(0, 100 × R²)**

---

## Problem Understanding

Predict `demand` (continuous, 0–1) for each (geohash, day, timestamp) combination.
- Train: 77,299 rows × 11 columns (days 48 & 49)
- Test:  41,778 rows × 10 columns (day 49 only)
- Metric: `score = max(0, 100 × R²(actual, predicted))`

---

## EDA Findings

| Finding | Impact |
|---|---|
| Test is entirely **day 49**; train has 69,427 rows of day 48 + 7,872 of day 49 | Day 48 → day 49 demand correlates at **r = 0.85** by geohash |
| `geohash × ts_mins` mean from day 48 explains **79% variance** in day 49 alone | Strongest single feature |
| Only 0.8% missing RoadType, 3.2% Temperature, 1.0% Weather | Minor imputation needed |
| 1,180 of 1,249 geohashes appear in both train and test | High coverage for target encoding |
| Timestamp is 15-min intervals (96 slots/day); day is an integer (48, 49) | Rich temporal structure |
| Demand is right-skewed (median=0.048, max=1.0) | No log-transform needed; tree models handle this |

---

## Feature Engineering

### Temporal Features
- `ts_mins`: timestamp parsed to minutes since midnight (0–1425)
- `hour`, `minute`: extracted from ts_mins
- `hour_sin/cos`, `ts_sin/cos`: cyclic encoding to capture periodicity
- `is_peak`: 1 if hour in {7,8,9,17,18,19,20}
- `is_night`: 1 if hour in {0,1,2,3,4,5}

### Geospatial Features
- `lat`, `lon`: decoded from 6-character geohash using pygeohash
- `geo5`, `geo4`, `geo3`: coarser geohash prefixes for hierarchical aggregation
- `lat_bin`, `lon_bin`: 20-bucket grid for spatial density aggregation

### Target Encoding (from Day 48 → applied to Day 49 / Test)
All encodings computed **exclusively from day 48** to prevent leakage on test:

| Feature | Group Keys | Aggregation |
|---|---|---|
| `geo_ts_mean/std` | geohash × ts_mins | mean, std |
| `geo_hr_mean/median` | geohash × hour | mean, median |
| `geo_mean/std/median/max` | geohash | mean, std, median, max |
| `ts_global_mean/median` | ts_mins | mean, median |
| `hr_global_mean/median` | hour | mean, median |
| `geo5_hr_mean` | geo5 × hour | mean |
| `geo4_ts_mean` | geo4 × ts_mins | mean |
| `geo3_hr_mean` | geo3 × hour | mean |
| `latlon_hr_mean` | lat_bin × lon_bin × hour | mean |
| `geo_ts_mean_filled` | Fill chain: geo_ts → geo5_hr → geo3_hr → geo_hr → geo_mean → ts_global | robust imputation |
| `demand_vs_ts` | geo_ts_mean_filled − ts_global_mean | location demand deviation |

**Total: 36 features**

---

## Model Selection

Three gradient boosting models were trained with 5-fold KFold CV:

| Model | OOF R² |
|---|---|
| LightGBM | 0.9967 |
| XGBoost | 0.9968 |
| CatBoost | **0.9973** |
| **Ensemble** | **0.9974** |

### Hyperparameters
All models: `n_estimators=3000`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `early_stopping_rounds=100`

---

## Validation Strategy

- **5-fold KFold** on full training data
- Target encodings built from **day 48 only** and applied to test (day 49) — mirrors the real train/test split
- Ensemble weights optimized by Nelder-Mead on OOF predictions: `LGB≈0.20, XGB≈0.09, CAT≈0.71`

---

## Why This Works

The dominant signal is **location × time demand pattern**. A geohash at peak hour on day 48 behaves nearly identically on day 49. The `geo_ts_mean` feature alone achieves R² ≈ 0.53 with no model; the GBDT models learn the residuals from road type, weather, temperature, and spatial proximity features.

---

## Folder Structure

```
submission/
├── solution.py            # Full runnable training script
├── requirements.txt       # Python dependencies
├── approach.md            # This document
└── outputs/
    └── submission.csv     # Final predictions (41778 × 2)
```

---

## Tools Used
- **pygeohash** — geohash decoding
- **LightGBM 4.3** — gradient boosting
- **XGBoost 2.0** — gradient boosting
- **CatBoost 1.2** — gradient boosting
- **scikit-learn** — KFold, r2_score
- **scipy** — Nelder-Mead ensemble weight optimization
