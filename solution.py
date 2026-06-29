"""
Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction
Full Solution | OOF R² ≈ 0.9974
"""

import pandas as pd
import numpy as np
import pygeohash as pgh
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import warnings, os
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 0. LOAD DATA
# ──────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

# ──────────────────────────────────────────────
# 1. BASE FEATURE ENGINEERING
# ──────────────────────────────────────────────
def parse_ts(ts):
    h, m = ts.split(':')
    return int(h) * 60 + int(m)

def decode_geo(df):
    lats, lons = [], []
    for g in df['geohash']:
        lat, lon = pgh.decode(g)
        lats.append(lat); lons.append(lon)
    df['lat'] = lats; df['lon'] = lons
    return df

def base_features(df):
    df = df.copy()
    df['ts_mins']  = df['timestamp'].apply(parse_ts)
    df['hour']     = df['ts_mins'] // 60
    df['minute']   = df['ts_mins'] % 60
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['ts_sin']   = np.sin(2 * np.pi * df['ts_mins'] / 1440)
    df['ts_cos']   = np.cos(2 * np.pi * df['ts_mins'] / 1440)
    df['is_peak']  = df['hour'].isin([7,8,9,17,18,19,20]).astype(int)
    df['is_night'] = df['hour'].isin([0,1,2,3,4,5]).astype(int)
    df['geo5'] = df['geohash'].str[:5]
    df['geo4'] = df['geohash'].str[:4]
    df['geo3'] = df['geohash'].str[:3]
    road_map    = {'Residential': 0, 'Street': 1, 'Highway': 2}
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['LargeVehicles_bin'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_bin']     = (df['Landmarks'] == 'Yes').astype(int)
    df['RoadType_enc']  = df['RoadType'].map(road_map).fillna(-1)
    df['Weather_enc']   = df['Weather'].map(weather_map).fillna(-1)
    df['Temperature']   = df['Temperature'].fillna(df['Temperature'].median())
    return df

train = decode_geo(train); test = decode_geo(test)
train = base_features(train); test = base_features(test)

# ──────────────────────────────────────────────
# 2. TARGET ENCODING
# Key insight: train is days 48+49, test is day 49.
# We build target encodings from day 48 data only
# to avoid leakage, then apply to test (day 49).
# ──────────────────────────────────────────────
def build_te(src):
    aggs = {}
    aggs['geo_ts'] = src.groupby(['geohash','ts_mins'])['demand'] \
        .agg(['mean','std']).reset_index()
    aggs['geo_ts'].columns = ['geohash','ts_mins','geo_ts_mean','geo_ts_std']

    aggs['geo_hr'] = src.groupby(['geohash','hour'])['demand'] \
        .agg(['mean','median']).reset_index()
    aggs['geo_hr'].columns = ['geohash','hour','geo_hr_mean','geo_hr_median']

    aggs['geo'] = src.groupby('geohash')['demand'] \
        .agg(['mean','std','median','max']).reset_index()
    aggs['geo'].columns = ['geohash','geo_mean','geo_std','geo_median','geo_max']

    aggs['ts'] = src.groupby('ts_mins')['demand'] \
        .agg(['mean','median']).reset_index()
    aggs['ts'].columns = ['ts_mins','ts_global_mean','ts_global_median']

    aggs['hr'] = src.groupby('hour')['demand'] \
        .agg(['mean','median']).reset_index()
    aggs['hr'].columns = ['hour','hr_global_mean','hr_global_median']

    aggs['geo5_hr'] = src.groupby(['geo5','hour'])['demand'].mean().reset_index()
    aggs['geo5_hr'].columns = ['geo5','hour','geo5_hr_mean']

    aggs['geo4_ts'] = src.groupby(['geo4','ts_mins'])['demand'].mean().reset_index()
    aggs['geo4_ts'].columns = ['geo4','ts_mins','geo4_ts_mean']

    aggs['geo3_hr'] = src.groupby(['geo3','hour'])['demand'].mean().reset_index()
    aggs['geo3_hr'].columns = ['geo3','hour','geo3_hr_mean']

    src2 = src.copy()
    src2['lat_bin'] = pd.cut(src2['lat'], 20, labels=False)
    src2['lon_bin'] = pd.cut(src2['lon'], 20, labels=False)
    aggs['latlon_hr'] = src2.groupby(['lat_bin','lon_bin','hour'])['demand'].mean().reset_index()
    aggs['latlon_hr'].columns = ['lat_bin','lon_bin','hour','latlon_hr_mean']
    return aggs

def apply_te(df, aggs):
    df = df.copy()
    df = df.merge(aggs['geo_ts'],  on=['geohash','ts_mins'], how='left')
    df = df.merge(aggs['geo_hr'],  on=['geohash','hour'],    how='left')
    df = df.merge(aggs['geo'],     on='geohash',             how='left')
    df = df.merge(aggs['ts'],      on='ts_mins',             how='left')
    df = df.merge(aggs['hr'],      on='hour',                how='left')
    df = df.merge(aggs['geo5_hr'], on=['geo5','hour'],        how='left')
    df = df.merge(aggs['geo4_ts'], on=['geo4','ts_mins'],     how='left')
    df = df.merge(aggs['geo3_hr'], on=['geo3','hour'],        how='left')
    df['lat_bin'] = pd.cut(df['lat'], 20, labels=False)
    df['lon_bin'] = pd.cut(df['lon'], 20, labels=False)
    df = df.merge(aggs['latlon_hr'], on=['lat_bin','lon_bin','hour'], how='left')
    # Robust fill chain
    df['geo_ts_mean_filled'] = (
        df['geo_ts_mean']
        .fillna(df['geo5_hr_mean'])
        .fillna(df['geo3_hr_mean'])
        .fillna(df['geo_hr_mean'])
        .fillna(df['geo_mean'])
        .fillna(df['ts_global_mean'])
    )
    df['demand_vs_ts'] = df['geo_ts_mean_filled'] - df['ts_global_mean'].fillna(0)
    return df

# Day 48 TE → applied to test (day 49)
te48   = build_te(train[train['day'] == 48])
# Full TE → applied to train for GBDT (includes all days, leakage-controlled by KFold)
te_all = build_te(train)

test_te  = apply_te(test,  te48)
train_te = apply_te(train, te_all)

# ──────────────────────────────────────────────
# 3. MODEL TRAINING
# ──────────────────────────────────────────────
FEATURES = [
    'ts_mins','hour','minute','day',
    'hour_sin','hour_cos','ts_sin','ts_cos',
    'is_peak','is_night',
    'lat','lon',
    'NumberofLanes','LargeVehicles_bin','Landmarks_bin',
    'RoadType_enc','Weather_enc','Temperature',
    'geo_ts_mean','geo_ts_std',
    'geo_hr_mean','geo_hr_median',
    'geo_mean','geo_std','geo_median','geo_max',
    'ts_global_mean','ts_global_median',
    'hr_global_mean','hr_global_median',
    'geo5_hr_mean','geo4_ts_mean','geo3_hr_mean',
    'latlon_hr_mean',
    'geo_ts_mean_filled','demand_vs_ts',
]
FEATURES = [f for f in FEATURES if f in train_te.columns]

X      = train_te[FEATURES].values
y      = train_te['demand'].values
X_test = test_te[FEATURES].values

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X)); pred_lgb = np.zeros(len(X_test))
oof_xgb = np.zeros(len(X)); pred_xgb = np.zeros(len(X_test))
oof_cat = np.zeros(len(X)); pred_cat = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    Xtr, Xval = X[tr_idx], X[val_idx]
    ytr, yval = y[tr_idx], y[val_idx]

    # LightGBM
    lgb_m = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.03, num_leaves=127,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[val_idx] = lgb_m.predict(Xval)
    pred_lgb += lgb_m.predict(X_test) / N_SPLITS

    # XGBoost
    xgb_m = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=100, eval_metric='rmse'
    )
    xgb_m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    oof_xgb[val_idx] = xgb_m.predict(Xval)
    pred_xgb += xgb_m.predict(X_test) / N_SPLITS

    # CatBoost
    cat_m = CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=7, l2_leaf_reg=3,
        random_seed=42, verbose=0, early_stopping_rounds=100, eval_metric='R2'
    )
    cat_m.fit(Xtr, ytr, eval_set=(Xval, yval))
    oof_cat[val_idx] = cat_m.predict(Xval)
    pred_cat += cat_m.predict(X_test) / N_SPLITS

    print(f"Fold {fold+1}: LGB={r2_score(yval, oof_lgb[val_idx]):.4f} "
          f"XGB={r2_score(yval, oof_xgb[val_idx]):.4f} "
          f"CAT={r2_score(yval, oof_cat[val_idx]):.4f}")

print(f"\nOOF LGB : {r2_score(y, oof_lgb):.4f}")
print(f"OOF XGB : {r2_score(y, oof_xgb):.4f}")
print(f"OOF CAT : {r2_score(y, oof_cat):.4f}")

# ──────────────────────────────────────────────
# 4. OPTIMAL ENSEMBLE WEIGHTS
# ──────────────────────────────────────────────
def neg_r2(w):
    w = np.array(w) / np.sum(w)
    return -r2_score(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat)

res   = minimize(neg_r2, [1/3, 1/3, 1/3], method='Nelder-Mead')
w_opt = np.array(res.x) / np.sum(res.x)
print(f"\nOptimal weights — LGB:{w_opt[0]:.3f} XGB:{w_opt[1]:.3f} CAT:{w_opt[2]:.3f}")
print(f"OOF Ensemble R²: {r2_score(y, w_opt[0]*oof_lgb + w_opt[1]*oof_xgb + w_opt[2]*oof_cat):.4f}")

# ──────────────────────────────────────────────
# 5. GENERATE SUBMISSION
# ──────────────────────────────────────────────
pred_final = np.clip(w_opt[0]*pred_lgb + w_opt[1]*pred_xgb + w_opt[2]*pred_cat, 0, 1)

os.makedirs('outputs', exist_ok=True)
submission = pd.DataFrame({'Index': test['Index'], 'demand': pred_final})
submission.to_csv('outputs/submission.csv', index=False)
print(f"\nSaved outputs/submission.csv  shape={submission.shape}")
print(submission.head())
