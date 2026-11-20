"""Phase 3 - feature engineering and next-hour target construction.

Prediction time is the END of hour t.  Everything used as an input must be
known by then:  volume up to t, weather at t, queues at t (masked if the
sensor was faulty) and CALENDAR facts about the target hour t+1 (calendar is
known in advance).  Target:  traffic_volume at t+1.

Columns deliberately NOT used (see docs/EDA): congestion_level, congestion_code
(pure thresholds of traffic_volume -> leakage), total_queue (replaced by a
masked version), weather_description (duplicate of weather_main),
ambulance_*/incident_reported (event flags with no relation to volume).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

QCOLS = list(C.QUEUE_COLS.values())
LAGS = [1, 2, 3, 6, 12, 24, 48]
# volume of the TARGET hour one day / one week ago = lag 23 / lag 167 from t
TARGET_ANALOGUE_LAGS = [23, 167]
MAX_LAG = 167


def mask_faulty_queues(df: pd.DataFrame) -> pd.DataFrame:
    """Faulty-sensor queue values are untrustworthy (blank in the raw file,
    imputed in the cleaned file).  Set them to NaN, then fill CAUSALLY with
    the last healthy reading (forward fill, max 3 h); anything left is filled
    with the training-independent global median of healthy readings."""
    out = df.copy()
    healthy = out["sensor_faulty"] == 0
    for c in QCOLS:
        s = out[c].where(healthy)
        s = s.ffill(limit=3)
        out[c + "_h"] = s.fillna(out.loc[healthy, c].median())
    out["queue_total_h"] = out[[c + "_h" for c in QCOLS]].sum(axis=1)
    return out


def build_dataset(df: pd.DataFrame):
    """Return X (features), y (next-hour volume) and meta (timestamps etc.)."""
    d = mask_faulty_queues(df)
    v = d["traffic_volume"].astype(float)
    f = pd.DataFrame(index=d.index)

    f["volume_t"] = v
    for k in LAGS:
        f[f"volume_lag{k}"] = v.shift(k)
    for k in TARGET_ANALOGUE_LAGS:
        f[f"volume_lag{k}"] = v.shift(k)
    for w in (3, 6, 24):
        f[f"vol_roll_mean{w}"] = v.rolling(w).mean()
    for w in (6, 24):
        f[f"vol_roll_std{w}"] = v.rolling(w).std()
    f["vol_diff1"] = v.diff(1)

    # calendar of the TARGET hour (t+1)
    tgt_time = d["date_time"] + pd.Timedelta(hours=1)
    hour = tgt_time.dt.hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["target_hour"] = hour
    f["target_dow"] = tgt_time.dt.dayofweek
    f["target_is_weekend"] = (tgt_time.dt.dayofweek >= 5).astype(int)
    f["target_month"] = tgt_time.dt.month
    f["is_holiday"] = d["is_holiday"]

    # weather at t
    for c in ("temp", "rain_1h", "snow_1h", "clouds_all"):
        f[c] = d[c]
    for cat in C.WEATHER_CATEGORIES:
        f[f"weather_{cat}"] = (d["weather_main"] == cat).astype(int)

    # queues at t (masked) + sensor flag
    for c in QCOLS:
        f[c + "_h"] = d[c + "_h"]
    f["queue_total_h"] = d["queue_total_h"]
    f["queue_total_roll3"] = d["queue_total_h"].rolling(3).mean()
    f["sensor_faulty"] = d["sensor_faulty"]

    y = v.shift(-1)
    meta = pd.DataFrame({"time": d["date_time"], "target_time": tgt_time,
                         "volume_t": v})
    keep = (~y.isna()) & (~f.isna().any(axis=1))
    return f[keep], y[keep].rename("y_next_hour"), meta[keep]


def chronological_split(X, y, meta):
    """70/15/15 split in time order with a small gap at each boundary."""
    n = len(X)
    i_tr = int(n * C.TRAIN_FRAC)
    i_va = int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
    g = C.SPLIT_GAP_HOURS
    sl = {"train": slice(0, i_tr), "val": slice(i_tr + g, i_va),
          "test": slice(i_va + g, n)}
    out = {k: (X.iloc[s], y.iloc[s], meta.iloc[s]) for k, s in sl.items()}
    return out
