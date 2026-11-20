"""Phase 2 - safe data loading and data-quality reporting."""
from __future__ import annotations
import pandas as pd
from . import config as C

REQUIRED_CLEAN = ["date_time", "temp", "weather_main", "rain_1h", "snow_1h",
                  "clouds_all", "holiday", "traffic_volume", "queue_north",
                  "queue_south", "queue_east", "queue_west",
                  "ambulance_detected", "ambulance_approach",
                  "incident_reported", "sensor_status", "congestion_level",
                  "congestion_code", "total_queue"]


def _read(path) -> pd.DataFrame:
    try:
        # keep_default_na=False: the literal string "None" in
        # ambulance_approach is a *category*, not a missing value.
        df = pd.read_csv(path, keep_default_na=False, na_values=[""],
                         parse_dates=["date_time"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found: {path}. Put the CSV in "
                                f"the data/ folder.") from exc
    return df


def load_clean() -> pd.DataFrame:
    df = _read(C.CLEAN_CSV)
    missing = [c for c in REQUIRED_CLEAN if c not in df.columns]
    if missing:
        raise ValueError(f"traffic_cleaned.csv is missing columns: {missing}")
    df = df.sort_values("date_time").reset_index(drop=True)
    df["is_holiday"] = (df["holiday"].astype(str) == C.HOLIDAY_LABEL).astype(int)
    df["sensor_faulty"] = (df["sensor_status"] == "faulty").astype(int)
    df["ambulance_approach"] = df["ambulance_approach"].fillna("None")
    return df


def load_raw() -> pd.DataFrame:
    return _read(C.RAW_CSV).sort_values("date_time").reset_index(drop=True)


def quality_report(clean: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """Facts about the data that drive modelling decisions."""
    q = list(C.QUEUE_COLS.values())
    gaps = clean["date_time"].diff().dropna()
    faulty = clean["sensor_faulty"] == 1
    raw_missing_rows = raw[q].isna().all(axis=1)
    healthy = ~raw[q].isna().any(axis=1)
    band = clean.groupby("congestion_level")["traffic_volume"].agg(["min", "max"])
    return {
        "rows": int(len(clean)), "columns": int(clean.shape[1]),
        "period": [str(clean.date_time.min()), str(clean.date_time.max())],
        "all_gaps_are_1h": bool((gaps == pd.Timedelta(hours=1)).all()),
        "duplicate_timestamps": int(clean.date_time.duplicated().sum()),
        "raw_queue_missing_rows": int(raw_missing_rows.sum()),
        "raw_missing_rows_equal_faulty_rows": bool(
            (raw_missing_rows == (raw.sensor_status == "faulty")).all()),
        "clean_equals_raw_on_healthy_rows": bool(
            (raw.loc[healthy, q].values == clean.loc[healthy, q].values).all()),
        "faulty_rows": int(faulty.sum()),
        "faulty_share_pct": round(100 * faulty.mean(), 2),
        "volume_identical_raw_vs_clean": bool(
            (raw.traffic_volume == clean.traffic_volume).all()),
        "congestion_band_ranges": {k: [int(v["min"]), int(v["max"])]
                                   for k, v in band.iterrows()},
        "weather_description_duplicates_weather_main": bool(
            clean.groupby("weather_main")["weather_description"].nunique().max() == 1),
        "ambulance_events": int(clean.ambulance_detected.sum()),
        "incident_events": int(clean.incident_reported.sum()),
        "holiday_hours": int(clean.is_holiday.sum()),
    }
