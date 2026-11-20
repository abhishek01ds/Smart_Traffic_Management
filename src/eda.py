"""Phase 2 - exploratory data analysis: figures + a JSON summary."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from . import config as C
from .data_loader import load_clean, load_raw, quality_report

sns.set_theme(style="whitegrid")
Q = list(C.QUEUE_COLS.values())


def _save(name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / name, dpi=130); plt.close()


def run_eda() -> dict:
    df, raw = load_clean(), load_raw()
    rep = quality_report(df, raw)
    df["hour"] = df.date_time.dt.hour
    df["dow"] = df.date_time.dt.dayofweek
    df["month"] = df.date_time.dt.month

    # 1 volume distribution + hourly profile
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df.traffic_volume, bins=50, ax=ax[0])
    for t in (C.LOW_MAX, C.HIGH_MIN):
        ax[0].axvline(t, color="r", ls="--")
    ax[0].set_title("Traffic volume distribution (dashed = congestion bands)")
    sns.lineplot(data=df, x="hour", y="traffic_volume", hue=df.dow >= 5, ax=ax[1])
    ax[1].set_title("Hourly profile: weekday vs weekend (True)")
    _save("eda_volume.png")

    # 2 heatmap dow x hour
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.pivot_table(index="dow", columns="hour", values="traffic_volume"),
                cmap="YlOrRd")
    plt.title("Mean volume: day-of-week x hour"); _save("eda_heatmap.png")

    # 3 directional queues
    plt.figure(figsize=(7, 4))
    df[Q].mean().rename(lambda s: s.replace("queue_", "").title()).plot.bar(color="steelblue")
    plt.ylabel("Mean queue (vehicles)"); plt.title("Directional congestion")
    _save("eda_queues.png")

    # 4 correlation
    plt.figure(figsize=(7, 5))
    cols = ["traffic_volume"] + Q + ["temp", "rain_1h", "snow_1h", "clouds_all"]
    sns.heatmap(df.loc[df.sensor_faulty == 0, cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation (healthy-sensor rows)"); _save("eda_corr.png")

    # 5 faulty sensor investigation
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x="sensor_status", y="total_queue", ax=ax[0])
    ax[0].set_title("total_queue: faulty rows are imputed, not measured")
    r = df.loc[df.sensor_faulty == 0, Q + ["traffic_volume"]].corr()["traffic_volume"][Q]
    rf = df.loc[df.sensor_faulty == 1, Q + ["traffic_volume"]].corr()["traffic_volume"][Q]
    pd.DataFrame({"healthy": r, "faulty": rf}).rename(
        lambda s: s.replace("queue_", "").title()).plot.bar(ax=ax[1])
    ax[1].set_title("corr(queue, volume): healthy vs faulty rows"); ax[1].set_ylim(0, 1)
    _save("eda_faulty_sensor.png")

    # 6 weather vs volume
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="weather_main", y="traffic_volume", errorbar=None)
    plt.xticks(rotation=30); plt.title("Mean volume by weather"); _save("eda_weather.png")

    # 7 leakage evidence
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="congestion_level", y="traffic_volume",
                order=["Low", "Medium", "High"], ax=ax)
    ax.set_title("congestion_level is a threshold of traffic_volume -> LEAKAGE")
    _save("eda_leakage.png")

    # statistics
    inc = stats.mannwhitneyu(df.loc[df.incident_reported == 1, "traffic_volume"],
                             df.loc[df.incident_reported == 0, "traffic_volume"])
    amb = stats.mannwhitneyu(df.loc[df.ambulance_detected == 1, "traffic_volume"],
                             df.loc[df.ambulance_detected == 0, "traffic_volume"])
    rep["stats"] = {
        "corr_queue_vs_volume_healthy": {k: round(float(v), 3) for k, v in r.items()},
        "corr_queue_vs_volume_faulty": {k: round(float(v), 3) for k, v in rf.items()},
        "autocorr_lag1": round(float(df.traffic_volume.autocorr(1)), 3),
        "autocorr_lag24": round(float(df.traffic_volume.autocorr(24)), 3),
        "autocorr_lag168": round(float(df.traffic_volume.autocorr(168)), 3),
        "mean_volume_weekday": round(float(df.loc[df.dow < 5, "traffic_volume"].mean()), 1),
        "mean_volume_weekend": round(float(df.loc[df.dow >= 5, "traffic_volume"].mean()), 1),
        "mean_volume_incident_vs_none": [round(float(df.loc[df.incident_reported == 1, "traffic_volume"].mean()), 1),
                                          round(float(df.loc[df.incident_reported == 0, "traffic_volume"].mean()), 1)],
        "incident_mannwhitney_p": round(float(inc.pvalue), 4),
        "ambulance_mannwhitney_p": round(float(amb.pvalue), 4),
        "mean_queue_by_direction": {k: round(float(df[v].mean()), 2) for k, v in C.QUEUE_COLS.items()},
        "queue_share_by_direction_healthy": {k: round(float(df.loc[df.sensor_faulty == 0, v].sum() / df.loc[df.sensor_faulty == 0, Q].sum().sum()), 3)
                                              for k, v in C.QUEUE_COLS.items()},
        "congestion_class_share_pct": {k: round(float(v), 1) for k, v in
                                       (100 * df.congestion_level.value_counts(normalize=True)).items()},
        "ambulance_by_direction": df.loc[df.ambulance_detected == 1, "ambulance_approach"].value_counts().to_dict(),
        "peak_hour_mean_volume": {"hour": int(df.groupby("hour").traffic_volume.mean().idxmax()),
                                   "volume": round(float(df.groupby("hour").traffic_volume.mean().max()), 0)},
    }
    C.RES_DIR.mkdir(parents=True, exist_ok=True)
    (C.RES_DIR / "eda_summary.json").write_text(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2))
