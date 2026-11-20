"""Phase 8 - run all simulation experiments and save tidy result tables."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config as C
from .data_loader import load_clean
from .simulation import (STRATEGIES, Scenario, hours_from_frame, make_arrivals,
                         simulate, stress_scenarios)

METRICS = ["avg_queue_veh", "avg_delay_s_per_veh", "remaining_queue", "max_queue_veh",
           "ambulance_avg_delay_s", "ambulance_max_delay_s", "emergency_preemptions", "ambulances"]


def prior_from_data(df) -> dict:
    h = df[df.sensor_faulty == 0]
    q = {d: h[c].sum() for d, c in C.QUEUE_COLS.items()}
    t = sum(q.values()); return {d: q[d] / t for d in q}


def load_pred_map() -> dict:
    p = pd.read_csv(C.RES_DIR / "predictions_test.csv", parse_dates=["time", "target_time"])
    best = json.loads((C.RES_DIR / "best_model.json").read_text())["selection"]["selected"]
    # prediction made at the end of hour t is the forecast for hour t+1 -> key by target_time
    return dict(zip(p["target_time"], p[best]))


def run_all(n_seeds: int = 10, strategies=None) -> dict:
    df = load_clean()
    prior = prior_from_data(df)
    pred_map = load_pred_map()
    t0 = min(pred_map)
    test = df[df.date_time >= t0].reset_index(drop=True)
    strategies = strategies or list(STRATEGIES)
    rows = []

    # --- A: controlled stress scenarios, many random replicates
    scns = stress_scenarios(test, pred_map, prior)
    for scn in scns:
        for seed in range(n_seeds):
            A = make_arrivals(scn, seed)
            rng = np.random.default_rng(1000 + seed)
            # randomise ambulance second per replicate
            for h in scn.hours:
                h.ambulances = [(int(rng.integers(120, 3400)), d) for _, d in h.ambulances]
            A = make_arrivals(scn, seed)
            for st in strategies:
                r = simulate(scn, st, seed, prior=prior, arrivals=A); r.pop("_hour_delay")
                rows.append({**r, "group": "stress"})
    stress = pd.DataFrame(rows)
    stress.to_csv(C.RES_DIR / "sim_stress_runs.csv", index=False)

    # --- B: full replay of every complete test day with the dataset's real event flags
    days, rows = [], []
    band_rows = []
    test["date"] = test.date_time.dt.date
    full = [g for _, g in test.groupby("date") if len(g) == 24]
    for gi, g in enumerate(full):
        rng = np.random.default_rng(gi)
        hs = hours_from_frame(g, pred_map, prior, rng, real_events=True)
        scn = Scenario(f"Replay {g.date.iloc[0]}", hs)
        A = make_arrivals(scn, gi)
        for st in strategies:
            r = simulate(scn, st, gi, prior=prior, arrivals=A)
            hd = r.pop("_hour_delay")
            rows.append({**r, "group": "replay", "date": str(g.date.iloc[0])})
            arr_h = A.reshape(24, 3600, 4).sum(axis=(1, 2))
            for hi, hv in enumerate(g.traffic_volume.values):
                band = "Low" if hv < C.LOW_MAX else ("High" if hv > C.HIGH_MIN else "Medium")
                band_rows.append(dict(strategy=st, band=band, delay=hd[hi], arrivals=arr_h[hi]))
    replay = pd.DataFrame(rows); replay.to_csv(C.RES_DIR / "sim_replay_days.csv", index=False)
    band = pd.DataFrame(band_rows).groupby(["strategy", "band"]).agg(delay=("delay", "sum"), arr=("arrivals", "sum"), hours=("arrivals", "size")).reset_index()
    band["avg_delay_s_per_veh"] = band.delay / band.arr
    band.to_csv(C.RES_DIR / "sim_replay_by_congestion.csv", index=False)

    # --- C: sensitivity of the headline result to the demand-scale assumption
    sens = []
    scn = scns[0]
    for sc in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        sig = dict(C.SIGNAL, demand_scale=sc)
        for seed in range(5):
            A = make_arrivals(scn, seed, sig)
            for st in ("Fixed-time", "Adaptive (full)"):
                r = simulate(scn, st, seed, prior=prior, sig=sig, arrivals=A); r.pop("_hour_delay")
                sens.append({"demand_scale": sc, **r})
    pd.DataFrame(sens).to_csv(C.RES_DIR / "sim_sensitivity.csv", index=False)

    def agg(df_):
        g = df_.groupby(["scenario", "strategy"] if "scenario" in df_ else ["strategy"])[METRICS].agg(["mean", "std"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]; return g.reset_index()
    stress_s = agg(stress); stress_s.to_csv(C.RES_DIR / "sim_stress_summary.csv", index=False)
    replay_s = agg(replay.assign(scenario="Replay of test days (real flags)"))
    replay_s.to_csv(C.RES_DIR / "sim_replay_summary.csv", index=False)
    return {"stress": stress_s, "replay": replay_s, "n_days": len(full), "prior": prior}


if __name__ == "__main__":
    run_all()


def calibrate_cycle_high(n_days: int = 30, candidates=(120, 135, 150), key="cycle_high_s", band="High") -> pd.DataFrame:
    """Choose the adaptive cycle for 'High' predicted demand on VALIDATION days
    (never on the test days used for the final report)."""
    df = load_clean(); prior = prior_from_data(df)
    pv = pd.read_csv(C.RES_DIR / "predictions_val.csv", parse_dates=["time", "target_time"])
    best = json.loads((C.RES_DIR / "best_model.json").read_text())["selection"]["selected"]
    pm = dict(zip(pv["target_time"], pv[best]))
    val = df[(df.date_time >= pv.target_time.min()) & (df.date_time <= pv.target_time.max())].copy()
    val["date"] = val.date_time.dt.date
    days = [g for _, g in val.groupby("date") if len(g) == 24][:n_days]
    rows = []
    for ch in candidates:
        sig = dict(C.SIGNAL, **{key: ch})
        for gi, g in enumerate(days):
            scn = Scenario("val", hours_from_frame(g, pm, prior, np.random.default_rng(gi), real_events=False))
            A = make_arrivals(scn, gi, sig)
            r = simulate(scn, "Adaptive (full)", gi, prior=prior, sig=sig, arrivals=A)
            hd = r.pop("_hour_delay"); arr = A.reshape(24, 3600, 4).sum(axis=(1, 2))
            hv = g.traffic_volume.values; hi = (hv > C.HIGH_MIN) if band == "High" else (hv < C.LOW_MAX)
            rows.append(dict(**{key: ch}, day=gi, delay_all=hd.sum() / arr.sum(),
                             delay_band=hd[hi].sum() / max(arr[hi].sum(), 1) if hi.any() else np.nan))
    out = pd.DataFrame(rows).groupby(key)[["delay_all", "delay_band"]].mean().reset_index()
    out["band"] = band
    out.to_csv(C.RES_DIR / f"calibration_{key}.csv", index=False)
    return out


if __name__ == "__main__":
    print(calibrate_cycle_high())
    print(calibrate_cycle_high(candidates=(60, 75, 90, 105, 120), key="cycle_low_s", band="Low"))
