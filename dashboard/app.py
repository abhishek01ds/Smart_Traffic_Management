"""Phase 9 - Streamlit dashboard:  streamlit run dashboard/app.py

Five pages: Traffic Overview | ML Model Comparison | Intelligent Decision Engine |
Adaptive Signal Optimization | Performance Evaluation.
All heavy results are read from outputs/ (created by run_pipeline.py).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.data_loader import load_clean
from src.decision_engine import Observation, decide, observation_from_row, DEFAULT_PRIOR, level_of
from src.experiments import prior_from_data
from src.explain import local_contributions
from src.features import build_dataset
from src.signal_optimizer import fixed_time_plan, PHASE_ORDER

st.set_page_config(page_title="Smart Traffic Decision System", layout="wide", page_icon="🚦")
DIRS = C.DIRECTIONS
LEVEL_COLOR = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


# ------------------------------------------------------------------ loaders
@st.cache_data
def get_data():
    df = load_clean()
    X, y, meta = build_dataset(df)
    return df, X, y, meta


@st.cache_data
def get_results():
    R = C.RES_DIR
    rd = lambda f, **k: pd.read_csv(R / f, **k) if (R / f).exists() else None
    return dict(
        metrics=rd("model_metrics.csv"), base=rd("baseline_metrics.csv"),
        pred=rd("predictions_test.csv", parse_dates=["time", "target_time"]),
        best=json.loads((R / "best_model.json").read_text()) if (R / "best_model.json").exists() else None,
        pair=rd("pairwise_bootstrap.csv"), seg=rd("error_by_congestion_band.csv"),
        stress=rd("sim_stress_summary.csv"), replay=rd("sim_replay_summary.csv"),
        replay_days=rd("sim_replay_days.csv"), band=rd("sim_replay_by_congestion.csv"),
        sens=rd("sim_sensitivity.csv"), eda=json.loads((R / "eda_summary.json").read_text()) if (R / "eda_summary.json").exists() else None,
    )


@st.cache_resource
def get_model():
    return joblib.load(C.MODEL_DIR / "best_model.joblib")


def fig_path(name):
    p = C.FIG_DIR / name
    return str(p) if p.exists() else None


def show_fig(name, caption=None):
    p = fig_path(name)
    if p: st.image(p, caption=caption, width="stretch")


# --------------------------------------------------------- intersection plot
def draw_intersection(queues: dict, green: dict, order, t_in_cycle: float, clearance: int,
                      emergency: str | None = None, title=""):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.axis("off")
    ax.add_patch(Rectangle((-10, -1.6), 20, 3.2, color="#444")); ax.add_patch(Rectangle((-1.6, -10), 3.2, 20, color="#444"))
    ax.add_patch(Rectangle((-1.6, -1.6), 3.2, 3.2, color="#666"))
    # which approach is green now
    t = t_in_cycle; active, state = None, "red"
    for d in order:
        g = green[d]
        if t < g: active, state = d, "green"; break
        t -= g
        if t < clearance: active, state = d, "yellow"; break
        t -= clearance
    pos = {"North": (0.8, 2.6, 0, 1), "South": (-0.8, -2.6, 0, -1), "East": (2.6, -0.8, 1, 0), "West": (-2.6, 0.8, -1, 0)}
    for d in DIRS:
        x0, y0, dx, dy = pos[d]
        n = int(min(round(queues.get(d, 0)), 14))
        for i in range(n):
            ax.add_patch(Rectangle((x0 + dx * (0.8 + 1.15 * i) - 0.4, y0 + dy * (0.8 + 1.15 * i) - 0.4), 0.8, 0.8,
                                   color="#d32f2f" if d == emergency else "#1976d2"))
        col = {"green": "#2e7d32", "yellow": "#fbc02d", "red": "#c62828"}[state if d == active else "red"]
        lx, ly = x0 * 1.25 + dx * 0.4, y0 * 1.25 + dy * 0.4
        ax.add_patch(Circle((lx, ly), 0.55, color=col))
        lab = (f"{d}\nq={queues.get(d, 0):.0f}  g={green[d]}s" + (" 🚑" if d == emergency else ""))
        ax.text(x0 * 3.4 + dx * 1.2 if dx else x0 * 3.4, y0 * 3.4 + dy * 1.2 if dy else y0 * 3.4, lab, ha="center", va="center", fontsize=9,
                fontweight="bold")
    ax.set_title(title)
    return fig


def plan_chart(plans: dict, order_map: dict):
    fig, ax = plt.subplots(figsize=(8, 2.0 + 0.5 * len(plans)))
    cols = {"North": "#1565c0", "East": "#6a1b9a", "South": "#00838f", "West": "#ef6c00"}
    for yi, (name, green) in enumerate(plans.items()):
        left = 0
        for d in order_map[name]:
            ax.barh(yi, green[d], left=left, color=cols[d], edgecolor="white")
            ax.text(left + green[d] / 2, yi, f"{d[0]} {green[d]}s", ha="center", va="center", color="white", fontsize=8)
            left += green[d] + C.SIGNAL["clearance_s"]
    ax.set_yticks(range(len(plans))); ax.set_yticklabels(list(plans)); ax.set_xlabel("seconds in cycle (gaps = clearance)")
    plt.tight_layout(); return fig


# ----------------------------------------------------------------- page 1
def page_overview(df, res):
    st.header("Dashboard 1 - Traffic Overview")
    c = st.columns(4)
    dmin, dmax = df.date_time.min().date(), df.date_time.max().date()
    rng = st.slider("Date range", dmin, dmax, (dmin, dmax))
    d = df[(df.date_time.dt.date >= rng[0]) & (df.date_time.dt.date <= rng[1])]
    c[0].metric("Total traffic volume", f"{d.traffic_volume.sum():,.0f}")
    c[1].metric("Mean hourly volume", f"{d.traffic_volume.mean():,.0f}")
    c[2].metric("High-congestion hours", f"{100 * (d.congestion_level == 'High').mean():.1f}%")
    c[3].metric("Faulty-sensor hours", f"{int(d.sensor_faulty.sum())}")

    st.subheader("Current condition (pick a moment)")
    ts = st.select_slider("Timestamp", options=list(d.date_time.dt.strftime("%Y-%m-%d %H:%M")),
                          value=d.date_time.dt.strftime("%Y-%m-%d %H:%M").iloc[min(len(d) - 1, 17)])
    row = d[d.date_time == pd.Timestamp(ts)].iloc[0]
    lvl = level_of(row.traffic_volume)
    a, b = st.columns([1, 2])
    a.markdown(f"### Congestion: <span style='color:{LEVEL_COLOR[lvl]}'>{lvl}</span>", unsafe_allow_html=True)
    a.write(f"Volume **{row.traffic_volume:,}** veh/h | weather **{row.weather_main}** | sensor **{row.sensor_status}**")
    if row.sensor_status == "faulty":
        a.warning("Queue values for this hour are imputed (sensor faulty).")
    qdf = pd.DataFrame({"Direction": DIRS, "Queue": [row[C.QUEUE_COLS[x]] for x in DIRS]}).set_index("Direction")
    b.bar_chart(qdf)
    hi = qdf["Queue"].idxmax(); a.info(f"Highest queue now: **{hi}** ({int(qdf.loc[hi, 'Queue'])} veh)")

    st.subheader("Direction-wise congestion over the selected period")
    mq = pd.DataFrame({x: [d[C.QUEUE_COLS[x]].mean()] for x in DIRS}).T.rename(columns={0: "Mean queue"})
    c1, c2 = st.columns(2); c1.bar_chart(mq)
    c2.write("Long-run share of queued vehicles:")
    c2.dataframe(pd.DataFrame(res["eda"]["stats"]["queue_share_by_direction_healthy"], index=["share"]).T.style.format("{:.1%}"))
    st.subheader("Traffic trends")
    daily = d.set_index("date_time").traffic_volume.resample("D").mean()
    st.line_chart(daily, height=220)
    c3, c4 = st.columns(2)
    c3.line_chart(d.groupby(d.date_time.dt.hour).traffic_volume.mean(), height=220)
    c3.caption("Mean volume by hour of day")
    show_fig("eda_heatmap.png")


# ----------------------------------------------------------------- page 2
def page_models(res):
    st.header("Dashboard 2 - ML Model Comparison")
    m, best = res["metrics"], res["best"]["selection"]
    st.success(f"Selected model: **{best['selected']}** (lowest validation RMSE; confirmed on the unseen test set: "
               f"{best['test_confirms_selection']}).")
    show = m[["model", "test_MAE", "test_RMSE", "test_R2", "val_RMSE", "train_time_s", "predict_ms_per_1000"]].copy()
    st.dataframe(show.style.format({"test_MAE": "{:.1f}", "test_RMSE": "{:.1f}", "test_R2": "{:.4f}", "val_RMSE": "{:.1f}",
                                    "train_time_s": "{:.2f}", "predict_ms_per_1000": "{:.1f}"}), width="stretch")
    show_fig("model_metrics_bar.png")
    st.subheader("Actual vs predicted (test period)")
    p = res["pred"]
    d0, d1 = p.target_time.min().date(), p.target_time.max().date()
    r = st.slider("Window", d0, d1, (d0, d0 + pd.Timedelta(days=6)))
    sel = st.multiselect("Models", list(C.SIMPLICITY_RANK), default=list(C.SIMPLICITY_RANK))
    w = p[(p.target_time.dt.date >= r[0]) & (p.target_time.dt.date <= r[1])].set_index("target_time")
    st.line_chart(w[["actual"] + sel], height=320)
    c1, c2 = st.columns(2)
    with c1: show_fig("pred_scatter.png")
    with c2: show_fig("residuals.png")
    st.subheader("Why this model?")
    mm = m.set_index("model"); b = best["selected"]
    st.markdown(f"""
* **{b}** has test RMSE **{mm.loc[b, 'test_RMSE']:.0f}** veh/h and MAE **{mm.loc[b, 'test_MAE']:.0f}** veh/h
  (about {mm.loc[b, 'test_MAE_pct_of_mean_volume']:.1f}% of mean volume), R² = {mm.loc[b, 'test_R2']:.3f}.
* Paired day-block bootstrap (95% CI of RMSE difference) shows whether the gap is real, see table below.
* Trade-off: Linear Regression is fastest and most interpretable but misses non-linear hour/lag interactions;
  Random Forest and Gradient Boosting capture them. Final choice follows the pre-declared rule in `config.py`.
""")
    st.dataframe(res["pair"], width="stretch")
    st.caption("Reference baselines (not ML models):")
    st.dataframe(res["base"], width="stretch")
    st.subheader("Explainability")
    c1, c2 = st.columns(2)
    with c1: show_fig("permutation_importance_best.png")
    with c2: show_fig("shap_summary_best.png")
    show_fig("permutation_importance_all.png")


# ----------------------------------------------------------------- page 3
def page_decisions(df, X, y, meta, res):
    st.header("Dashboard 3 - Intelligent Decision Engine")
    model = get_model()
    prior = prior_from_data(df)
    test_times = list(res["pred"].time)
    st.sidebar.markdown("### Decision-engine inputs")
    mode = st.sidebar.radio("Input source", ["Replay a test-period hour", "What-if (manual)"])
    # -- pick an hour
    amb_hours = df[(df.date_time >= test_times[0]) & (df.ambulance_detected == 1)].date_time
    quick = st.selectbox("Jump to an interesting hour", ["(choose below)"] + [f"Ambulance: {t}" for t in amb_hours.dt.strftime("%Y-%m-%d %H:%M")[:30]]
                         + [f"Faulty sensor: {t}" for t in df[(df.date_time >= test_times[0]) & (df.sensor_faulty == 1)].date_time.dt.strftime("%Y-%m-%d %H:%M")[:15]]
                         + [f"Incident: {t}" for t in df[(df.date_time >= test_times[0]) & (df.incident_reported == 1)].date_time.dt.strftime("%Y-%m-%d %H:%M")[:15]])
    default = pd.Timestamp(quick.split(": ", 1)[1]) if ": " in quick else pd.Timestamp(test_times[40])
    ts = st.select_slider("Hour", options=test_times, value=min(test_times, key=lambda t: abs(t - default)),
                          format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"))
    row = df[df.date_time == ts].iloc[0]
    xi = X.loc[meta.index[meta.time == ts]]
    pred = float(model.predict(xi)[0])
    obs = observation_from_row(row, pred)
    if mode.startswith("What"):
        c = st.columns(4)
        for i, dname in enumerate(DIRS):
            obs.queues[dname] = c[i].number_input(f"Queue {dname}", 0, 60, int(obs.queues[dname]))
        c = st.columns(5)
        obs.ambulance_detected = c[0].checkbox("Ambulance", obs.ambulance_detected)
        obs.ambulance_approach = c[1].selectbox("From", ["None"] + DIRS, index=(["None"] + DIRS).index(obs.ambulance_approach) if obs.ambulance_approach in DIRS + ["None"] else 0)
        obs.incident_reported = c[2].checkbox("Incident", obs.incident_reported)
        obs.sensor_faulty = c[3].checkbox("Sensor faulty", obs.sensor_faulty)
        obs.weather_main = c[4].selectbox("Weather", C.WEATHER_CATEGORIES, index=C.WEATHER_CATEGORIES.index(obs.weather_main))
        obs.predicted_volume = st.slider("Predicted next-hour volume", 200, 6500, int(pred))
    contrib = local_contributions(model, xi, X.iloc[:2000], top=6)
    obs.top_features = [(n, v) for n, v, _ in contrib]
    dec = decide(obs, prior=prior)

    a, b, c = st.columns(3)
    a.metric("Ambulance", "DETECTED " + obs.ambulance_approach if obs.ambulance_detected else "none")
    b.metric("Incident", "REPORTED" if obs.incident_reported else "none")
    c.metric("Sensor health", "FAULTY" if obs.sensor_faulty else "healthy")
    for al in dec.alerts: st.error(al) if ("EMERGENCY" in al or "Incident" in al) else st.warning(al)
    st.subheader("Decision explanation")
    st.code(dec.explanation())
    st.subheader("Triggered rules")
    st.dataframe(pd.DataFrame([{"Rule": r.rule_id, "Name": r.name, "Why it fired": r.detail} for r in dec.rules]) if dec.rules
                 else pd.DataFrame({"Info": ["No special rule - standard queue-proportional plan"]}), width="stretch")
    c1, c2 = st.columns(2)
    c1.subheader("Prediction drivers (selected model)")
    c1.dataframe(pd.DataFrame(contrib, columns=["feature", "contribution (veh/h)" if "named_steps" not in dir(model) else "contribution (scaled)", "value"]), width="stretch")
    c2.subheader("Recommended green times")
    c2.dataframe(pd.DataFrame({"Direction": dec.phase_order, "Green (s)": [dec.green[d] for d in dec.phase_order]}), width="stretch")
    st.subheader("Sensor health & event monitoring (whole dataset)")
    mm = df.set_index("date_time").resample("MS").agg(faulty=("sensor_faulty", "sum"), incidents=("incident_reported", "sum"), ambulances=("ambulance_detected", "sum"))
    st.bar_chart(mm)
    st.caption("Ambulance events by approach direction: " + json.dumps(res["eda"]["stats"]["ambulance_by_direction"]))


# ----------------------------------------------------------------- page 4
def page_signals(df, X, y, meta, res):
    st.header("Dashboard 4 - Adaptive Signal Optimization")
    model = get_model(); prior = prior_from_data(df)
    st.caption("Queues below are the dataset's observed vehicle counts for the chosen hour (or what-if values). "
               "Simulation results elsewhere use a separate, clearly-labelled simulated queue process.")
    test_times = list(res["pred"].time)
    ts = st.select_slider("Hour", options=test_times, value=test_times[40], format_func=lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"))
    row = df[df.date_time == ts].iloc[0]
    pred = float(model.predict(X.loc[meta.index[meta.time == ts]])[0])
    c = st.columns(5)
    amb = c[0].checkbox("Emergency demo: ambulance", bool(row.ambulance_detected))
    amb_dir = c[1].selectbox("Ambulance from", DIRS, index=DIRS.index(row.ambulance_approach) if row.ambulance_approach in DIRS else 2)
    fault = c[2].checkbox("Sensor faulty", row.sensor_status == "faulty")
    inc = c[3].checkbox("Incident", bool(row.incident_reported))
    boost = c[4].slider("Boost North queue", 0, 30, 0)
    obs = observation_from_row(row, pred)
    obs.queues["North"] += boost
    obs.ambulance_detected, obs.ambulance_approach = amb, amb_dir if amb else "None"
    obs.sensor_faulty, obs.incident_reported = fault, inc
    dec = decide(obs, prior=prior)
    fx = fixed_time_plan(); fx_order = list(PHASE_ORDER)
    left, right = st.columns([1, 1])
    with left:
        t = st.slider("Second within cycle", 0, dec.cycle_s - 1, 5)
        st.pyplot(draw_intersection(obs.queues, dec.green, dec.phase_order, t, C.SIGNAL["clearance_s"],
                                    dec.emergency_direction, "Adaptive plan (blue = queue, red = ambulance lane)"))
    with right:
        st.subheader("Fixed-time vs adaptive green allocation")
        st.pyplot(plan_chart({"Fixed-time": fx, "Adaptive": dec.green}, {"Fixed-time": fx_order, "Adaptive": dec.phase_order}))
        st.dataframe(pd.DataFrame({"Direction": DIRS, "Queue": [obs.queues[d] for d in DIRS],
                                   "Fixed green (s)": [fx[d] for d in DIRS], "Adaptive green (s)": [dec.green[d] for d in DIRS]}), width="stretch")
        st.code(dec.explanation())
    st.subheader("Live mini-simulation (1 hour, same arrivals for both strategies)")
    seed = st.number_input("Random seed", 0, 999, 1)
    if st.button("Run simulation for this hour"):
        from src.simulation import HourSpec, Scenario, make_arrivals, simulate, _shares
        h = HourSpec(volume=float(row.traffic_volume), pred=pred, shares=_shares(row, prior), weather=row.weather_main,
                     rain=float(row.rain_1h), snow=float(row.snow_1h), incident=inc, faulty=fault,
                     ambulances=[(900, amb_dir)] if amb else [])
        scn = Scenario("live", [h]); A = make_arrivals(scn, int(seed)); out = {}
        for s in ("Fixed-time", "Adaptive (full)"):
            r = simulate(scn, s, int(seed), prior=prior, arrivals=A); r.pop("_hour_delay"); out[s] = r
        t = pd.DataFrame(out).T[["avg_queue_veh", "avg_delay_s_per_veh", "remaining_queue", "ambulance_avg_delay_s", "emergency_preemptions"]]
        st.dataframe(t.astype(float).style.format("{:.1f}"), width="stretch")
        st.caption("Simulated estimates under stated assumptions - not measurements from the CSV.")


# ----------------------------------------------------------------- page 5
def page_eval(df, res):
    st.header("Dashboard 5 - Performance Evaluation")
    t1, t2, t3, t4 = st.tabs(["Congestion analysis", "Prediction performance", "Simulation results", "Adaptive strategy evaluation"])
    with t1:
        s = res["eda"]["stats"]
        st.write("Congestion class share (%):", s["congestion_class_share_pct"])
        show_fig("eda_volume.png"); show_fig("eda_queues.png")
        st.write("Peak hour (mean volume):", s["peak_hour_mean_volume"])
        st.subheader("Faulty-sensor investigation"); show_fig("eda_faulty_sensor.png")
    with t2:
        st.dataframe(res["metrics"].drop(columns=["best_params"]), width="stretch")
        st.dataframe(res["seg"], width="stretch")
        show_fig("mae_by_hour.png"); show_fig("train_time.png")
    with t3:
        if res["stress"] is None:
            st.warning("Run `python run_pipeline.py --phase simulate` first."); return
        cols = ["scenario", "strategy", "avg_queue_veh_mean", "avg_delay_s_per_veh_mean", "remaining_queue_mean", "ambulance_avg_delay_s_mean", "emergency_preemptions_mean"]
        sc = st.selectbox("Scenario", sorted(res["stress"].scenario.unique()))
        t = res["stress"][res["stress"].scenario == sc][cols].set_index("strategy").drop(columns="scenario")
        st.dataframe(t.style.format("{:.1f}"), width="stretch")
        st.bar_chart(t[["avg_delay_s_per_veh_mean"]])
        st.subheader("Replay of every complete test day (real dataset event flags)")
        rp = res["replay"][["strategy", "avg_queue_veh_mean", "avg_delay_s_per_veh_mean", "ambulance_avg_delay_s_mean", "emergency_preemptions_mean"]].set_index("strategy")
        st.dataframe(rp.style.format("{:.1f}"), width="stretch")
    with t4:
        if res["band"] is None: return
        b = res["band"]
        pv = b[b.strategy.isin(["Fixed-time", "Adaptive (full)"])].pivot(index="band", columns="strategy", values="avg_delay_s_per_veh").loc[["Low", "Medium", "High"]]
        st.write("Average simulated delay per vehicle by congestion band (replay days)"); st.bar_chart(pv)
        if res["sens"] is not None:
            sv = res["sens"].groupby(["demand_scale", "strategy"]).avg_delay_s_per_veh.mean().unstack()
            st.write("Sensitivity to the demand-scale assumption (evening peak scenario)"); st.line_chart(sv)
        rd = res["replay_days"]
        w = rd.pivot(index="date", columns="strategy", values="avg_delay_s_per_veh")
        st.write("Per-day average delay (s/veh)"); st.line_chart(w[["Fixed-time", "Adaptive (full)"]])
        st.info("All numbers on this tab come from the simplified simulation (see README assumptions). They are not field measurements.")


def main():
    if not (C.RES_DIR / "best_model.json").exists():
        st.error("Results not found. Run `python run_pipeline.py` first."); return
    df, X, y, meta = get_data(); res = get_results()
    st.sidebar.title("🚦 From Traffic Data to Intelligent Decisions")
    page = st.sidebar.radio("Dashboard", ["1 Traffic Overview", "2 ML Model Comparison", "3 Decision Engine",
                                          "4 Adaptive Signal Optimization", "5 Performance Evaluation"])
    {"1": lambda: page_overview(df, res), "2": lambda: page_models(res), "3": lambda: page_decisions(df, X, y, meta, res),
     "4": lambda: page_signals(df, X, y, meta, res), "5": lambda: page_eval(df, res)}[page[0]]()


main()
