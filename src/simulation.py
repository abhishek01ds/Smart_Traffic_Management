"""Phase 8 - simplified single-intersection simulation (NOT real-world control).

Assumptions (all in config.SIGNAL and documented in the report):
 * 4 approaches, one phase each (N>E>S>W) -> no conflicting greens.
 * Arrivals: Poisson, rate = dataset hourly volume x demand_scale x direction
   share / 3600 per second.  Direction share = 50 % that hour's CSV queue share
   + 50 % long-run share (the CSV has no per-direction volume).
 * Discharge: saturation_veh_per_s per approach after a 2 s start-up loss.
 * Queue dynamics solved per phase with a Lindley (reflected random walk)
   recursion at 1-second resolution.
 * Delay = integral of queue length over time (vehicle-seconds) -> average
   delay per vehicle by Little's law.  These are SIMULATED estimates only.
 * Ambulance: detected `ambulance_lead_s` before reaching the stop line; its
   delay = wait for green + time to clear the vehicles ahead of it.
 * All strategies in one comparison share the SAME arrival stream
   (common random numbers) so differences are due to control only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from . import config as C
from .decision_engine import decide, Observation, DEFAULT_PRIOR
from .signal_optimizer import PHASE_ORDER, fixed_time_plan

DIRS = C.DIRECTIONS
IDX = {d: i for i, d in enumerate(DIRS)}

STRATEGIES = {
    "Fixed-time": dict(kind="fixed"),
    "Adaptive (full)": dict(kind="adaptive"),
    "Adaptive (no cycle change)": dict(kind="adaptive", adapt_cycle=False),
    "Adaptive (no ML prediction)": dict(kind="adaptive", use_prediction=False),
    "Adaptive (reactive: current volume)": dict(kind="adaptive", pred_source="current"),
    "Adaptive (oracle: true volume)": dict(kind="adaptive", pred_source="oracle"),
    "Adaptive (no ambulance rule)": dict(kind="adaptive", enable_emergency=False),
    "Adaptive (blind to sensor fault)": dict(kind="adaptive", trust_sensor_flag=False),
}


@dataclass
class HourSpec:
    volume: float
    pred: float
    shares: dict
    weather: str = "Clear"
    rain: float = 0.0
    snow: float = 0.0
    incident: bool = False
    incident_dir: str = "East"
    faulty: bool = False
    ambulances: list = field(default_factory=list)   # [(second_in_hour, direction)]
    timestamp: str = ""


@dataclass
class Scenario:
    name: str
    hours: list


def lindley(q0: float, a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """q_t = max(0, q_{t-1} + a_t - s_t) in closed form (reflected walk)."""
    y = q0 + np.cumsum(a - s)
    return y - np.minimum.accumulate(np.minimum(y, 0.0))


def make_arrivals(scn: Scenario, seed: int, sig=None) -> np.ndarray:
    s = sig or C.SIGNAL
    rng = np.random.default_rng(seed)
    blocks = []
    for h in scn.hours:
        lam = np.array([h.volume * s["demand_scale"] * h.shares[d] / 3600.0 for d in DIRS])
        blocks.append(rng.poisson(lam, size=(3600, 4)))
    return np.vstack(blocks).astype(float)


def hour_flow(h: HourSpec, sig) -> np.ndarray:
    """Discharge rate per approach for that hour (weather + incident)."""
    r = np.full(4, sig["saturation_veh_per_s"])
    if h.weather in ("Thunderstorm", "Snow", "Fog") or h.rain >= 7.6 or h.snow > 0.5:
        r *= sig["adverse_flow_factor"]
    if h.incident:
        r[IDX[h.incident_dir]] *= sig["incident_flow_factor"]
    return r


def simulate(scn: Scenario, strategy: str, seed: int = 0, prior=None, sig=None,
             arrivals=None, record: bool = False) -> dict:
    sig = sig or C.SIGNAL
    prior = prior or DEFAULT_PRIOR
    cfg = dict(STRATEGIES[strategy])
    kind = cfg.pop("kind")
    pred_source = cfg.pop("pred_source", "model")
    H = len(scn.hours); T = 3600 * H
    A = arrivals if arrivals is not None else make_arrivals(scn, seed, sig)
    rng = np.random.default_rng(seed + 7919)
    garbage = rng.integers(0, 31, size=(T // 30 + 5, 4))      # faulty-sensor readings

    # ambulance events: global time of arrival at stop line
    ambs = []
    for hi, h in enumerate(scn.hours):
        for sec, d in h.ambulances:
            ta = hi * 3600 + int(sec)
            ambs.append(dict(ta=ta, det=ta - sig["ambulance_lead_s"], dir=d, served=False,
                             delay=None, preempted=False))
    ambs.sort(key=lambda a: a["ta"])

    q = np.zeros(4)
    t = 0
    total_delay = 0.0; max_q = 0.0; cyc_q = []
    hour_delay = np.zeros(H); preemptions = 0; timeline = []; cycles = 0
    min_g, startup = sig["min_green_s"], sig["startup_lost_s"]

    def run_phase(t, d, g, cl):
        """Advance g seconds of green for d then cl seconds of clearance."""
        nonlocal q, total_delay
        hi = min(t // 3600, H - 1); h = scn.hours[hi]
        n = min(g + cl, T - t)
        if n <= 0: return t
        a = A[t:t + n]; i = IDX[d]
        rate = hour_flow(h, sig)[i]
        s = np.zeros(n)
        gg = min(g, n)
        if gg > startup: s[startup:gg] = rate
        traj = lindley(q[i], a[:, i], s)
        others = [j for j in range(4) if j != i]
        qo = q[others] + np.cumsum(a[:, others], axis=0)
        delay = traj.sum() + qo.sum()
        total_delay += delay; hour_delay[hi] += delay
        # ambulances on this approach whose arrival is before green ends
        # Ambulance model: vehicles ahead pull aside, so the ambulance only
        # waits for its own green (+ a small yield time if it arrives on red).
        for am in ambs:
            if am["served"] or am["dir"] != d or am["ta"] >= t + gg: continue
            if am["ta"] >= t + startup:                  # green already flowing
                am["delay"] = 0.0
            else:                                        # arrived on red / start-up
                am["delay"] = (t + startup) - am["ta"] + sig["ambulance_yield_s"]
            am["served"] = True
        q[:] = 0; q[i] = traj[-1]; q[others] = qo[-1]
        return t + n

    while t < T:
        hi = min(t // 3600, H - 1); h = scn.hours[hi]
        cycles += 1
        cyc_q.append(q.sum())
        # ---------- plan
        if kind == "fixed":
            green = fixed_time_plan(sig); order = list(PHASE_ORDER)
            clearance = sig["clearance_s"]; cyc_len = sig["cycle_s"]; label = "fixed"
        else:
            seen = {d: float(q[IDX[d]]) for d in DIRS}
            if h.faulty:
                seen = {d: float(garbage[(t // 30) % len(garbage)][IDX[d]]) for d in DIRS}
            cur_vol = scn.hours[hi - 1].volume if hi > 0 else h.volume
            pv = {"model": h.pred, "current": cur_vol, "oracle": h.volume}[pred_source]
            obs = Observation(queues=seen, current_volume=cur_vol, predicted_volume=pv,
                              incident_reported=h.incident, sensor_faulty=h.faulty,
                              weather_main=h.weather, rain_1h=h.rain, snow_1h=h.snow)
            dec = decide(obs, prior=prior, sig=sig, **cfg)
            green, order = dec.green, dec.phase_order
            clearance = sig["clearance_s"] + (sig["adverse_extra_clearance_s"] if any(r.rule_id == 6 for r in dec.rules) else 0)
            cyc_len = dec.cycle_s; label = ",".join(str(r.rule_id) for r in dec.rules)
        if record:
            timeline.append(dict(t=t, hour=hi, queues=q.copy(), green=dict(green), cycle=cyc_len, rules=label))
        # ---------- execute phases
        preempt_enabled = kind == "adaptive" and cfg.get("enable_emergency", True)
        for d in order:
            if t >= T: break
            g = green[d]
            am = None
            if preempt_enabled:
                for a_ in ambs:
                    if (not a_["served"]) and (not a_["preempted"]) and a_["dir"] != d and a_["det"] < t + g + clearance:
                        am = a_; break
            if am is not None:
                g_eff = int(np.clip(am["det"] - t, min(g, 5), g))
                t = run_phase(t, d, g_eff, clearance)
                am["preempted"] = True; preemptions += 1
                eg = sig["emergency_green_s"]
                t = run_phase(t, am["dir"], eg, clearance)
                if record:
                    timeline.append(dict(t=t, hour=min(t // 3600, H - 1), queues=q.copy(),
                                         green={am["dir"]: eg}, cycle=eg + clearance, rules="1-preempt"))
                break                                    # re-plan a fresh cycle
            t = run_phase(t, d, g, clearance)
    # ambulances never served within the horizon: charge remaining time
    for a_ in ambs:
        if not a_["served"]:
            a_["delay"] = max(T - a_["ta"], 0); a_["served"] = False
    arrived = A.sum()
    out = dict(strategy=strategy, scenario=scn.name, seed=seed, hours=H, arrivals=float(arrived),
               avg_queue_veh=total_delay / T, avg_queue_cycle_start=float(np.mean(cyc_q)),
               max_queue_veh=float(max(max_q, max(cyc_q))), remaining_queue=float(q.sum()),
               avg_delay_s_per_veh=total_delay / max(arrived, 1), total_delay_veh_s=total_delay,
               ambulances=len(ambs),
               ambulance_avg_delay_s=float(np.mean([a["delay"] for a in ambs])) if ambs else np.nan,
               ambulance_max_delay_s=float(np.max([a["delay"] for a in ambs])) if ambs else np.nan,
               emergency_preemptions=preemptions, cycles=cycles)
    if record:
        out["timeline"] = timeline; out["hour_delay"] = hour_delay.tolist()
        out["ambulance_events"] = [dict(a) for a in ambs]
    out["_hour_delay"] = hour_delay
    return out


# ---------------------------------------------------------------- scenarios
def _shares(row, prior):
    if row["sensor_status"] == "faulty":
        return dict(prior)
    q = {d: float(row[c]) for d, c in C.QUEUE_COLS.items()}
    tot = sum(q.values())
    qs = {d: (q[d] + 1) / (tot + 4) for d in DIRS}
    return {d: 0.5 * qs[d] + 0.5 * prior[d] for d in DIRS}


def hours_from_frame(df: pd.DataFrame, pred_map: dict, prior, rng, real_events=True) -> list:
    """df = consecutive rows of traffic_cleaned (one per hour)."""
    out = []
    for _, r in df.iterrows():
        ts = r["date_time"]
        pred = pred_map.get(ts, float(r["traffic_volume"]))
        amb = []
        if real_events and r["ambulance_detected"] == 1 and r["ambulance_approach"] in DIRS:
            amb = [(int(rng.integers(120, 3400)), r["ambulance_approach"])]
        out.append(HourSpec(volume=float(r["traffic_volume"]), pred=float(pred), shares=_shares(r, prior),
                            weather=str(r["weather_main"]), rain=float(r["rain_1h"]), snow=float(r["snow_1h"]),
                            incident=bool(real_events and r["incident_reported"] == 1), incident_dir="East",
                            faulty=bool(real_events and r["sensor_status"] == "faulty"),
                            ambulances=amb, timestamp=str(ts)))
    return out


def stress_scenarios(df: pd.DataFrame, pred_map: dict, prior, seed=0) -> list:
    """Controlled 8-hour evening windows (14:00-21:00) on a typical test weekday."""
    rng = np.random.default_rng(seed)
    d = df.copy(); d["date"] = d.date_time.dt.date
    wk = d[(d.date_time.dt.dayofweek < 4)]
    day_mean = wk[wk.date_time.dt.hour.between(14, 21)].groupby("date").traffic_volume.mean()
    day = (day_mean - day_mean.median()).abs().idxmin()          # typical weekday
    win = d[(d.date == day) & d.date_time.dt.hour.between(14, 21)]
    base = hours_from_frame(win, pred_map, prior, rng, real_events=False)

    def clone(**kw):
        import copy
        hs = copy.deepcopy(base)
        for i, h in enumerate(hs):
            for k, v in kw.get("per_hour", lambda i, h: {})(i, h).items(): setattr(h, k, v)
        return hs
    amb_dirs = DIRS * 3
    scns = [
        Scenario("Evening peak (no events)", clone()),
        Scenario("Ambulance stress (16 ambulances)", clone(per_hour=lambda i, h: {"ambulances": [
            (600, amb_dirs[(2 * i) % 12]), (2400, amb_dirs[(2 * i + 1) % 12])]})),
        Scenario("Sensor failure (hours 3-6)", clone(per_hour=lambda i, h: {"faulty": 2 <= i <= 5})),
        Scenario("Incident (East lane blocked, hours 3-5)", clone(per_hour=lambda i, h: {"incident": 2 <= i <= 4})),
        Scenario("Adverse weather (thunderstorm)", clone(per_hour=lambda i, h: {"weather": "Thunderstorm", "rain": 8.0})),
        Scenario("Combined stress", clone(per_hour=lambda i, h: {
            "ambulances": [(900, amb_dirs[i % 12])], "faulty": 2 <= i <= 4, "incident": 3 <= i <= 5,
            "weather": "Thunderstorm" if i >= 4 else "Clear", "rain": 8.0 if i >= 4 else 0.0})),
    ]
    for s in scns: s.name += f" [{day}]"
    return scns
