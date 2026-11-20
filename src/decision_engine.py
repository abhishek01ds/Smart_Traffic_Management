"""Phase 6 - hybrid decision engine: ML prediction + IF-THEN rules + optimisation.

`decide(obs)` returns a `Decision` that carries everything the professor (and
the dashboard) needs to see: condition, prediction, triggered rules, chosen
direction, green times and a plain-language reason.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from . import config as C
from .signal_optimizer import allocate, PHASE_ORDER

DIRS = C.DIRECTIONS
SEVERE_WEATHER = {"Thunderstorm", "Snow"}
ADVERSE_WEATHER = {"Thunderstorm", "Snow", "Fog"}

# Historical (healthy-sensor) directional demand share, used as a safe prior.
# Value comes from EDA (queue shares); recomputed in run_pipeline and stored.
DEFAULT_PRIOR = {"North": 0.356, "South": 0.261, "East": 0.215, "West": 0.168}


@dataclass
class Observation:
    queues: dict                      # {"North": int, ...}
    current_volume: float
    predicted_volume: float
    ambulance_detected: bool = False
    ambulance_approach: str = "None"
    incident_reported: bool = False
    sensor_faulty: bool = False
    weather_main: str = "Clear"
    rain_1h: float = 0.0
    snow_1h: float = 0.0
    timestamp: str = ""
    top_features: list = field(default_factory=list)   # [(name, contribution)]


@dataclass
class RuleHit:
    rule_id: int
    name: str
    detail: str


@dataclass
class Decision:
    condition: str
    predicted_level: str
    cycle_s: int
    green: dict
    phase_order: list
    selected_direction: str
    emergency_direction: str | None
    rules: list
    alerts: list
    operator_review: bool
    queue_basis: str                  # "sensor" | "fallback-prior"
    demand: dict
    reasons: list
    top_features: list
    predicted_volume: float
    current_volume: float
    queues_used: dict

    def explanation(self) -> str:
        lines = [f"Condition now: {self.condition} (volume {self.current_volume:.0f}); "
                 f"predicted next hour: {self.predicted_volume:.0f} ({self.predicted_level})."]
        for r in self.rules:
            lines.append(f"Rule {r.rule_id} [{r.name}] -> {r.detail}")
        if not self.rules:
            lines.append("No special rule fired -> standard queue-proportional plan.")
        g = ", ".join(f"{d} {self.green[d]}s" for d in self.phase_order)
        lines.append(f"Plan (cycle {self.cycle_s}s, order {'>'.join(self.phase_order)}): {g}.")
        lines.append(f"Main direction: {self.selected_direction}.")
        if self.operator_review:
            lines.append("OPERATOR REVIEW RECOMMENDED.")
        return "\n".join(lines)


def level_of(volume: float) -> str:
    return "Low" if volume < C.LOW_MAX else ("High" if volume > C.HIGH_MIN else "Medium")


def decide(obs: Observation, prior: dict | None = None, sig: dict | None = None,
           use_prediction: bool = True, trust_sensor_flag: bool = True,
           enable_emergency: bool = True, adapt_cycle: bool = True) -> Decision:
    """Apply rules 1-6 and the queue-based optimiser.

    use_prediction / trust_sensor_flag / enable_emergency / adapt_cycle exist
    so the ablation experiments in the simulation can switch parts off.
    """
    s = sig or C.SIGNAL
    prior = prior or DEFAULT_PRIOR
    rules, alerts, reasons = [], [], []
    operator = False
    cur_lvl = level_of(obs.current_volume)
    pred_lvl = level_of(obs.predicted_volume)

    # ---- Rule 5: faulty sensor (decided first: it decides which data we trust)
    use_prior = bool(obs.sensor_faulty and trust_sensor_flag)
    if use_prior:
        rules.append(RuleHit(5, "Faulty sensor",
                             "queue readings flagged unreliable -> ignored; using historical "
                             "direction shares as a safe fallback"))
        alerts.append("Sensor fault: queue data not trusted. Schedule maintenance.")
        queues_used = {d: 0.0 for d in DIRS}
        share = dict(prior)
        queue_basis = "fallback-prior"
    else:
        queues_used = {d: float(obs.queues[d]) for d in DIRS}
        tot = sum(queues_used.values())
        qshare = {d: (queues_used[d] + 1) / (tot + 4) for d in DIRS}
        share = {d: 0.5 * qshare[d] + 0.5 * prior[d] for d in DIRS}
        queue_basis = "sensor"

    # ---- Rule 3: future congestion (predicted volume)
    cycle = s["cycle_s"]
    rising = (obs.predicted_volume >= 1.15 * obs.current_volume and obs.predicted_volume >= C.LOW_MAX) \
        or pred_lvl == "High"
    if use_prediction and adapt_cycle:
        cycle = {"Low": s["cycle_low_s"], "Medium": s["cycle_s"], "High": s["cycle_high_s"]}[pred_lvl]
    if use_prediction and rising:
        rules.append(RuleHit(3, "Future congestion",
                             f"predicted {obs.predicted_volume:.0f} veh/h ({pred_lvl}) vs now "
                             f"{obs.current_volume:.0f} -> prepare for rising demand"
                             + (f", cycle set to {cycle}s" if adapt_cycle else "")))
        reasons.append("Rising demand predicted; plan prepared in advance.")

    # ---- Rule 6: weather
    clearance = s["clearance_s"]
    adverse = (obs.weather_main in ADVERSE_WEATHER) or obs.rain_1h >= 7.6 or obs.snow_1h > 0.5
    if adverse:
        clearance += s["adverse_extra_clearance_s"]
        cycle = max(cycle, s["cycle_s"])
        severe = obs.weather_main in SEVERE_WEATHER or obs.rain_1h >= 7.6
        rules.append(RuleHit(6, "Adverse weather",
                             f"{obs.weather_main} (rain {obs.rain_1h:.1f} mm, snow {obs.snow_1h:.1f} mm) -> "
                             f"longer clearance ({clearance}s) and no short cycles"
                             + ("; operator review advised" if severe else "")))
        if severe:
            operator = True; alerts.append("Severe weather: operator review advised.")

    # ---- expected arrivals per direction for the next cycle (vehicles)
    veh_per_cycle = (obs.predicted_volume if use_prediction else obs.current_volume) \
        * s["demand_scale"] * cycle / 3600.0
    demand = {d: queues_used[d] + veh_per_cycle * share[d] for d in DIRS}

    # ---- Rule 2: dominant queue
    max_g = s["max_green_s"]
    if not use_prior:
        top = max(DIRS, key=lambda d: queues_used[d])
        others = [queues_used[d] for d in DIRS if d != top]
        mean_o = sum(others) / 3
        if queues_used[top] >= 5 and queues_used[top] >= 1.5 * max(mean_o, 1):
            rules.append(RuleHit(2, "High traffic queue",
                                 f"{top} queue {queues_used[top]:.0f} is {queues_used[top] / max(mean_o, 1):.1f}x "
                                 f"the mean of the others -> larger green share"))
            reasons.append(f"{top} has the dominant queue, so it gets the longest green.")

    # ---- Rule 4: incident
    if obs.incident_reported:
        rules.append(RuleHit(4, "Traffic incident",
                             "incident reported -> alert + operator intervention; green times capped "
                             f"at {s['incident_max_green_s']}s so no approach is starved"))
        alerts.append("Incident reported: operator intervention required.")
        operator = True
        max_g = s["incident_max_green_s"]

    # ---- Rule 1: ambulance (highest priority; fail-safe even if sensor is faulty)
    emergency = None
    if enable_emergency and obs.ambulance_detected and obs.ambulance_approach in DIRS:
        emergency = obs.ambulance_approach
        rules.insert(0, RuleHit(1, "Ambulance priority",
                                f"ambulance approaching from {emergency} -> terminate conflicting greens "
                                f"after {clearance}s clearance, give {emergency} an emergency green of "
                                f"{s['emergency_green_s']}s, then resume the cycle"))
        alerts.append(f"EMERGENCY: ambulance from {emergency}.")
        if obs.sensor_faulty:
            alerts.append("Ambulance flag came from a faulty-sensor record: verify visually "
                          "(fail-safe: priority still granted).")
    elif obs.ambulance_detected and obs.ambulance_approach not in DIRS:
        alerts.append("Ambulance detected but direction unknown: operator must confirm.")
        operator = True

    green = allocate(demand, cycle, clearance, s["min_green_s"], max_g,
                     emergency_dir=emergency, emergency_g=s["emergency_green_s"])
    sel = emergency or max(DIRS, key=lambda d: green[d])
    order = ([emergency] + [d for d in PHASE_ORDER if d != emergency]) if emergency else list(PHASE_ORDER)
    return Decision(condition=cur_lvl, predicted_level=pred_lvl, cycle_s=cycle, green=green,
                    phase_order=order, selected_direction=sel, emergency_direction=emergency,
                    rules=sorted(rules, key=lambda r: r.rule_id), alerts=alerts,
                    operator_review=operator, queue_basis=queue_basis, demand=demand,
                    reasons=reasons, top_features=obs.top_features,
                    predicted_volume=obs.predicted_volume, current_volume=obs.current_volume,
                    queues_used=queues_used)


def observation_from_row(row, predicted_volume: float, top_features=None) -> Observation:
    """Build an Observation from one row of traffic_cleaned.csv."""
    return Observation(
        queues={d: float(row[c]) for d, c in C.QUEUE_COLS.items()},
        current_volume=float(row["traffic_volume"]), predicted_volume=float(predicted_volume),
        ambulance_detected=bool(row["ambulance_detected"]), ambulance_approach=str(row["ambulance_approach"]),
        incident_reported=bool(row["incident_reported"]), sensor_faulty=row["sensor_status"] == "faulty",
        weather_main=str(row["weather_main"]), rain_1h=float(row["rain_1h"]), snow_1h=float(row["snow_1h"]),
        timestamp=str(row["date_time"]), top_features=top_features or [])
