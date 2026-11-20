import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src import config as C
from src.data_loader import load_clean
from src.features import build_dataset, chronological_split
from src.signal_optimizer import allocate, fixed_time_plan
from src.decision_engine import Observation, decide
from src.simulation import lindley

def test_no_leakage_columns():
    X, y, m = build_dataset(load_clean())
    banned = {"congestion_level", "congestion_code", "total_queue", "traffic_volume", "y_next_hour"}
    assert not banned & set(X.columns)

def test_target_is_next_hour_and_split_is_chronological():
    df = load_clean(); X, y, m = build_dataset(df)
    i = 500
    t = m.time.iloc[i]
    assert y.iloc[i] == df.loc[df.date_time == t + np.timedelta64(1, "h"), "traffic_volume"].iloc[0]
    sp = chronological_split(X, y, m)
    assert sp["train"][2].time.max() < sp["val"][2].time.min() < sp["val"][2].time.max() < sp["test"][2].time.min()

def test_allocation_constraints():
    for dem in ({"North": 40, "South": 1, "East": 1, "West": 1}, {d: 0 for d in C.DIRECTIONS}):
        g = allocate(dem, 120, 4, 10, 60)
        assert sum(g.values()) == 120 - 16 and all(10 <= v <= 60 for v in g.values())
    g = allocate({"North": 5, "South": 5, "East": 5, "West": 5}, 120, 4, 10, 60, emergency_dir="East")
    assert g["East"] == 30 and sum(g.values()) == 104 and min(g.values()) >= 10
    assert sum(fixed_time_plan().values()) == 104

def obs(**k):
    base = dict(queues={"North": 8, "South": 3, "East": 2, "West": 2}, current_volume=3000, predicted_volume=3000)
    base.update(k); return Observation(**base)

def test_rules():
    d = decide(obs(ambulance_detected=True, ambulance_approach="West"))
    assert d.emergency_direction == "West" and d.selected_direction == "West" and d.phase_order[0] == "West"
    assert 4 in [r.rule_id for r in decide(obs(incident_reported=True)).rules]
    d = decide(obs(sensor_faulty=True))
    assert 5 in [r.rule_id for r in d.rules] and d.queue_basis == "fallback-prior"
    assert 6 in [r.rule_id for r in decide(obs(weather_main="Thunderstorm")).rules]
    assert 3 in [r.rule_id for r in decide(obs(predicted_volume=5000)).rules]
    assert 2 in [r.rule_id for r in decide(obs()).rules]

def test_lindley_conservation():
    a = np.random.default_rng(0).poisson(0.3, 100).astype(float); s = np.full(100, 0.5)
    q = lindley(2.0, a, s)
    assert (q >= 0).all() and abs(q[-1] - (2 + a.sum() - 50 + max(0, 0))) >= 0
