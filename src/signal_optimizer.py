"""Phase 7 - adaptive green-time allocation.

Four-phase plan: every approach has its OWN phase (N, E, S, W order), so two
conflicting movements are never green together (conflict safety by design).

Method (Webster-style proportional allocation with hard constraints):
   demand_d = queue_d + expected_arrivals_d       (vehicles)
   green_d  ~ demand_d                            (proportional)
   subject to  min_green <= green_d <= max_green  and  sum(green) = cycle - lost
solved by iterative water-filling, then rounded to whole seconds.
"""
from __future__ import annotations
from . import config as C

PHASE_ORDER = ["North", "East", "South", "West"]


def water_fill(weights: dict, total: float, lo: float, hi: float) -> dict:
    """Proportional split of `total` with box constraints [lo, hi] per item."""
    keys = list(weights)
    if not (len(keys) * lo - 1e-9 <= total <= len(keys) * hi + 1e-9):
        raise ValueError("infeasible: total green outside [n*min, n*max]")
    alloc, free = {}, set(keys)
    remaining = total
    while free:
        wsum = sum(max(weights[k], 1e-9) for k in free)
        trial = {k: remaining * max(weights[k], 1e-9) / wsum for k in free}
        viol = [k for k in free if trial[k] < lo - 1e-9 or trial[k] > hi + 1e-9]
        if not viol:
            alloc.update(trial); break
        # clamp the most violating item(s) and redistribute
        k = max(viol, key=lambda k: max(lo - trial[k], trial[k] - hi))
        alloc[k] = min(max(trial[k], lo), hi)
        remaining -= alloc[k]; free.discard(k)
    return alloc


def round_to_total(alloc: dict, total: int, lo: int, hi: int) -> dict:
    """Largest-remainder rounding that keeps sum == total and respects bounds."""
    base = {k: int(min(max(v // 1, lo), hi)) for k, v in alloc.items()}
    diff = int(total - sum(base.values()))
    order = sorted(alloc, key=lambda k: alloc[k] - alloc[k] // 1, reverse=(diff > 0))
    i = 0
    while diff != 0 and i < 10 * len(order):
        k = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if lo <= base[k] + step <= hi:
            base[k] += step; diff -= step
        i += 1
    return base


def fixed_time_plan(sig: dict | None = None) -> dict:
    """Baseline: equal green split, fixed cycle."""
    s = sig or C.SIGNAL
    total = s["cycle_s"] - 4 * s["clearance_s"]
    g = total // 4
    return {d: int(g) for d in PHASE_ORDER}


def allocate(demand: dict, cycle_s: int, clearance_s: int, min_g: int, max_g: int,
             emergency_dir: str | None = None, emergency_g: int | None = None) -> dict:
    """Return integer green seconds per direction (sum = cycle - 4*clearance)."""
    total = int(cycle_s - 4 * clearance_s)
    if emergency_dir:
        e = int(min(emergency_g or C.SIGNAL["emergency_green_s"], max_g))
        rest = {d: demand[d] for d in demand if d != emergency_dir}
        # feasibility: the other three need at least 3*min
        e = min(e, total - 3 * min_g)
        part = water_fill(rest, total - e, min_g, max_g)
        part = round_to_total(part, total - e, min_g, max_g)
        part[emergency_dir] = e
        return {d: int(part[d]) for d in PHASE_ORDER}
    part = water_fill(demand, total, min_g, max_g)
    part = round_to_total(part, total, min_g, max_g)
    return {d: int(part[d]) for d in PHASE_ORDER}
