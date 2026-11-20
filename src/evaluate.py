"""Phase 5 - model comparison, statistical check and final model selection."""
from __future__ import annotations
import json, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from . import config as C
from .train_models import SLUG

sns.set_theme(style="whitegrid")
NAMES = list(SLUG)


def _save(name):
    plt.tight_layout(); plt.savefig(C.FIG_DIR / name, dpi=130); plt.close()


def select_best(res: pd.DataFrame) -> dict:
    """Rule fixed in advance (see config): choose the lowest VALIDATION RMSE;
    if a simpler model is within SELECTION_TOLERANCE of it, prefer the simpler
    one.  The TEST set is then used only to confirm, never to choose."""
    r = res.set_index("model")
    best = r["val_RMSE"].idxmin()
    limit = r.loc[best, "val_RMSE"] * (1 + C.SELECTION_TOLERANCE)
    cands = [m for m in r.index if r.loc[m, "val_RMSE"] <= limit]
    chosen = min(cands, key=lambda m: C.SIMPLICITY_RANK[m])
    best_test = r["test_RMSE"].idxmin()
    return {"lowest_val_rmse_model": best, "within_tolerance": cands, "selected": chosen,
            "lowest_test_rmse_model": best_test,
            "test_confirms_selection": bool(chosen == best_test),
            "tolerance": C.SELECTION_TOLERANCE}


def block_bootstrap_rmse_diff(y, p1, p2, block=24, n=2000, seed=C.RANDOM_STATE):
    """Paired day-block bootstrap of RMSE(model1) - RMSE(model2)."""
    rng = np.random.default_rng(seed)
    e1, e2 = (y - p1) ** 2, (y - p2) ** 2
    nb = len(y) // block
    e1b = e1[:nb * block].reshape(nb, block); e2b = e2[:nb * block].reshape(nb, block)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, nb, nb)
        diffs.append(np.sqrt(e1b[idx].mean()) - np.sqrt(e2b[idx].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


def run_evaluation() -> dict:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(C.RES_DIR / "model_metrics.csv")
    base = pd.read_csv(C.RES_DIR / "baseline_metrics.csv")
    pt = pd.read_csv(C.RES_DIR / "predictions_test.csv", parse_dates=["time", "target_time"])
    sel = select_best(res)
    y = pt["actual"].values
    mean_vol = float(y.mean())
    res["test_MAE_pct_of_mean_volume"] = 100 * res["test_MAE"] / mean_vol
    res["test_RMSE_pct_of_mean_volume"] = 100 * res["test_RMSE"] / mean_vol
    res.to_csv(C.RES_DIR / "model_metrics.csv", index=False)

    # pairwise significance (selected vs each other model)
    pairs = []
    for other in NAMES:
        if other == sel["selected"]: continue
        d, lo, hi = block_bootstrap_rmse_diff(y, pt[sel["selected"]].values, pt[other].values)
        pairs.append({"selected": sel["selected"], "other": other,
                      "rmse_diff_selected_minus_other": d, "ci95_low": lo, "ci95_high": hi,
                      "significant": bool(hi < 0 or lo > 0)})
    pd.DataFrame(pairs).to_csv(C.RES_DIR / "pairwise_bootstrap.csv", index=False)

    # segment errors (by hour + congestion band of the target)
    pt["abs_err"] = {m: (pt[m] - pt["actual"]).abs() for m in NAMES}.get(NAMES[0])
    seg = []
    band = pd.cut(pt["actual"], [0, C.LOW_MAX, C.HIGH_MIN, 1e9], labels=["Low", "Medium", "High"], right=False)
    for m in NAMES:
        e = (pt[m] - pt["actual"]).abs()
        for b in ["Low", "Medium", "High"]:
            seg.append({"model": m, "band": b, "MAE": float(e[band == b].mean()), "n": int((band == b).sum())})
    pd.DataFrame(seg).to_csv(C.RES_DIR / "error_by_congestion_band.csv", index=False)

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, k in zip(ax, ["test_MAE", "test_RMSE", "test_R2"]):
        sns.barplot(data=res, x="model", y=k, ax=a, hue="model", legend=False)
        a.set_title(k.replace("test_", "") + " (test)"); a.set_xlabel("")
        a.tick_params(axis="x", rotation=15)
        for p in a.patches: a.annotate(f"{p.get_height():.3f}" if k.endswith("R2") else f"{p.get_height():.0f}",
                                        (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    _save("model_metrics_bar.png")

    last = pt.iloc[:24 * 7]
    plt.figure(figsize=(13, 4.5))
    plt.plot(last.target_time, last.actual, "k", lw=2, label="Actual")
    for m in NAMES: plt.plot(last.target_time, last[m], lw=1.2, label=m)
    plt.title("Actual vs predicted next-hour volume (first test week)"); plt.legend(); _save("actual_vs_pred_week.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, m in zip(ax, NAMES):
        a.scatter(pt.actual, pt[m], s=4, alpha=.4); lim = [0, pt.actual.max()]
        a.plot(lim, lim, "r--"); a.set_title(m); a.set_xlabel("Actual"); a.set_ylabel("Predicted")
    _save("pred_scatter.png")

    plt.figure(figsize=(8, 4))
    for m in NAMES: sns.kdeplot(pt[m] - pt.actual, label=m)
    plt.title("Residual distribution (test)"); plt.xlabel("Prediction error"); plt.legend(); _save("residuals.png")

    h = pt.assign(hour=pt.target_time.dt.hour)
    plt.figure(figsize=(9, 4))
    for m in NAMES:
        plt.plot(h.groupby("hour").apply(lambda g: (g[m] - g.actual).abs().mean(), include_groups=False), label=m)
    plt.title("MAE by target hour of day"); plt.xlabel("Hour"); plt.ylabel("MAE"); plt.legend(); _save("mae_by_hour.png")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=res.melt(id_vars="model", value_vars=["train_time_s"]), x="model", y="value", hue="model", legend=False)
    plt.ylabel("seconds"); plt.title("Training time (final refit)"); _save("train_time.png")

    out = {"selection": sel, "baselines": base.to_dict("records"), "pairwise": pairs,
           "mean_test_volume": mean_vol}
    best = sel["selected"]
    shutil.copy(C.MODEL_DIR / f"{SLUG[best]}.joblib", C.MODEL_DIR / "best_model.joblib")
    (C.RES_DIR / "best_model.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), indent=2))
