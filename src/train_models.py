"""Phase 4 - train exactly three regressors under identical conditions.

Protocol (same for all models):
 1. hyper-parameters chosen on the VALIDATION block only (small grid),
 2. final model refit on TRAIN+VAL with the chosen parameters,
 3. TEST block touched once, for the final report.
"""
from __future__ import annotations
import itertools, json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from . import config as C
from .data_loader import load_clean
from .features import build_dataset, chronological_split

SLUG = {"Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting"}

GRIDS = {
    "Linear Regression": [{}],
    "Random Forest": [dict(n_estimators=200, max_depth=d, min_samples_leaf=l, max_features=0.5)
                      for d, l in itertools.product([12, None], [2, 5])],
    "Gradient Boosting": [dict(n_estimators=300, learning_rate=lr, max_depth=d, subsample=0.8)
                          for lr, d in itertools.product([0.05, 0.1], [3, 4])],
}


def make_model(name: str, **params):
    if name == "Linear Regression":
        # scaling does not change OLS predictions but makes coefficients comparable
        return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    if name == "Random Forest":
        return RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **params)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=C.RANDOM_STATE, **params)
    raise ValueError(name)


def metrics(y, p) -> dict:
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def train_all() -> dict:
    X, y, meta = build_dataset(load_clean())
    sp = chronological_split(X, y, meta)
    (Xtr, ytr, _), (Xva, yva, mva), (Xte, yte, mte) = sp["train"], sp["val"], sp["test"]
    Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True); C.RES_DIR.mkdir(parents=True, exist_ok=True)

    rows, best_params, pred_val, pred_test = [], {}, {}, {}
    for name, grid in GRIDS.items():
        # --- tune on validation
        scored = []
        for params in grid:
            m = make_model(name, **params).fit(Xtr, ytr)
            scored.append((metrics(yva, m.predict(Xva))["RMSE"], params))
        _, bp = min(scored, key=lambda t: t[0])
        best_params[name] = bp
        # validation metrics of the tuned model (trained on TRAIN only)
        m_tr = make_model(name, **bp).fit(Xtr, ytr)
        pred_val[name] = m_tr.predict(Xva)
        mv = metrics(yva, pred_val[name])
        # --- final refit on train+val; timing; test evaluation
        t0 = time.perf_counter(); final = make_model(name, **bp).fit(Xtv, ytv)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter(); pred_test[name] = final.predict(Xte)
        pred_ms = 1000 * (time.perf_counter() - t0)
        mt = metrics(yte, pred_test[name])
        joblib.dump(final, C.MODEL_DIR / f"{SLUG[name]}.joblib")
        rows.append({"model": name, **{f"val_{k}": v for k, v in mv.items()},
                     **{f"test_{k}": v for k, v in mt.items()},
                     "train_time_s": fit_s, "predict_ms_per_1000": pred_ms * 1000 / len(Xte),
                     "n_candidates_tuned": len(grid), "best_params": json.dumps(bp)})
        print(f"{name:18s} val RMSE {mv['RMSE']:.1f} | test RMSE {mt['RMSE']:.1f} "
              f"MAE {mt['MAE']:.1f} R2 {mt['R2']:.4f} | fit {fit_s:.2f}s", flush=True)

    res = pd.DataFrame(rows); res.to_csv(C.RES_DIR / "model_metrics.csv", index=False)
    # reference baselines (NOT machine-learning models; context only)
    base = {"Persistence (volume_t)": Xte["volume_t"].values,
            "Seasonal naive (same hour last week)": Xte["volume_lag167"].values}
    brows = [{"baseline": k, **metrics(yte, v)} for k, v in base.items()]
    pd.DataFrame(brows).to_csv(C.RES_DIR / "baseline_metrics.csv", index=False)

    # save predictions for val and test (dashboard + evaluation)
    def frame(Xs, ys, ms, preds):
        o = pd.DataFrame({"time": ms.time.values, "target_time": ms.target_time.values,
                          "actual": ys.values, "volume_t": ms.volume_t.values})
        for k, v in preds.items(): o[k] = v
        for k, v in base.items():
            if Xs is Xte: o[k] = v
        return o
    frame(Xva, yva, mva, pred_val).to_csv(C.RES_DIR / "predictions_val.csv", index=False)
    frame(Xte, yte, mte, pred_test).to_csv(C.RES_DIR / "predictions_test.csv", index=False)
    (C.RES_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    split_info = {k: {"rows": len(v[0]), "from": str(v[2].time.min()), "to": str(v[2].time.max())}
                  for k, v in sp.items()}
    split_info["n_features"] = X.shape[1]; split_info["features"] = list(X.columns)
    (C.RES_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2))
    return {"results": res, "split": split_info}


if __name__ == "__main__":
    train_all()
