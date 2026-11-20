"""Phase 5/8 - explainability: permutation importance, model-native importance,
SHAP (if installed) and a per-prediction local explanation."""
from __future__ import annotations
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from . import config as C
from .data_loader import load_clean
from .features import build_dataset, chronological_split
from .train_models import SLUG

sns.set_theme(style="whitegrid")


def _final_estimator(model):
    return model.named_steps["model"] if hasattr(model, "named_steps") else model


def local_contributions(model, x_row: pd.DataFrame, background: pd.DataFrame, top=6):
    """Top feature contributions for ONE prediction.
    Linear model -> exact coef*(x - mean) in scaled space.
    Tree models  -> SHAP TreeExplainer if available, else permutation-style
                    'replace by background mean' effect (model-agnostic)."""
    est = _final_estimator(model)
    cols = list(x_row.columns)
    if hasattr(model, "named_steps"):
        sc = model.named_steps["scale"]
        z = sc.transform(x_row)[0]
        contrib = est.coef_ * z                         # relative to the mean row
    else:
        try:
            import shap
            sv = shap.TreeExplainer(est).shap_values(x_row)
            contrib = np.asarray(sv).reshape(-1)
        except Exception:
            base = model.predict(x_row)[0]
            contrib = []
            mean_row = background.mean()
            for c in cols:
                tmp = x_row.copy(); tmp[c] = mean_row[c]
                contrib.append(base - model.predict(tmp)[0])
            contrib = np.array(contrib)
    order = np.argsort(-np.abs(contrib))[:top]
    return [(cols[i], float(contrib[i]), float(x_row.iloc[0, i])) for i in order]


def run_explainability(best_name: str) -> dict:
    X, y, meta = build_dataset(load_clean())
    sp = chronological_split(X, y, meta)
    Xte, yte = sp["test"][0], sp["test"][1]
    Xtr = pd.concat([sp["train"][0], sp["val"][0]])
    out = {}
    imp_frames = []
    for name, slug in SLUG.items():
        m = joblib.load(C.MODEL_DIR / f"{slug}.joblib")
        pi = permutation_importance(m, Xte, yte, n_repeats=3, random_state=C.RANDOM_STATE,
                                    scoring="neg_root_mean_squared_error")
        d = pd.DataFrame({"feature": Xte.columns, "perm_importance_rmse": pi.importances_mean,
                          "std": pi.importances_std, "model": name})
        imp_frames.append(d)
    imp = pd.concat(imp_frames)
    imp.to_csv(C.RES_DIR / "permutation_importance.csv", index=False)

    top = (imp[imp.model == best_name].sort_values("perm_importance_rmse", ascending=False).head(15))
    plt.figure(figsize=(8, 5)); sns.barplot(data=top, y="feature", x="perm_importance_rmse", color="teal")
    plt.title(f"Permutation importance on TEST - {best_name}\n(RMSE increase when the feature is shuffled)")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "permutation_importance_best.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, name in zip(ax, SLUG):
        t = imp[imp.model == name].sort_values("perm_importance_rmse", ascending=False).head(8)
        sns.barplot(data=t, y="feature", x="perm_importance_rmse", ax=a, color="steelblue"); a.set_title(name)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "permutation_importance_all.png", dpi=130); plt.close()

    # native importance
    rf = joblib.load(C.MODEL_DIR / "random_forest.joblib")
    gb = joblib.load(C.MODEL_DIR / "gradient_boosting.joblib")
    lr = joblib.load(C.MODEL_DIR / "linear_regression.joblib")
    nat = pd.DataFrame({"feature": Xte.columns, "RF_impurity": rf.feature_importances_,
                        "GB_impurity": gb.feature_importances_,
                        "LR_abs_std_coef": np.abs(lr.named_steps["model"].coef_)})
    nat.to_csv(C.RES_DIR / "native_importance.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, col in zip(ax, ["LR_abs_std_coef", "RF_impurity", "GB_impurity"]):
        t = nat.sort_values(col, ascending=False).head(10)
        sns.barplot(data=t, y="feature", x=col, ax=a, color="darkorange"); a.set_title(col)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "native_importance.png", dpi=130); plt.close()

    # SHAP summary for the selected model (best effort)
    shap_ok = False
    try:
        import shap
        best = joblib.load(C.MODEL_DIR / "best_model.joblib")
        est = _final_estimator(best)
        sample = Xte.sample(min(600, len(Xte)), random_state=C.RANDOM_STATE)
        if hasattr(best, "named_steps"):
            Z = best.named_steps["scale"].transform(sample)
            sv = shap.LinearExplainer(est, best.named_steps["scale"].transform(Xtr)).shap_values(Z)
        else:
            sv = shap.TreeExplainer(est).shap_values(sample)
        plt.figure(); shap.summary_plot(sv, sample, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(C.FIG_DIR / "shap_summary_best.png", dpi=130, bbox_inches="tight"); plt.close()
        shap_ok = True
    except Exception as exc:                              # SHAP is optional
        print("SHAP skipped:", exc)
    out = {"shap_summary_created": shap_ok,
           "top_features_best": top.head(8)[["feature", "perm_importance_rmse"]].to_dict("records")}
    (C.RES_DIR / "explainability.json").write_text(json.dumps(out, indent=2))
    return out
