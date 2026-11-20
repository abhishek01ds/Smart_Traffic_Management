"""One-command reproduction of every result in the project.

    python run_pipeline.py              # all phases
    python run_pipeline.py --phase eda  # a single phase
    phases: eda, train, evaluate, explain, calibrate, simulate

Training + tuning takes a few minutes (single CPU); simulation ~2-4 minutes.
"""
import argparse, json, time
from src import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "eda", "train", "evaluate", "explain", "calibrate", "simulate"])
    ap.add_argument("--seeds", type=int, default=10, help="replicates per stress scenario")
    a = ap.parse_args()
    for d in (C.FIG_DIR, C.MODEL_DIR, C.RES_DIR): d.mkdir(parents=True, exist_ok=True)
    run = lambda p: a.phase in ("all", p)
    t0 = time.time()
    if run("eda"):
        from src.eda import run_eda; print("[EDA]"); run_eda()
    if run("train"):
        from src.train_models import train_all; print("[TRAIN]"); train_all()
    if run("evaluate"):
        from src.evaluate import run_evaluation; print("[EVALUATE]")
        print(json.dumps(run_evaluation()["selection"], indent=2))
    if run("explain"):
        from src.explain import run_explainability
        best = json.loads((C.RES_DIR / "best_model.json").read_text())["selection"]["selected"]
        print("[EXPLAIN]", best); run_explainability(best)
    if run("calibrate"):
        from src.experiments import calibrate_cycle_high; print("[CALIBRATE on validation days]")
        print(calibrate_cycle_high()); print(calibrate_cycle_high(candidates=(60, 75, 90, 105, 120), key="cycle_low_s", band="Low"))
    if run("simulate"):
        from src.experiments import run_all; print("[SIMULATE]"); run_all(n_seeds=a.seeds)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
