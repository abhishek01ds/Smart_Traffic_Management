# From Traffic Data to Intelligent Decisions
**AI-Powered Smart Traffic Management and Adaptive Signal Decision System**

A decision-support prototype: it predicts next-hour traffic volume with three ML models, picks the best one
from evidence, applies six IF-THEN expert rules, allocates green time per direction under safety constraints,
tests the result against a fixed-time plan in a simplified simulation, and explains every decision in a Streamlit dashboard.

> **Honesty note.** Simulation numbers are *simulated estimates under stated assumptions*, not measurements
> from the CSV and not a claim about real-world traffic control.

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py            # re-creates every result (a few minutes; ~5 min on 1 CPU)
streamlit run dashboard/app.py    # 5-page dashboard
python -m pytest -q tests         # unit tests
```
Pre-computed models, figures and result tables are already in `outputs/`, so the dashboard runs immediately.
Phases can be run singly: `python run_pipeline.py --phase eda|train|evaluate|explain|calibrate|simulate`.

## Structure
| File | Phase / purpose |
|---|---|
| `src/config.py` | all paths, thresholds, signal constraints, simulation assumptions |
| `src/data_loader.py` | safe loading, data-quality report (raw vs cleaned) |
| `src/eda.py` | exploratory analysis, faulty-sensor and leakage investigation |
| `src/features.py` | next-hour target, lags, rolling stats, calendar features, chronological split |
| `src/train_models.py` | the 3 models: tuning on validation, refit on train+val, timing |
| `src/evaluate.py` | MAE/RMSE/R2, bootstrap significance, selection rule, figures |
| `src/explain.py` | permutation importance, native importance, SHAP, local explanations |
| `src/decision_engine.py` | Rules 1-6 + explanations |
| `src/signal_optimizer.py` | constrained proportional (water-filling) green allocation |
| `src/simulation.py`, `src/experiments.py` | intersection simulation, scenarios, ablations, calibration |
| `dashboard/app.py` | Streamlit dashboard (5 pages) |
| `docs/PROJECT_REPORT.md`, `docs/VIVA_QA.md` | academic documentation, viva preparation |

## Headline results (test period 2023-09-14 to 2023-12-31)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 221.1 | 289.3 | 0.9617 |
| **Random Forest (selected)** | **164.4** | **217.7** | **0.9783** |
| Gradient Boosting | 173.0 | 228.6 | 0.9761 |

Simulation (108 replayed test days): mean delay 83.9 s/veh (fixed-time) vs 41.2 s/veh (adaptive);
ambulance delay ~51 s vs ~2.5 s. See the report for caveats (weak equal-split baseline, assumed demand scale,
ML forecast adds little beyond reactive control).
