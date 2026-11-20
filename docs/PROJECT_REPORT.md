# AI-Powered Smart Traffic Management and Adaptive Signal Decision System
*Theme: "From Traffic Data to Intelligent Decisions"*

All numbers below are produced by `run_pipeline.py` (random seed 42) and stored in `outputs/results/`.

## 1. Abstract
We build a decision-support prototype for a four-way signalised intersection using 17,520 hourly records (2022-2023).
Three regressors (Linear Regression, Random Forest, Gradient Boosting) predict next-hour traffic volume with strictly
past-only features and a chronological split. Random Forest was selected on validation error (test MAE 164.4, RMSE 217.7,
R2 0.978). The prediction feeds a hybrid engine of six IF-THEN rules and a queue-proportional green-time optimiser with
minimum/maximum green, emergency override and fallback rules. In a simplified simulation replaying 108 test days, the adaptive
strategy reduced mean delay from 83.9 to 41.2 s/vehicle and ambulance delay from ~51 s to ~2.5 s versus fixed-time. The
simulation is an illustration under stated assumptions, not evidence of real-world performance.

## 2. Problem statement
Fixed-time signals ignore current queues, forecast demand, emergencies, sensor failures and weather. The task is to turn data into
*decisions*: which direction is most congested, how much traffic is coming, who needs emergency priority, how to split green time, and how to
behave when data or conditions are unreliable.

## 3. Objectives
1. Analyse congestion by direction and time. 2. Predict next-hour volume with three models and choose the best from evidence.
3. Implement six explainable expert rules. 4. Optimise green time for N/S/E/W under safety constraints.
5. Compare adaptive vs fixed-time control in a documented simulation. 6. Present everything in an interactive dashboard.

## 4. Existing vs proposed system
*Existing:* fixed-time plan, equal split, no emergency awareness. *Proposed:* forecast + current queues + rules + optimisation, with
fallbacks and a human-readable reason for every plan.

## 5. Dataset description
`traffic_cleaned.csv` (primary, 20 columns, 2022-01-01 to 2023-12-31, hourly, no gaps or duplicate timestamps) and
`Metro_Interstate_Traffic_Volume.csv` (reference, 17 columns). Findings from inspection:
* The raw file has **blank queue values in exactly the 377 faulty-sensor rows** (2.15%); the cleaned file contains *imputed* numbers there.
  Clean equals raw on all healthy rows; traffic volume is identical in both files.
* `congestion_level`/`congestion_code` are pure thresholds of volume (Low 200-2499, Medium 2500-4500, High 4501-6040): **target leakage**, excluded.
* `weather_description` duplicates `weather_main`. `holiday` is blank except 336 holiday hours.
* Incident and ambulance flags show no association with volume (Mann-Whitney p = 0.99 and 0.09): useful for decision rules, not for volume prediction.
* Congestion classes: Low 53.4%, Medium 33.1%, High 13.5%. Queue share: North 35.6%, South 26.1%, East 21.5%, West 16.8%.
  **North is the most congested direction.** Mean peak: 17:00, ~4,791 veh/h. Weekday mean 2,632 vs weekend 1,910.
* Autocorrelation of volume: lag 1 = 0.873, lag 24 = 0.842, lag 168 = 0.972 (strong weekly pattern).
* Ambulance events: 535 (East 150, North 136, South 126, West 123). Incidents: 324.

## 6. System architecture
Data -> cleaning/masking -> features -> 3 models -> selection -> **Decision engine** (rules 1-6) -> **Optimiser** (green times) ->
**Simulation** (fixed vs adaptive) -> **Dashboard** with explanations. Modules are listed in the README.

## 7. Preprocessing and feature engineering
Prediction time = end of hour *t*; target = volume at *t+1*.
* Faulty-sensor queues set to NaN, forward-filled causally (max 3 h), remainder filled with the median of healthy readings; `sensor_faulty` kept as a feature.
* Volume lags 1, 2, 3, 6, 12, 24, 48 plus lag 23 and 167 (the *target hour's* value one day / week earlier), rolling mean (3, 6, 24 h), rolling std (6, 24 h), 1-h difference.
* Calendar of the target hour (cyclic hour, day-of-week, weekend, month, holiday), weather at *t* (one-hot weather, temperature, rain, snow, clouds), masked queues and their total.
* 42 features. Rows without full history (first week) dropped.
* **Chronological split 70/15/15** with a 1-hour gap: train 12,146 rows (to 2023-05-29), validation 2,602, test 2,602 (2023-09-14 to 2023-12-31).
* Same features and same protocol for all three models: hyper-parameters tuned on validation only, refit on train+validation, test touched once.
* Caveat: holiday flags are calendar knowledge (known in advance).

## 8. The three models
* **Linear Regression (baseline).** Least-squares fit; assumes linearity, additive effects, independent errors; interpretable (standardised coefficients); cannot model interactions such as hour x weekday.
* **Random Forest.** Averages many decorrelated deep trees (bagging + random feature subsets); captures non-linearity and interactions; robust, little scaling needed; larger and slower to train than OLS; cannot extrapolate beyond training range. Selected parameters: 200 trees, depth 12, min leaf 2, max_features 0.5.
* **Gradient Boosting.** Trees added sequentially to fit residuals; strong on structured data; more sensitive to hyper-parameters and sequential to train. Selected: 300 trees, learning rate 0.1, depth 4, subsample 0.8.
Reference baselines (not ML models): persistence RMSE 744 / R2 0.746; seasonal-naive (same hour last week) RMSE 349 / R2 0.944.

## 9. Model comparison and selection
| Model | Val RMSE | Test MAE | Test RMSE | Test R2 | MAE % of mean volume | Train time (s) |
|---|---|---|---|---|---|---|
| Linear Regression | 307.2 | 221.1 | 289.3 | 0.9617 | 9.1% | 0.03 |
| **Random Forest** | **225.2** | **164.4** | **217.7** | **0.9783** | **6.8%** | 18.7 |
| Gradient Boosting | 238.7 | 173.0 | 228.6 | 0.9761 | 7.2% | 28.0 |

**Selection rule (fixed before looking at test):** lowest validation RMSE; a simpler model wins if within 2% of the best. Linear Regression was 36% worse on validation, so
Random Forest was selected; the test set **confirmed** it (lowest on all three metrics).
Paired day-block bootstrap of RMSE difference (RF - other): vs Linear Regression -71.7 (95% CI -84.1 to -60.8); vs Gradient Boosting -10.9 (CI -14.7 to -7.1). Both intervals exclude zero.
RF is best in all congestion bands (e.g. MAE in High hours: RF 169, GB 180, LR 260).
*Is the improvement meaningful?* Over Linear Regression yes (RMSE -25%). Over Gradient Boosting it is statistically reliable but small (~5% RMSE, ~9 veh/h MAE), and
a tuned GB could plausibly close part of it, so the choice between the two is weak. Balance: RF predicts in ~25 ms per 1,000 rows and trains in seconds, adequate for hourly decisions; interpretability is
provided by permutation importance/SHAP rather than coefficients. Most important features (RF permutation importance, RMSE increase): volume at the same hour last week (lag 167, 872), current volume (301), hour of day (108).
The strong weekly signal also explains why the seasonal-naive baseline is already good (R2 0.944).

## 10. Decision-making framework (IF-THEN rules)
Implemented in `src/decision_engine.py`; every decision returns triggered rules, alerts and a plain-language explanation.
1. **Ambulance priority:** IF ambulance detected AND direction known THEN emergency green (30 s) for that direction after the clearance interval, phase order starts with it. Unknown direction -> operator confirmation. Honoured even when the sensor is faulty (fail-safe; flagged for visual check).
2. **High queue:** IF the largest queue >= 5 and >= 1.5x the mean of the others THEN that direction gets a larger green share.
3. **Future congestion:** IF predicted volume >= 1.15x current (and >= 2,500) OR predicted High (>4,500) THEN plan for the rising demand using predicted arrivals; cycle shortened to 60 s when Low is predicted.
4. **Incident:** IF incident reported THEN alert, operator intervention, green capped at 45 s so no approach is starved.
5. **Faulty sensor:** IF sensor faulty THEN ignore queues, use historical direction shares (safe fallback), raise maintenance alert.
6. **Weather:** IF thunderstorm/snow/fog (or heavy rain) THEN +2 s clearance, cycle >= 120 s; operator review for severe cases.
Priority: emergency > safety constraints (clearance/min/max green) > incident/weather > optimisation.

## 11. Adaptive signal optimisation
Four phases (N, E, S, W), one approach per phase, so conflicting movements are never green together. Demand per direction = current queue + expected arrivals
(predicted volume x site scale x direction share). Green time is proportional to demand (Webster-style), solved by iterative water-filling under
min 10 s / max 60 s, total green = cycle - 4 x clearance (4 s). Emergency: fixed 30 s for the ambulance approach, others share the rest.
Cycle lengths (Low 60 s, Medium and High 120 s) were chosen on *validation* days (`calibration_*.csv`). My first design (longer 150 s cycle for High) was worse on validation (53.2 vs 61.2 s delay in High hours) and was dropped.

## 12. Traffic simulation methodology
Second-by-second queue dynamics per phase (reflected random walk). Arrivals Poisson with rate = dataset volume x 0.4 x direction share / 3600; direction share = 50% that hour's CSV queue share + 50% long-run share.
Discharge 1.0 veh/s per approach after 2 s start-up loss (0.85x in bad weather; 0.5x on a blocked approach). Delay = integral of queue (vehicle-seconds) -> mean delay per vehicle.
Ambulance: detected 15 s before the stop line; delay = wait for own green (+3 s yield if it arrives on red). All strategies share identical arrival streams.
Experiments: (a) replay of all 108 complete test days with the dataset's real ambulance/incident/fault/weather flags; (b) six controlled 8-hour evening scenarios x 8 random replicates; (c) ablations; (d) sensitivity to the demand scale.
**Assumptions to defend:** demand scale 0.4 (share of the dataset's corridor volume reaching this intersection; set so average peak demand is moderate versus capacity; 0.5 gave unrealistic ~1,000 s delays), two-lane saturation flow, no turning movements or pedestrians.

## 13. Results and discussion
**Replay of 108 test days (mean over days):**
| Strategy | Avg queue (veh) | Delay (s/veh) | Ambulance delay (s) |
|---|---|---|---|
| Fixed-time (26 s each, 120 s cycle) | 23.6 | 83.9 | 51.1 |
| **Adaptive (full)** | **11.2** | **41.2** | **2.5** |
Adaptive was better on all 108 days. By congestion band (s/veh): Low 42.6 -> 25.9, Medium 77.2 -> 43.8, High 152.4 -> 54.9.
**Stress scenarios (s/veh, fixed -> adaptive):** evening peak 97.5 -> 47.7; ambulance stress 97.5 -> 49.1 (ambulance delay 42.0 -> 1.8 s); sensor failure 97.5 -> 44.7; incident 167.2 -> 58.0; thunderstorm 333.4 -> 66.2; combined 218.5 -> 139.2.
**Ablations (what each part is worth):**
* Ambulance rule: ambulance delay 39.5 s without it vs 2.5 s with it, at essentially no cost in the replay and ~5% more delay in the combined-stress scenario (132.5 vs 139.2).
* Faulty-sensor rule: in the sensor-failure scenario, trusting garbage readings gives 66.0 s vs 44.7 s with the fallback.
* **ML forecast:** adaptive control with no forecast or no cycle change: 45.5-45.6 s; with the model forecast: 41.2 s; with *current volume* instead of a forecast: 41.0 s; with the *true* future volume (oracle): 41.3 s. So the gain comes from adapting the cycle length to demand, **not specifically from the ML forecast**, which adds essentially nothing over a reactive rule in this setup. This is an honest limitation; the forecast would matter more where actions need lead time.
**Caveats:** (i) the fixed-time baseline splits green equally although North carries ~36% of traffic, which flatters the adaptive result; a demand-tuned fixed plan would be a stronger baseline. (ii) At lower demand scales the gap narrows (0.25: 42.5 vs 40.2 s); at >=0.4 fixed-time degrades sharply (101.5 at 0.4, 475.9 at 0.5). (iii) Results depend on the assumed scale and saturation flow.

## 14. Limitations and ethical considerations
Single intersection; hourly data (decisions need seconds-level detection); per-direction volumes unavailable (shares inferred from queues); the dataset's queue/ambulance/incident columns appear synthetic, so conclusions about them are not field evidence;
simulation simplifications (no turns, pedestrians, gap-out). Emergency preemption affects other drivers; a false ambulance flag causes unnecessary disruption, and the fail-safe choice (honour it) is a value judgement.
Faulty sensors, incidents and severe weather request human review: the system supports operators, it does not replace them. Fairness: minimum green prevents starving minor approaches.
Safety-critical deployment would need certification, field validation and fail-safe hardware.

## 15. Future improvements
Per-direction detector counts; multi-intersection coordination; reinforcement-learning or model-predictive control; prediction intervals; real-time streaming; computer-vision ambulance detection; stronger baselines (Webster/actuated control); calibration on local data.

## 16. Conclusion
Data inspection exposed leakage and imputed values that a naive model would have trusted. A Random Forest predicts next-hour volume with R2 0.978 (MAE 6.8% of mean volume), and a transparent rule + optimisation layer turned that
information into explainable signal plans that outperform an equal-split fixed plan in a documented simulation. The honest takeaway: most of the benefit comes from responding to current conditions and handling emergencies and faults safely; the forecast itself contributes little in this setup.
