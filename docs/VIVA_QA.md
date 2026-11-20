# Viva preparation - questions and answers
(Numbers refer to the final run; see `outputs/results/`.)

## Data and leakage
**Q1. Why did you not use `congestion_level` as a feature?**
It is a pure threshold of `traffic_volume` (Low <2500, Medium 2500-4500, High >4500). The target is next-hour volume, and current-hour congestion is a function of current volume, so it adds nothing legitimate and in a classification framing would leak the answer. I kept it only for display and for evaluation bands.

**Q2. What did you find about the faulty sensors?**
377 rows (2.15%). In the raw file their queue values are blank; the cleaned file filled them with numbers. Those numbers are imputed, not measured, so I masked them, forward-filled for at most 3 hours using only past values, and added a `sensor_faulty` flag. The decision engine also refuses to trust queues when the flag is set.

**Q3. How did you prevent data leakage?**
Features use only information known at the end of hour t (volume up to t, weather at t, masked queues at t). Calendar features describe the target hour because the calendar is known in advance. The target is volume at t+1. The split is chronological with a 1-hour gap; hyper-parameters are tuned on validation only; the test set is used once. A unit test checks that banned columns are absent and that the target is the next hour.

**Q4. Why not shuffle for cross-validation?**
Shuffling lets the model train on hours after the ones it is tested on and on neighbours of test points, which inflates scores. Time order must be respected.

## Models and metrics
**Q5. Why these three models?**
Linear Regression = interpretable baseline; Random Forest = bagged deep trees for non-linearity and interactions with low variance; Gradient Boosting = sequential residual fitting, often strongest on tabular data.

**Q6. Which model won and why?**
Random Forest: validation RMSE 225.2 (GB 238.7, LR 307.2); test MAE 164.4, RMSE 217.7, R2 0.978. Selection was made on validation under a rule fixed in advance and confirmed on test. Paired bootstrap: RF better than LR by ~72 RMSE (CI -84 to -61) and than GB by ~11 (CI -15 to -7).

**Q7. Is the improvement over Gradient Boosting practically meaningful?**
Statistically reliable but small (~5% RMSE, ~9 veh/h). A more heavily tuned GB might close the gap, so I would not claim RF is fundamentally better. Over Linear Regression the gain (~25% RMSE) is meaningful.

**Q8. Explain MAE, RMSE, R2 and why all three.**
MAE = average absolute error in vehicles/hour (robust, easy to read). RMSE squares errors, so it punishes large misses, which matter for peak hours. R2 = share of variance explained (0.978). Together they show typical error, tail error and overall fit.

**Q9. The seasonal-naive baseline already gets R2 0.944. Why is ML worth it?**
It cuts RMSE from 349 to 218 (-38%), because it combines the weekly pattern with recent volume, hour of day, holiday and weather. It is still honest to say much of the signal is a weekly repetition.

**Q10. What drives the predictions?**
Permutation importance on test (RMSE increase when shuffled): same hour last week (872), current volume (301), hour of day (108). Queues and weather matter little for volume. SHAP gives the same ranking and per-prediction explanations.

**Q11. Why can't a tree model extrapolate, and does it matter?**
Trees predict averages of training targets, so they cannot exceed the observed range (max 6,040). For unusual surges the forecast would saturate; I would monitor residuals.

## Decision engine and optimisation
**Q12. Why rules instead of learning the decisions?**
There is no ground-truth "correct decision" label in the data, and emergency/safety behaviour must be auditable and predictable. Rules guarantee behaviour for rare, critical events; the ML model supplies the forecast and the optimiser supplies the allocation.

**Q13. How are conflicting rules resolved?**
Order: emergency > safety constraints (clearance, min/max green) > incident/weather limits > optimisation. E.g., ambulance + incident: emergency green is given, and other approaches keep min green.

**Q14. How is green time computed?**
Demand = queue + expected arrivals; green proportional to demand (Webster-style), constrained to 10-60 s each, summing to cycle minus 4 x clearance, solved by water-filling. Each approach has its own phase, so no conflicting movements are ever green together.

**Q15. What if the sensor is faulty but an ambulance flag is on?**
Priority is still granted (a missed ambulance is worse than a needless preemption) and the operator is told to verify visually.

## Simulation and real-world applicability
**Q16. Can you claim the system would reduce delay by 50% on a real road?**
No. It is a simplified simulation with assumed arrival rates, a demand scale of 0.4, fixed saturation flow and no turns or pedestrians. It shows the *relative* behaviour of the strategies under identical assumptions.

**Q17. Is the fixed-time baseline fair?**
It is weak: equal green although North carries ~36% of traffic. A demand-tuned fixed plan would narrow the gap. I report this limitation, plus a sensitivity analysis: at lower demand (scale 0.25) the strategies are close (42.5 vs 40.2 s); at 0.4-0.5 fixed-time collapses.

**Q18. Does the ML forecast actually help the controller?**
Ablations say mostly no: full adaptive 41.2 s, using current volume instead of a forecast 41.0 s, using the true future volume 41.3 s. The benefit comes from adapting the cycle length and the queue-based split. The forecast would matter more where actions need lead time (e.g., pre-emptive coordination).

**Q19. What did calibration change?**
I first set a longer cycle (150 s) for predicted High demand. On validation days it was worse (High-hour delay 61.2 s vs 53.2 s at 120 s), so I removed it. Low-demand cycle was set to 60 s (25.9 s vs 39.4 s at 120 s in Low hours). These were chosen on validation days, not the test days.

**Q20. What are the ethical concerns?**
Emergency preemption delays other drivers; false detections cause disruption; sensor failure and incidents need human oversight; minimum green protects minor approaches from starvation; real deployment needs certification, field trials and fail-safe hardware.

**Q21. What would you do next?**
Per-direction detector counts, multi-intersection coordination, model-predictive or reinforcement-learning control, prediction intervals, camera-based ambulance detection, and comparison with Webster or actuated control.
