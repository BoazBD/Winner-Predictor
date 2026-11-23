# Pre-registration (written 2026-11-10, BEFORE any look at kickoff >= 2026-03-01)

Final test period: kickoff 2026-03-01 .. 2026-11-07. Never used for exploration, feature design,
model fitting or threshold choice in this work (all of that used kickoff < 2026-03-01 only).

Model: `model.py` (level = pooled logistic on odds-ratio de-margined, rounding-corrected probabilities,
fit on closing prices; drift = LightGBM on calibrated closing-line value minus current value).
Fit ONCE on kickoff < 2026-03-01, no refit during the test.

Bet rule: at each time any market of a match changes, score every outcome present in that snapshot;
bet 1 unit the first time EV crosses the threshold, once per (market, outcome).

Primary policy   : ev > -0.03          (DEV walk-forward: 555 bets, ROI +0.5%, 90% CI -2.8..+3.7)
Secondary        : ev > -0.02          (DEV: 266 bets, +3.8%)
Ablation         : ev_now > -0.04      (level only, no drift model; DEV: 542 bets, -2.1%)
References       : flat betting every outcome at close; user's rule (fav margin-free p>=0.70,
                   overround<=1.16, >=10 changes so far) - note that rule was chosen by its authors with this period visible.
Expectation stated in advance: true ROI of primary about -2% (model's own closing-line estimate -2.3%).
