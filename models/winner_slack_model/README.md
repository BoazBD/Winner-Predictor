# Winner "rounding slack" model (soccer 1X2)

Bets the 1X2 favourite when Winner's own price rounding leaves it slightly over-priced:
- game is in Winner's lowest-margin tier (overround ~1.111 = 90% payout: top-5 leagues, Champions League, Israeli league)
- outcome is the favourite, odds <= 1.45
- rounding slack rr = odds / (price implied by the 1.111 margin) - 1 >= 0.01
- bet once per (market, outcome), the first time the flag turns on.

Track record (flat 1-unit stakes): development Dec 2023 - Jun 2026: -1.1% ROI over 955 bets (90% CI -4.1%..+1.9%);
sealed test Jul - Nov 2026 (chosen without seeing results): +5.4% over 136 bets (90% CI -1.8%..+12.8%).
Realistic expectation: around break-even to slightly negative. About 5-6 bets per week.

Files: slack_api.py (in-process API), predict.py (CLI), fable_core.py (features + rule), train.py (retrain),
model/ (trained on all data to 2026-11-07).

Training, slack_api.py and predict.py read the de-duplicated odds history (`../odds_history.py`, shared by every
model): one row per price segment, plus each market's latest snapshot when scoring. The flag depends only on prices,
so its first trigger is the same as on every raw snapshot.

Requires Python 3.10+, numpy, pandas, pyarrow, duckdb, scikit-learn (same major version as training: 1.x), joblib.
Retrain: python train.py all_processed_winner.parquet model
