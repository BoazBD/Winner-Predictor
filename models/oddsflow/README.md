# oddsflow — odds-movement model for Winner soccer 1X2

Reads the 1X2 price history of a match up to now, outputs P(home / draw / away) and flags an outcome when
`p_model × odds − 1 > 0`. Honest expectation: about break-even (see REPORT.md); it is a filter that removes the
bookmaker's margin, not a demonstrated profit.

## Use
```python
from predict_api import Model
m = Model()                      # loads ./model/ (model.json + trees.npz)
out = m.score(df, latest=True)   # df = raw api_odds rows, full history of the open markets, strings OK
bets = out[out.flag & ~out.flag_before]
```
Pass ALL rows (every sport and market): non-soccer / non-1X2 rows are ignored for scoring but are used to
recognise scraper runs (a market missing from a run = withdrawn). Output columns:
`unique_id, result_id, run_time, kick, outcome (0 home / 1 draw / 2 away), odds, p_model, ev, flag` plus
`tier, p_market, drift_hat, hours_to_kick, flag_before`. With `latest=True` only markets present in the newest
run are returned (others cannot be bet). `flag_before=True` means the rule already fired earlier in the supplied
history for that (market, outcome) — bet only once. Measured: 405 open games with 3 weeks of history (104k raw
rows) score in 1.8 s using 0.8 GB.

## Features (odds history and timing only — no team or league identity)
Per snapshot, per outcome (`features.py`, list `FEATS`):
- **Level**: odds, margin-free logit, overround, payout tier (inferred from the overround itself: ≈1.111 / 1.17 /
  1.25), favourite rank, the other two outcomes' logits.
- **Rounding**: prices sit on a 0.05 grid while the unrounded implied probabilities sum to the tier overround, so
  `overround − tier` is rounding error. It is allocated to the three prices in proportion to the squared width of
  each price's rounding interval, giving rounding-corrected ("latent") probabilities and a per-price slack.
- **Path**: change since open, last move and the move before it, change over the last 1/3/6/12/24/48 h, distance
  from the running min/max, up/down move counts, price vs its time-weighted average, total absolute movement.
- **Timing**: hours to kickoff, hours since open, since the last change, since this price (re)appeared, how long
  the previous price stood, changes per day, withdrawals so far and length of the last one, kickoff moved,
  kickoff hour and weekday, snapshot hour.

## Cleaning
Soccer rows whose `event` contains `1X2`; duplicates per (market, run) dropped; rows at or after kickoff dropped;
prices ≤ 1 dropped. History is compressed into segments: a new segment starts when a price changes, the kickoff
changes, or the market reappears after missing ≥ 1 scraper run. Training (`build_data.py`) and the live API use the
same function, `odds_history.soccer_1x2_history` (`../odds_history.py`, shared by every model).

## Model (`model_lib.py`)
1. **Drift**: LightGBM regression (3 seeds × 600 trees, 31 leaves) of `closing logit − current logit`
   (rounding-corrected), trained on every decision point of all 122,674 soccer 1X2 matches.
2. **Calibration**: logistic regression fitted on closing prices vs results (cubic in logit, tier intercepts and
   slopes, draw / away terms) — this is the favourite-longshot correction. Applied to `current logit + predicted
   drift`, then renormalised to sum to 1.
Trees are exported to numpy arrays, so inference needs no lightgbm.

## Betting rule
Bet 1 unit on an outcome the first time `ev = p_model × odds − 1 > 0` at a snapshot where the market is present;
once per (market, outcome). In practice this only fires on favourites (odds 1.05–1.45) in the 90%-payout tier,
about 4–6 times a week, mostly days before kickoff. The threshold is `ev_threshold` in `model/model.json`.

## Dependencies
Inference: Python 3.11–3.12, numpy, pandas ≥ 2.0, duckdb (the shared cleaning step `../odds_history.py`).
Tested on macOS / Python 3.11.3 only; the model itself is pure numpy (no compiled model, no pickle), so it should run
unchanged on Lambda Python 3.12, but that has not been run.
Training / backtests: + duckdb 2.0.0, lightgbm 4.7.0, scikit-learn 1.9.1, pyarrow 26.0.1 (Python 3.11.3).

## Retrain
```
python train.py                      # rebuilds work/ from ../../all_processed_winner_081126.parquet, writes model/  (~4 min)
python train.py --src NEW.parquet    # any file with the same schema
python backtest.py dev && python report.py dev        # walk-forward on kickoff < 2026-03-01  (~25 min)
python backtest.py final && python report.py final    # the pre-registered test (already run once; see REPORT.md)
```
The shipped `model/` is trained on everything up to 2026-11-07, so the next clean test is matches after that.

## Files
`predict_api.py` live API · `features.py` segmentation + features · `table.py` decision-point table ·
`model_lib.py` fit / predict / numpy trees · `build_data.py` raw → segments · `train.py` · `backtest.py` ·
`report.py` · `PREREGISTRATION.md` · `REPORT.md` · `exp/` exploration scripts and saved reports
(`exp/report_dev.txt`, `exp/report_final.txt`) · `work/` compact data and backtest outputs.
