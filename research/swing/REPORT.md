# Swing model (buy, then sell before kickoff): result

*Research only, 2026-11-20. 14,954 Polymarket soccer games, 7.5M simulated orders, fills checked against 5.2M real trades.*

## Verdict: no edge. Not deployed.

The model makes one decision every 30 minutes, from 48 hours to 30 minutes before kickoff, for each outcome and both directions. The order:
- buys with a limit order at the midpoint rounded down (or 1–2¢ lower);
- then rests a sell order 1, 2, 3 or 5¢ higher;
- otherwise sells at the bid 5 minutes before kickoff.

A LightGBM model learned when to place the order from 33 features: the Winner–Polymarket gap, recent moves, trade flow, time to kickoff and others.

| | Fills | Return per $ [95% CI] |
|---|---|---|
| Every order, no model (train) | – | −3.2% to −5.1% depending on settings |
| Best model setting chosen on Apr–Jun | 581 | −1.0% [−1.4, −0.5] |
| Jul–Sep | 462 | −1.9% [−2.6, −1.3] |
| Oct–Nov 18 | 211 | −1.9% [−2.7, −1.1] |

The higher the model's confidence threshold, the worse the results got. Delayed prices, 3-hour-old Winner odds and a 200-share minimum fill size gave almost no trades and negative results.

## Why an earlier run looked profitable (+1.2% / +1.4%)

That run was a precision bug. The midpoint grid (`lead_lag/data/grid.parquet`) stores prices as float32, so a buy price of 0.86 became 0.8600000143. A real sale at exactly 0.86 then counted as trading through our price. That turned uncertain at-our-price fills, where we'd only get filled if we were first in the queue, into certain ones. With exact prices the edge disappears.

**Fix:** cast to float64 and round to 4 decimals; compare with a 5e-5 tolerance. The Longshot seller research used float64 prices and was re-checked; its results are identical.

## Files
- `s01_build.py`: simulated orders. Environment variables `DELAY`, `QMIN` and `OUT` select the stress tests.
- `s02_model.py`: training, threshold choice on Apr–Jun, then validation and test.
- `s03_robust.py`: stress tests.
- Logs are in `logs_*.txt`; the full numbers are in `out/results.json` and `out/robust.json`.
