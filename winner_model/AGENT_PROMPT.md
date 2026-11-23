# Task: build the best odds-movement model for Winner soccer 1X2

You are a senior data scientist. Build a model that reads how a soccer match's 1X2 odds at Winner (an Israeli
bookmaker) have moved up to the current moment, and outputs the probability of home win, draw and away win.
From those probabilities and the odds on offer right now, it must say whether any outcome is worth betting.
The model will be queried in real time on live odds, so everything it uses must be known at decision time.

You decide everything else: which games to train on (all soccer, big leagues only, a margin tier, a time window
before kickoff), how to clean and compress the price history, which model family, which betting rule.
Run creative, careful experiments. What matters is honest out-of-sample profit, not a good-looking backtest.

## Input restrictions
- Features may come ONLY from the odds history (ratio1/2/3 over time) and timing (snapshot time, kickoff time,
  time to kickoff, time since the market opened, time since the last change).
- Do NOT use team names, league names or any identity as features. You may use the league only to choose
  which games the model is trained on or applied to, and you must say so if you do.
- Soccer 1X2 markets only.

## Data (in the repo root, /Users/boazbardavid/Desktop/Winner)
- `all_processed_winner_081126.parquet`: everything. 52.5M rows; soccer 1X2 has 122,789 matches.
- `processed_winner_081126.parquet`: a subset with 23 big competitions only (about 9,900 soccer 1X2 matches).
- Date range: 24 Dec 2023 to 7 Nov 2026. A snapshot of every open market roughly every 15 minutes, from when the
  market opens (median about 5 days before kickoff) until about 13 minutes before kickoff.

Columns (all strings unless noted; names are in Hebrew):
- `run_time`: snapshot time. Kickoff = `event_date` + `time`.
- `type` = 'Soccer'. The 1X2 market's `event` contains the substring `1X2` (it also has invisible
  right-to-left characters, so match by substring).
- `unique_id` = market id; for 1X2, `unique_id == result_id` (match id).
- `ratio1` = home, `ratio2` = draw, `ratio3` = away. Decimal odds on a 0.05 grid below 10.
- `bet1_won`, `tie_won`, `bet2_won` (bool) = which outcome won; `scorea`, `scoreb` = final score.
  These are the LABELS. Never use them, or any result of a match not finished at decision time, as a feature.

## Facts already established (verify, don't rediscover blindly)
- Winner prices 1X2 with a fixed payout per league: 90% (overround 1.111: top-5 leagues, Champions League,
  Israeli league), 85% (1.176: Europa/Conference League, national-team games, some cups) or 80% (1.25: almost
  everything else). When a price moves, the margin stays the same; it looks like a data feed plus a fixed margin.
- Because prices are rounded to 0.05, the overround deviates slightly from its tier. That deviation reveals
  whether a price was rounded up or down, and is worth about ±2–5% on short favourites.
- Flat betting at closing odds loses 15–25%. Strong favourite-longshot bias: odds ≤ 1.2 lose about 7%,
  odds 2–3 about 20%, odds above 6 lose 50–70%.
- Closing odds are more accurate than opening odds. In hindsight about 3% of minor-league outcomes are at some
  point priced more than 3% above their closing value, and betting those wins about +11%. Nobody has yet found a
  way to identify them in advance from Winner's own data.
- Models built on past results (Elo, Poisson, Dixon-Coles, etc.) and sequence models (GRU, TCN, transformer) beat
  the market's log-loss only slightly (about 0.001–0.003), mostly by correcting the favourite-longshot bias.
- Best honest results so far, all on soccer 1X2:
  - A value model (calibration + rounding + LightGBM predicting the line move): −0.6% ROI over 317 bets on an
    untouched test, about 9 bets a week at average odds 1.17.
  - A "rounding slack" rule (favourite, odds ≤ 1.45, 90% tier, rounding slack ≥ 1%): −1.1% over 955 bets in
    development, +5.4% over 136 bets on its test.
  Your target is to beat these. Even a model that loses 0–2% is a reasonable result; say so honestly if
  that is the ceiling.

## Traps that produced fake profits before
- Leakage: features only known at the end of the market (total number of price changes, closing price) or
  results of matches not yet finished.
- Placeability: when Winner moves a line it often withdraws the market. A backtest may only bet a market at a
  snapshot where it actually appears. Carrying old prices forward once produced a fake +5%.
- In-play rows: about 0.1% of rows are after kickoff; drop them.
- Multiple testing: with hundreds of variants, some show +10–20% on a few hundred bets by chance. At odds
  1.2–2.5 you need well over 1,000 bets to tell +2% from −2%. Report confidence intervals.
- Predicted EV has not ranked profit in the past. Check whether higher predicted EV actually earns more.

## Evaluation protocol (required)
- Walk forward by kickoff date. Never train on matches that kick off after the ones you test on.
- Do all exploration and selection on kickoff < 2026-03-01. Before you look at anything later, write down the
  final model, the betting rule and the thresholds in a file. Then run that once on kickoff 2026-03-01 to
  2026-11-07 and report it unchanged, good or bad. (Earlier work has already looked at that period, so treat it
  as a strong check, not a perfectly clean one.)
- Bet rule for backtests: 1 unit, the first time an outcome qualifies, once per (market, outcome), only at
  prices present in that snapshot.
- Report: ROI per bet with a 90% bootstrap CI clustered by match, number of bets, average odds, bets per week,
  ROI by period and by odds band, log-loss against margin-free market probabilities, and closing-line value
  (how often and by how much your price beat the final price). Compare against flat betting and the two
  models above.

## Deliverables (put everything in /Users/boazbardavid/Desktop/Winner/models/<your_model_name>/)
1. `predict_api.py` with a class that loads once and scores live data:
   ```python
   m = Model()                       # loads artifacts from ./model/
   out = m.score(df, latest=True)    # df = raw rows, schema below; strings are fine
   ```
   - Input: raw rows of the live Athena table `api_odds` with columns `event_date, time, league, event,
     option1, ratio1, option2, ratio2, option3, ratio3, run_time, result_id, unique_id, date_parsed, type`.
     It contains the full snapshot history of the currently open markets and many non-1X2 or non-soccer rows,
     which must be ignored. There are no result columns.
   - Output: one row per (market, outcome) for the latest snapshot: `unique_id, result_id, run_time, kick,
     outcome (0 home / 1 draw / 2 away), odds, p_model, ev (= p_model × odds − 1), flag`.
     `p_model` must sum to 1 across a match's three outcomes. `flag=True` means "bet this now" under your rule.
   - It must run on Linux in AWS Lambda (Python 3.12, 2 GB RAM, CPU only) in under a minute for about 300 open
     games with up to 3 weeks of history each. No GPU at inference.
2. `train.py` that retrains from the parquet files from scratch, plus the trained artifacts in `model/`.
3. `README.md`: features, cleaning, model, betting rule, exact dependency versions, and how to retrain.
4. `REPORT.md`: what you tried (including what failed), the development and test results in the format above,
   and your honest expectation of live ROI. If you believe beating the margin needs outside data (for example
   sharp-bookmaker odds at the same timestamps), say so and say exactly which data would help.

## Environment
Mac, 8 cores, 16 GB RAM, Apple GPU (MPS) available for training. A Python environment with pandas, polars,
duckdb, lightgbm, xgboost, torch and scikit-learn is at `/Users/boazbardavid/Desktop/Winner/winner_model/.venv`;
you may create your own. The raw 1X2 data is large, so compress price histories early (for example, keep only
snapshots where a price changed or the market reappeared, plus the gaps between them).
