# sweep_v1: the "Panic catcher" (paper trading)

Concise context for anyone (human or AI agent) working on this system. **Paper only: no real orders are ever sent.**
It went live on 2026-11-22.

## The idea
- Polymarket soccer prices before kickoff are efficient. Research that tried to predict them (news, wallets, sharp bookmakers, Kalshi, LLM agents, related markets) all lost money.
- The one edge found is other traders' **impatience**. Order books are thin, so when someone fires a big market order, it sweeps through every resting order. The price crashes far below fair value for a moment, then snaps back within minutes (about 90% of the time).
- We keep cheap resting **buy** orders deep below the price, wait to be filled by such a sweep, and sell back at the pre-sweep price.
- **Winner's fair price** is the safety check: we only bid where Winner says the price is well above our bid, so a fill means panic, not news. Without this check it loses money.

## Rules (constants at the top of `sweep.py`)
| Rule | Value |
|---|---|
| Games | every Winner game matched to Polymarket, from **3 h to 10 min** before kickoff |
| Tokens | every outcome on **both sides**: buy "Yes" catches dumps, buy "No" catches spikes. Price c (Polymarket midpoint) must be 8–92¢ |
| Bids (2 per token) | **A:** L = floor(min(c − 15¢, 0.60·c)); **B:** L = floor(min(c − 20¢, 0.50·c)); L ≥ 2¢. 500 shares each (`SWEEP_SHARES`) |
| Winner filter | Winner fair probability of that token ≥ L + 6¢ |
| Refresh | every 5 min: last run's orders are checked, then new orders are placed at fresh prices |
| Fill | only if a real **maker** print on our token traded **strictly below L** while the order was live. Fill size = shares below L in that transaction, capped at 500. Fill price = L, no fee |
| Sell | limit sell at **T = floor(c)** (the pre-sweep price). It sells only as many shares as real buyers traded **strictly above T** (shared across positions on the same token and T) |
| Leftovers | at kick − 5 min, sell 1¢ under the last real bid-hit print of the final 2 h, paying the taker fee 0.05·p(1−p). If there was no such print, hold to the result and settle from Polymarket's resolution |
| Cooldown | 60 min per (game, outcome, side, level) after a fill |

**Owner decisions:**
- Keep small, quiet games. A tested filter, "game traded ≥ $500 in the previous 6 h", would cut fills held through the game from 14% to 3% for about −8% profit. The owner **declined** it.
- Real money is **not** planned until 2–3 months of paper results.

## Expected results (backtest on 16k games, real trades, Sep 2025 – Nov 18 2026)
2 × 500 shares per token:

| Period | Fills/month | Return per $ spent | $/month |
|---|---|---|---|
| Train (to Jun 2026) | 6.8 | +72% [+50, +87] | $221 |
| Validation (Jul–Sep) | 5.7 | +69% | $212 |
| Test (Oct 1 – Nov 18) | 3.1 | +100% (2 games) | $149 |

- **Exits:** 37% of fills sell fully at target; 14% sell partly at target and the rest before kickoff; 34% are all sold before kickoff (still about +57%); **14% hold some shares through the game** (about 4% of the money, about −$75 in total).
- **Money in resting orders:** about $4–8k on average, more on busy weekends.
- **Fills are rare and lumpy** (top fills make most of the $). Judge after 2–3 months; zero fills in a day is normal.
- **Clean live test:** the hold-out is kickoffs after 2026-11-18. All earlier data has been used for research.

## Architecture
- **`predict_winner_model` Lambda** (il-central-1, every 3 h at :05 UTC; `lambda_function.py`): `write_sweep_universe()` writes Firestore `meta/sweep_universe`.
  - Contents: matched games kicking off in the next 12 h, each with `uid`, names, `kick_ts` (UTC), `poly_slug`/`poly_title`, and per Winner outcome `o`: `cond` (conditionId), `token` (Yes token), `wf` (Winner margin-free probability) and `fee_rate`.
  - Tests showed Winner data up to 3 h old is fine.
- **`sweep_catch` Lambda** (il-central-1, same Docker image, handler `sweep.lambda_handler`, EventBridge rule `sweep_catch-5min` = `rate(5 minutes)`, 240 s timeout, 1 GB; `sweep.py`): each run takes about 5 s.
  - It skips new orders if the universe is more than 6.5 h old.
  - **Firestore it writes:**
    - `meta/sweep_state`: `orders` of the last run and `cool` (cooldowns).
    - `meta/sweep_status`: last run summary, the resting orders now and the `locked_usd`; read by the site.
    - `sweep_positions/{uid}_{o}_{s}_{lvl}_{fill_ts}`: one document per fill. Fields include `c, L, T, q, through_shares, q_exit, status` (`open` / `held` / `closed`), `how` (`target`, `liquidated`, `target+liquidated`, `held to result`), `pnl`, `stake`, `fill_ts` and `kick_ts`.
- **Site** (Flask on Cloud Run, `site/`): route `/sweep`, tab "Panic catcher" (Hebrew "תופס פאניקה"), template `templates/sweep.html`.
  - Data: `db.get_sweep()` through `cache.py`, refreshed about every 30 min.
  - URL: https://winner-site-dypbatdkoq-uc.a.run.app/sweep

## Polymarket APIs used (public, no key)
- **Midpoints:** `POST https://clob.polymarket.com/midpoints` with `[{"token_id": …}]`.
- **Trades:** `GET https://data-api.polymarket.com/v2/trades?condition_id=<c1,c2,…>&taker_only=false&limit=1000&cursor=…`
  - Newest first; time filters are ignored, so page back until older than needed.
  - In each `transaction_hash`, the **first row is the taker** and the rest are **makers**, whose prices are the exact book levels hit.
  - `outcome_index` 0 = Yes; Yes price = price, or 1 − price for No rows.
  - A maker ended up longer Yes if it BUY Yes or SELL No.
- **Resolution:** `GET https://gamma-api.polymarket.com/markets?condition_ids=…&closed=true` → `outcomePrices`.

## Operate
- **Deploy** (both Lambdas, one image): `lambdas/predict_winner_model/deploy_docker_lambda.sh` (needs Docker running: `open -a Docker`).
- **Site:** `site/deploy_gcp.sh` (needs `gcloud auth login`).
- **Logs:** `aws logs filter-log-events --region il-central-1 --log-group-name /aws/lambda/sweep_catch --filter-pattern '"result"'`. Each run logs `fills, exits, liquidated, held, settled, orders, games_quoted, errors`.
- **Local dry run:** `GOOGLE_APPLICATION_CREDENTIALS=service-account.json python -c "import sweep; print(sweep.run(dry=True))"` (no writes).
- **Never commit or share `service-account.json`.**

## Research trail
- **Reports:** `research/ideas/sweep_catch/REPORT.md` (and `.html`).
  - Original POC: `s01`–`s08`.
  - Production tuning: `s10_detail.py` (replay of all candidates, with many settings) and `s11_eval.py` (rule evaluator; `fills(spec)`, `table()`).
  - The final fills are in `out/s11_final_fills.parquet`.
- **Worked example of a real panic sale and exit:** `research/ideas/sweep_catch/example/panic_example.html` (Pereira vs América de Cali, 2026-11-14).
- **Comparison with the other five ideas:** `research/ideas/SUMMARY.html`.
- **Pitfall:** grid and trade prices must be rounded to 4 decimal places before fill checks. A float32 artefact once faked an edge.

## Going to real money (not built)
You would need:
- Polymarket access in your country, plus a USDC wallet;
- the `py-clob-client` library, with the key in AWS Secrets Manager;
- a live (websocket) fill feed, so sells are posted within seconds;
- cancel and replace every 5 min;
- spending caps, a kill switch and reconciliation with Polymarket's records.

Open orders lock their full cost. The main live risk is other bots quoting the same deep levels.
