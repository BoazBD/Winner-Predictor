# Optional hourly "watcher" (paper only, not built)

The research found no profitable rule (see REPORT.html), so there is nothing to trade. This optional job would only collect clean evidence on games after 2026-11-18.

**Rules to track:** the 3 least-bad rules, plus a no-skill control. Parameters are frozen exactly as in `out/s07_finalists.csv`.
- `copy_move_skill`: move score ≥ 0.5¢, bet ≥ $1k.
- `copy_ml`: predicted move ≥ 0.5¢.
- `copy_result_skill`: result score ≥ 2%, bet ≥ $5k.
- Control: `copy_all_big`, bet ≥ $20k.

All four use a limit buy at the midpoint rounded down and a +1¢ limit sell; anything still held is sold before kickoff.

**Bar for going live:** after 300 or more paper fills, a rule's 95% range is above 0, and it stays positive over the next 150 fills.

## Infrastructure
**Schedule:** an hourly Lambda triggered by EventBridge. Wallet scores and the ML model are rebuilt monthly into S3, using only earlier games.

**Each run:**
1. Fetch new trades, with wallets, for games kicking off within 72 hours, from `data-api /v2/trades?event_id=…&taker_only=false`. Keep a cursor per event.
2. Build bets and compute their features.
3. Create signals and paper limit orders.
4. Check open orders against real trades. A fill counts only if a trade went strictly through our price. The sell counts the same way; anything still held is closed at the bid shortly before kickoff, minus the fee.

**Bedrock:** not needed in the loop. For an optional weekly text summary, use `global.anthropic.claude-haiku-4-5-20251001-v1:0` in il-central-1.

**Firestore collections:**
- `wallet_paper_signals`
- `wallet_paper_trades`: status open / filled / expired / closed, plus fill and exit prices and return
- `wallet_paper_runs`
- `wallet_paper_summary`

The full field list is in the agent's draft; ask for it if this is built.
