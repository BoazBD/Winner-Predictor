# Idea 6: Related Polymarket markets for the same game: no-go

*Research only. Saved from the agent's final message, 2026-11-21.*

## Verdict
The side markets already sit about 1¢ from the fair price implied by the 1X2 market:
- totals;
- both teams to score (BTTS);
- spreads;
- exact score;
- halftime result.

The gaps don't close before kickoff, so they look like model error, not lag. True hedged locks are worth about $30 a month at most historically, and $0 in live scans. Betting on the gaps carries full game risk, and no rule chosen on train was significantly profitable on validation or test.

## Inventory
- **Where the markets live:** each game's side markets are separate events found by adding a suffix to the game's slug: `-more-markets`, `-exact-score`, `-halftime-result`, plus newer ones.
- **Coverage:** totals, BTTS and spreads are on about 95–100% of games from January 2026.
- **Liquidity:** side-market volume adds up to 57% of 1X2 volume, but most side markets are thin.

## Rules chosen on train (180 combinations tried)

| Rule | Train | Validation | Test |
|---|---|---|---|
| Taker, hold to result (θ = 3¢, last 6 hours) | +5.9% [−1.5, +13.3] | +7.9% [−5.7, +20.3] | −5.9% [−19.6, +9.6] |
| Taker, sell before kickoff (θ = 5¢) | +4.2% | −2.5% | −4.0% |
| Maker, hold (θ = 8¢, last 6 hours) | +13.1% | −8.8% | +7.0% |

- **Confidence ranges:** every range in the table includes zero.
- **Order size:** the median fill is only $10–16.
- **One possible lead (could be a data-mining result):** maker orders at θ = 5¢ over 72 hours returned +9.0% / +8.2% / +1.8%. They carry game risk.

## Live
- **Cadence:** these orders would need refreshing every 15 minutes, so an hourly program is too slow.
- **No LLM role:** market slugs follow fixed patterns, so finding the markets needs no AI.

## Files
- **Scripts:** `s00`–`s09`.
- **Outputs:** in `out/`, including `rv_grid.csv`, `hedge_summary_*.csv`, `live_scan_summary.csv` and the charts.
