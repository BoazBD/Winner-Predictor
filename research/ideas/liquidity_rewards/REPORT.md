# Idea 1: Liquidity rewards and maker rebates on soccer: no-go

*Research only. Saved from the research agent's final message, 2026-11-21.*

## Verdict
This loses money.

- **Rewards:** only 18 of 1,905 soccer 1X2 markets pay rewards, all of them EPL markets at $100 per market per day, against a thick professional order book.
- **Maker rebates:** these start at $200k of 30-day weighted maker volume, so we would get 0%.
- **Trading costs:** filled orders, plus selling the leftover shares before kickoff, cost more than the rewards pay at every size tested.

## Best rule on train
- **Orders:** EPL only, one tick behind the best price, 1,000 shares on each side.
- **Re-quoting:** every 15 minutes.
- **Inventory cap:** 3,000 shares.
- **Window:** from 00:00 UTC on match day to 60 minutes before kickoff.
- **Exit:** cross any leftover shares at the stop.
- **Search:** 992 settings were tried.

| Period | EPL games | Net $ per game [95% CI] |
|---|---|---|
| Train | 207 | −0.93 [−2.49, +0.36] |
| Validation | 14 | −0.20 [−3.80, +2.42] |
| Test | 14 | −5.27 [−17.4, +0.80] |

- **Scaling up:** with 20,000 shares at the best price, rewards come to about $58 per game and trading loses about −$107 to −$118.
- **Other top leagues:** trading alone loses about −$2 per game.
- **Fill model:** the result depends on how fills are modelled. Simulated fills assumed 20,000 shares queued ahead of us, matching live EPL depth. A positive result needs fills with no queue at all, which isn't realistic.

## Live
- **Speed:** this would need a loop of about 1 minute; hourly is too slow.
- **LLM:** no role for one.

## Lead (untested)
High-reward Polymarket markets outside sports have thin competition (`out/reward_scan_all.csv`). Price moves there could be costly for us, because these markets react to news.

## Files
- Scripts: `s01`–`s05`.
- Results and charts: `out/`.
