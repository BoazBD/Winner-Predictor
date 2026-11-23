# Idea 3: Polymarket vs Kalshi: no-go

*Research only. Saved from the agent's final message, 2026-11-21.*

## Verdict
Kalshi and Polymarket soccer prices track each other closely. The median gap is 0.5¢, and a hedged basket bought on both venues costs a median of $1.034 per $1 after both venues' fees.

- **Arbitrage:** real gaps appear in about 3% of games, last about a minute (mostly around lineup news) and are gone within 5–15 minutes.
  - A program that reacts within 5 minutes would make about $38 a month; an hourly one about $0.
  - A live snapshot of 266 upcoming games found no gap that could actually be traded.
- **Lead/lag:** trading Polymarket when Kalshi's price differs loses about −2.7¢ to −3.3¢ a share in every period.
- **Access:** Kalshi is for US residents only, and international Polymarket bars US persons, so one person can't hold both sides.

## Data
- **Coverage:** Kalshi's public API has about 180 soccer 1X2 series, covering 23,725 events.
- **Matching:** 10,912 games were matched to Polymarket, and the settled results agree in 99.9% of them.
- **Price history:** 1-minute price history was downloaded for a random 3,009 of the matched games.
- **Fees:** Kalshi takers pay 0.07·p·(1−p). Makers in the big-league series also pay 0.0175·p·(1−p).

## Risks
- **Settlement rules differ.** Kalshi settles cancelled or rescheduled games at a "fair price"; Polymarket resolves every outcome to No. That happened in 9 games (0.08%), and it breaks the hedge.
- **Unknown depth.** Kalshi's history has no order-book depth, and we would be racing other traders for the same prints.

## Files
- Scripts `s01`–`s07`.
- Results in `out/`: `summary.json`, `arb_events.csv`, `leadlag_grid.csv`, `live_snapshots.csv`.
