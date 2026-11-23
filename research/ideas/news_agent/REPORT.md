# Idea 5: LLM news and lineup agent: no-go

*Research only. Saved from the research agent's final message, 2026-11-21.*

## Verdict
On 497 big games (Aug 4 – Nov 7, 2026), team news from free feeds barely moves Polymarket afterwards: +0.03¢ after 60 minutes and +0.6¢ by kickoff. Trading on Sonnet's calls loses **−6.1% per $ [−8.3, −3.8]**, whether the agent acts after 15 minutes, every hour, or 3 hours later.

The direction the LLM picks is slightly better than random (p≈0.01), but the edge is far below the 2–3¢ round-trip cost. Lineup news is priced in within 15–30 minutes, and following the market's first move after lineups also loses (−4% to −8%).

## Data sources
- **GDELT GKG 15-minute files:** these work, but they are slow and thin.
  - Articles appear a median of 107 minutes after publication.
  - Only 285 of 497 games had any team-news headline.
- **GDELT DOC API:** rate-limited and noisy.
- **Google News RSS:** every item carries the same fake timestamp, so it can't be used.
- **Reddit:** returns 403 without authentication.
- **Lineups:** there is no free historical source of lineups with publish times.

## Agent tested
1. Haiku 4.5 screens each headline.
2. Sonnet 4.6 judges the price impact of the material headlines.
3. The agent trades when Sonnet expects a move of at least 1.5 points, buying as a taker and selling before kickoff.

| Signal | Trades | Return per $ [95% CI] |
|---|---|---|
| Sonnet ≥ 1.5, at the news | 193 | −6.1% [−8.3, −3.8] |
| Sonnet ≥ 1.5, hourly cron | 190 | −6.1% [−8.4, −3.5] |
| Haiku only | 594 | −7.0% [−9.1, −5.0] |
| Random-side placebo | – | −7.3% |

**Placebo checks:**
- Hindsight: swapping the home and away labels flipped Haiku's answer 91% of the time, so it reads the headline rather than remembering results.
- LLM cost is about $1–2 a month; this POC spent $3.07 in total.

## Caveats
- **Midpoint prices, not real fills:** the test uses midpoint prices and modelled costs, and real fills would be worse.
- **Missing faster sources:** GDELT misses X/Twitter and club channels, where news breaks first. A paid fast feed is untested.

## Recommendation
No-go, especially for an hourly agent.

## Files
- **Scripts:** `s01`–`s08`.
- **Prompts:** `prompts/`.
- **Results:** `out/` (`summary.json`, `trading.csv`, `event_study.csv`, `placebo.json` and the charts).
