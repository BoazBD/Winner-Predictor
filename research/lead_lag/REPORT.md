# Who moves first on big soccer odds moves: Winner or Polymarket?

*Data: 16,660 matched pre-match games, October 2024 to 18 November 2026, nearly all from September 2025 on. Written 20 November 2026.*

## Verdict
**Polymarket leads and Winner lags.** When Winner's odds make a big move, Polymarket has usually moved already:
- In **47%** of Winner's ≥3-point moves, Polymarket had already made half of the move before Winner did.
- In another **17%**, both moved inside the same 15-minute Winner scrape, so the order can't be told apart.
- Winner was clearly first in only **23%**. That is no more than chance: a random, unrelated Polymarket series gives 26%.

The cross-correlation of 15-minute price changes agrees. Polymarket's change in the previous 15 minutes predicts Winner's change now about 4 times as strongly as the reverse (0.064 vs 0.016).

**The "Winner moved, Polymarket hasn't yet" trade is real but too small to pay the costs.** After such a signal, Polymarket still drifts our way by **+0.6 to +0.85 probability points** on average by kickoff. But the taker fee plus half the spread costs about 2 cents per share. The net return is about **−2% per $ staked**, whether you check every 15 minutes, every hour or every 3 hours. In the fee era (April–November 2026), every combination of check frequency and threshold is negative, and 7 of the 9 confidence intervals are entirely below zero. **Checking more often does not help.**

**The earlier "11 of 12" finding, retested at scale.** When Winner really did move first, Polymarket reaches half of Winner's move within 6 hours in **55%** of cases, against 17% by chance. Most of that (35%) happens within 15 minutes.

## 1. Data
- **Polymarket:** 25,918 closed soccer 1X2 events. Busy days were re-queried in 3-hour windows because the API stops paging at about 2,000 events.
- **Matching:** the live matcher, unchanged, matched 17,232 (67%). 16,763 have price history on both venues.
- **Usable sample:** 16,660 games, after dropping any game where the two venues are more than 15 points apart at kickoff. That is 2,780 top-league games, 72 Israeli and 13,808 other. 11,370 games have at least one major move.
- **Coverage:** a few English Premier League games in September–October 2024, then MLS in June–August 2025. Broad coverage starts in **September 2025** (545 games), then 1,000–2,000 a month to May 2026 and 3,000–4,000 a month since August 2026 (`out/fig_coverage.png`).
- **Winner series:** margin-free probability per outcome, (1/odds) ÷ Σ(1/odds). A change is known only at the snapshot where it appears. Median scrape interval 15.0 minutes; 99th percentile 15.2.
- **Polymarket series:** the 1-minute `prices-history` value for each "Yes" token (equals the order-book midpoint). A centred 5-minute median is used for detecting and timing moves; the trading simulation uses the raw midpoint.
- **Window:** the 72 hours before kickoff, nothing after kickoff. Kickoff is the earlier of Winner's time + 1 minute and Polymarket's start time.

## 2. Definitions
- **Major move:** one outcome's probability rises (or falls) by at least X points within a trailing 3-hour window, X = 3, 5 or 8.
  - **Detection time** is the first minute the threshold is crossed (for Winner, the snapshot where it shows).
  - **Base** is the low (or high) of the window; the move size M is measured from it.
  - At most one event per venue per game per 6 hours, keeping the outcome with the largest move.
  - The move must still be at least half intact 30 minutes later (dropped 3,304 short-lived events).
- **Half-move time:** for each venue, the first minute its price has moved at least M/2 from its own value at the base time, in the move's direction.
- **Who moved first:** Winner's true change time lies between the previous snapshot and the one where it shows.
  - **Polymarket first:** Polymarket's half-move time is at or before Winner's earliest possible time (includes Polymarket having moved half in the 6 hours before the base).
  - **Winner first:** Polymarket's half-move time is after Winner's latest possible time.
  - **Same 15-minute window:** Polymarket's half-move time falls inside Winner's scrape interval.
  - **Never:** the other venue does not reach half before kickoff.
- **Chance baseline:** the other venue's series taken from a random different game, lined up by minutes to kickoff.
- **Random-time placebo:** the same move shape placed at a random minute of the same game.
- **Confidence intervals:** 95% bootstrap, resampling games.

## 3. Event study (`out/fig_who_first.png`, `out/fig_event_paths.png`, `out/fig_lead_hist.png`)

| Major move on | Size | Moves (games) | Winner first | Same 15-min window | Polymarket first | Other venue never followed |
|---|---|---|---|---|---|---|
| Winner | 3+ | 6,946 (5,772) | 22.7% [21.8, 23.5] | 17.4% [16.6, 18.3] | **47.5% [46.2, 48.7]** | 12.5% [11.6, 13.3] |
| Winner | 5+ | 1,864 (1,732) | 22.8% [21.0, 24.7] | 24.2% [22.4, 26.2] | 42.0% [40.0, 44.0] | 11.1% [9.9, 12.4] |
| Winner | 8+ | 301 (298) | 18.6% [14.6, 22.7] | 31.6% [26.4, 36.6] | 42.2% [37.0, 47.4] | 7.6% [4.7, 10.6] |
| Polymarket | 3+ | 19,746 (10,773) | 7.8% [7.4, 8.1] | 15.5% [15.0, 16.1] | 37.4% [36.7, 38.2] | 39.3% [38.6, 40.1] |
| Polymarket | 5+ | 8,320 (5,282) | 3.1% [2.7, 3.5] | 14.4% [13.6, 15.3] | 31.7% [30.5, 32.8] | 50.8% [49.3, 52.1] |
| Polymarket | 8+ | 3,875 (2,435) | 0.8% [0.5, 1.1] | 7.5% [6.5, 8.4] | 17.8% [16.5, 19.1] | 73.9% [72.6, 75.4] |

Chance baseline (other venue from a random unrelated game):

| Major move on | Size | Winner first | Same window | Polymarket first | Never |
|---|---|---|---|---|---|
| Winner | 3+ | 26% | 1% | 10% | 63% |
| Winner | 5+ | 17% | 0% | 5% | 77% |
| Polymarket | 3+ | 7% | 1% | 18% | 75% |
| Polymarket | 5+ | 3% | 1% | 8% | 89% |

- **On Winner's big moves,** "Polymarket first" and "same window" are far above chance; "Winner first" is not. In 66% of Winner's big moves, Winner moved toward the level Polymarket was already showing (`out/gap_closing.csv`): most are catch-up moves.
- **On Polymarket's big moves,** Polymarket has about 3× as many ≥3-point moves as Winner, many of them thin-book midpoint noise. Polymarket jumps of 12+ points are almost never followed by Winner (98% never).
- **Average path:** when a Winner move becomes visible, Polymarket has on average already done 49% of it and adds nothing over the next 6 hours (49% → 50%). When Polymarket moves, Winner has done about 15% at detection and ends near 35–40%. Placebos stay at about 0.

Conditional follow (the other venue had *not* moved half by the time the mover did):

| | Moves | ≤15 min | ≤1 h | ≤3 h | ≤6 h | By kickoff |
|---|---|---|---|---|---|---|
| Winner ≥3 first → Polymarket follows | 2,443 (35% of Winner's moves) | 35% | 42% | 50% | 55% | 64% |
| Chance | | 4% | 7% | 12% | 17% | 29% |
| Polymarket ≥3 first → Winner follows | 15,150 (77% of Polymarket's moves) | 15% | 28% | 33% | 35% | 49% |
| Chance | | 1% | 3% | 5% | 7% | 19% |

Lead times in minutes (Winner time at latest / middle / earliest point of its scrape interval):

| Trigger | n | Median gap (latest / mid / earliest) | Middle 50% (mid) | Within ±15 min |
|---|---|---|---|---|
| Winner ≥3: Polymarket minus Winner | 3,232 | −1.5 / **+6** / +14.5 (mid CI 6–8) | −4 to +57 | 52% |
| Winner ≥5: Polymarket minus Winner | 1,042 | −4.5 / +3 / +10.5 | −5 to +31 | 60% |
| Polymarket ≥3: Winner minus Polymarket | 10,940 | +12.5 / **+20** / +27.5 | +5 to +164 | 42% |

## 4. Cross-correlation and regressions (`out/fig_xcorr.png`, `out/xcorr.csv`)
15,546 games, 3.96 M 15-minute intervals.

| Lag | Correlation of 15-min changes |
|---|---|
| Polymarket 30 min earlier vs Winner now | 0.017 |
| **Polymarket 15 min earlier vs Winner now** | **0.064 [0.061, 0.068]** |
| Same interval | 0.102 |
| Winner now vs Polymarket 15 min later | 0.016 [0.015, 0.018] |
| Winner now vs Polymarket 30 min later | 0.007 |

- Summed over lags 1–8: 0.10 Polymarket-leads, 0.03 Winner-leads.
- 1-point Winner move → Polymarket +0.47 same interval, +0.15 [0.14, 0.17] over the next 6 h.
- 1-point Polymarket move → Winner +0.02 same interval, +0.025 over 6 h (diluted by Polymarket noise).
- Gap-closing regression: Polymarket closes most temporary gaps (Winner's share 3–4% overall, 22–23% in >$250k games) — mostly midpoint wandering and returning.

## 5. Breakdowns (Winner ≥3-point moves; more in `out/tables.md`)

| | Winner first | Same window | Polymarket first | Never followed |
|---|---|---|---|---|
| Top leagues (1,163) | 28% | 12% | 31% | **29%** |
| Other leagues (5,768) | 22% | 19% | 51% | 9% |
| Polymarket volume > $250k (1,306) | 27% | 12% | 36% | 25% |
| Volume < $10k (1,268) | 17% | 20% | 55% | 8% |
| 24–72 h before kickoff | 31% | 8% | 51% | 10% |
| 6–12 h | 31% | 16% | 42% | 11% |
| 1–3 h | 23% | 19% | 39% | 20% |
| **Last hour (2,209)** | 12% | 23% | **54%** | 12% |
| Favourite > 50% | 26% | 16% | 41% | 17% |
| 25–50% | 20% | 19% | 53% | 8% |
| Longshot < 25% (76) | 33% | 8% | 25% | 34% |

- Hour of day (Israel): Winner first more often 07–10h (29–34%) than 19–21h (13–18%).
- Day of week: flat, 20–27% "Winner first".

## 6. Trading translation (`out/fig_trading.png`, `out/trading*.csv`)
**Rule.** At each check time: if an outcome's Winner probability moved ≥X points over the last 6 hours and Polymarket moved less than half that in the same direction, buy on Polymarket ("Yes" if up, "No" if down). One bet per game, outcome, direction and check frequency.

**Costs.** Entry = midpoint one minute after the check + half spread + fee 0.05·p·(1−p). Half-spread 0.5c for ≥$10k volume, 1.5c below (from a live snapshot of 1,818 markets).

**Valuation.** At the Polymarket kickoff midpoint; realised settlement shown as a check.

| Check every | Winner move ≥ | Bets (games) | Polymarket moves our way, points | Return per $ | No fee | No fee, no spread | Realised |
|---|---|---|---|---|---|---|---|
| 15 min | 2 | 10,916 (6,232) | +0.50 [0.40, 0.59] | −2.5% [−2.7, −2.2] | −0.4% | +1.2% | −2.6% [−4.6, −0.6] |
| 15 min | 3 | 3,859 (2,662) | +0.59 [0.38, 0.79] | **−2.1%** [−2.6, −1.6] | +0.1% | +1.6% | −1.0% [−4.7, 2.6] |
| 15 min | 5 | 713 (588) | +1.08 [0.52, 1.61] | −0.9% [−2.2, 0.3] | +1.3% | +2.9% | −3.4% [−10.8, 3.4] |
| 1 h | 2 | 7,579 (4,648) | +0.62 [0.48, 0.74] | −2.1% [−2.5, −1.8] | 0.0% | +1.5% | −3.1% [−5.4, −0.4] |
| 1 h | 3 | 2,674 (1,934) | +0.68 [0.43, 0.92] | **−1.8%** [−2.3, −1.2] | +0.3% | +1.8% | −1.2% [−5.2, 3.0] |
| 1 h | 5 | 489 (411) | +0.83 [0.16, 1.44] | −1.3% [−2.8, 0.2] | +0.9% | +2.5% | −6.2% [−14.2, 2.4] |
| 3 h | 2 | 4,926 (3,216) | +0.62 [0.45, 0.77] | −2.0% [−2.4, −1.6] | 0.0% | +1.5% | −1.5% [−4.2, 1.3] |
| 3 h | 3 | 1,740 (1,299) | +0.85 [0.60, 1.15] | **−1.4%** [−2.1, −0.7] | +0.7% | +2.2% | +1.7% [−3.2, 6.7] |
| 3 h | 5 | 309 (267) | +0.99 [0.38, 1.75] | −0.9% [−2.4, 1.1] | +1.3% | +2.9% | −5.4% [−17.6, 6.7] |

Fee era only (April–November 2026), Winner move ≥3:

| Check every | Return per $ | Bets with gap > 4 points | Their return per $ |
|---|---|---|---|
| 15 min | −2.7% [−3.3, −2.2] | 392 | −1.5% [−3.8, 0.3] |
| 1 h | −2.4% [−3.0, −1.7] | 243 | −0.6% [−3.5, 3.0] |
| 3 h | −2.2% [−3.0, −1.3] | 150 | +0.8% [−3.6, 5.7] |

- Gross edge (0.5–0.85 pt drift, median 0) is smaller than costs (~1.5–2.5c per share).
- Drift is larger >12 h before kickoff (+1.2 to +1.9 pts, return −0.9% to +0.6%), in $10k–$250k markets (~+1 pt) and non-top leagues (+0.75 to +1.25 vs +0.1 to +0.2 top leagues). It is zero or negative when Polymarket is already past Winner's new level.
- The >4-point gap subset is +3.9% to +7.6% on the whole sample, driven by Sept 2025 (+46%) and Dec 2025 (+17%); most months since Jan 2026 are negative.

## 7. Robustness
- Shuffled-pair placebos fall to the chance table; random-time placebo paths are flat.
- By month (`out/fig_monthly.png`): Nov 2025–Nov 2026, "Polymarket first" is 38–45% of all ≥3-point moves every month, "Winner first" 7–15%; "same window" grew from ~7% to ~20%. The trade (1 h checks, 3 pts) is negative in 12 of 15 months.
- Headline numbers rest on ≥1,000 events in ≥1,000 games; the 8-point rows and gap > 4 subset are smaller.

## 8. Caveats
1. Polymarket prices are midpoints, noisy in thin books (this works against finding a Polymarket lead on Winner-triggered moves, and we still find one).
2. Spreads are modelled from one live snapshot.
3. Winner timing is only known to within 15 minutes.
4. Valuing at the kickoff price assumes it is fair; settlement results are noisy (±4–15%).
5. Covers the 67% of events the matcher matches; Israeli leagues almost absent; pre-Sept 2025 too thin.
6. Window, cooldown and half-move rule are judgment calls; 3/5/8-point results agree in direction.

## 9. Recommendations
1. **Stop treating a Winner move as a leading signal for Polymarket.** Don't scale the taker strategy; don't build 15-minute checks.
2. **If kept:** bet only >12 h before kickoff; non-top leagues with $10k–$250k volume; skip when Polymarket is already at/past Winner's new level; post limit orders; forward-test the >4-point gap variant on kickoffs after 2026-11-07.
3. **Use the lead the other way:** feed Polymarket's recent move into bet decisions and skip bets where it moved against us; treat Winner catch-up moves as no signal; if ever betting on Winner, stale Winner odds after a Polymarket move are the exploitable case (still not halfway 6 h later in ~65% of cases).

## Reproduce
Python `winner_model/.venv`; matplotlib in `research/lead_lag/.pylib`. Run `s00_spreads.py` … `s08_tables.py` in order (step 4 downloads ~23 GB of minute prices into `cache/prices/`).
