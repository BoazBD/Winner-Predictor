# oddsflow — report (2026-11-13)

## Bottom line
The model does not show a profit. On the pre-registered test (kickoff 1 Mar – 7 Nov 2026, run once, unchanged)
the primary rule returned **−4.9% over 140 bets (90% CI −12.1% .. +2.3%)**. Development walk-forward was −0.8%
over 426 bets. Pooled: **−1.8% over 566 bets (CI −5.4% .. +1.9%)**. It does not beat the two earlier models in
realised ROI; all three are statistically indistinguishable from each other and from zero.

What it does do reliably: it removes the margin. Flat betting the same market loses 22%; the bets it picks got a
better price than the close 67–71% of the time, by 4% on average. My estimate of true ROI is **−1% to +1%**, at
4–6 bets a week. I believe that is the ceiling for Winner's own 1X2 odds.

## Results

Bets: 1 unit, first snapshot at which the rule fires, once per (market, outcome), checked at every 15-minute
snapshot where the market is actually present. CIs are 90% bootstrap, clustered by match.

| | Dev walk-forward, Oct 2024 – Feb 2026 (74 wk) | Test, Mar – Nov 2026 (36 wk) |
|---|---|---|
| **Primary: EV > 0** | −0.8%, 426 bets (−5.0..+3.3), odds 1.27, 5.8/wk | **−4.9%, 140 bets (−12.1..+2.3), odds 1.24, 3.9/wk** |
| Secondary: EV > +0.01 | +2.2%, 276 bets (−2.6..+6.9), 3.8/wk | −1.6%, 84 bets (−10.5..+6.7), 2.3/wk |
| Secondary: EV > −0.01 | −1.7%, 566 bets (−5.6..+2.0), 7.7/wk | −0.5%, 207 bets (−6.8..+5.6), 5.8/wk |
| Rounding-slack rule (re-run here) | −0.3%, 645 bets (−4.1..+3.3) | +0.7%, 271 bets (−4.9..+6.0) |
| Value model (quoted) | +0.5%, 555 bets | −0.6%, 317 bets |
| Flat, every outcome at close | −22.2%, 175,053 bets | −22.7%, 92,253 bets |
| Flat, favourites at close | −14.9% | −14.3% |
| Flat, 90% tier / 85% / 80% | −11.7% / −20.9% / −22.9% | −12.1% / −18.6% / −23.3% |

Closing-line value, primary rule:

| | Dev | Test |
|---|---|---|
| Price better / equal / worse than the final price | 71% / 20% / 10% | 67% / 24% / 10% |
| Mean price advantage over the final price | +4.0% | +4.4% |
| EV if the calibrated closing price is the truth | +0.5% | +1.4% |
| EV the model predicted | +0.9% | +0.7% |

Log-loss against margin-free market probabilities (3-way, all decision points, per-match weight):

| | Dev | Test |
|---|---|---|
| Model / market | 0.9895 / 0.9938 (−0.0043) | 0.9843 / 0.9900 (−0.0057) |
| Last snapshot only | −0.0033 | −0.0065 |
| 90% tier / 85% / 80% | −0.0014 / −0.0085 / −0.0043 | −0.0016 / −0.0070 / −0.0059 |

Almost all of the log-loss gain is the favourite-longshot correction; the price path adds about 0.0002.

Primary rule by period and odds band:

| Period | Bets | ROI | | Odds | Dev bets / ROI | Test bets / ROI |
|---|---|---|---|---|---|---|
| 2024 Q4 | 143 | −8.3% | | ≤ 1.10 | 33 / −0.8% | 17 / −9.7% |
| 2025 Q1 | 110 | +1.6% | | 1.10–1.20 | 111 / −4.5% | 36 / −1.8% |
| 2025 Q2 | 52 | +4.6% | | 1.20–1.30 | 149 / +3.5% | 66 / −3.9% |
| 2025 Q3 | 32 | +17.0% | | 1.30–1.45 | 131 / −2.2% | 21 / −9.5% |
| 2025 Q4 | 53 | −1.5% | | > 1.45 | 2 | 0 |
| 2026 Q1 (dev, Jan–Feb) | 36 | −0.8% | | | | |
| 2026 Q1 (test, Mar) | 16 | −14.1% | | | | |
| 2026 Q2 | 47 | −21.6% | | | | |
| 2026 Q3 | 38 | +13.8% | | | | |
| 2026 Q4 | 39 | +0.6% | | | | |

Every bet is a favourite in the 90%-payout tier (home 80%, away 20%); the median bet is placed about 5 days
before kickoff, i.e. near the market's opening.

**Does higher predicted EV earn more?** Not in realised profit. Test, by predicted EV: 0–0.5% → −5.0% (70 bets),
0.5–1% → −6.4% (33), 1–2% → +0.2% (32), > 2% → −27% (5). Dev is equally unordered. The closing-line estimate
does rise with predicted EV (dev: −0.2%, +0.9%, +0.5%, +2.6%), so the ranking is real in price terms but far too
small to see in results at this sample size.

**Why the test missed.** 2026 Q2 lost 21.6% on 47 bets (30 winners, where about 38 were needed to break even at these odds) while
those same bets beat the close as usual. That is consistent with variance, and the CI includes zero, but I cannot
rule out that the calibrated closing price overstates favourites by a point or two — the test cannot separate
the two.

## What the model is
Stage 1 predicts how far each outcome's price will move by the close; stage 2 converts the expected closing
price to a probability with a favourite-longshot calibration. Three things add up to the bets it picks:
1. 90%-tier favourites are only about 5% under fair value to begin with;
2. a price rounded up on the 0.05 grid is worth 1–2% more;
3. those favourites shorten by about 4% between opening and close, and the model picks the ones that will.

## What I tried
- **Direct outcome model** (LightGBM on win/lose with the market logit as offset): log-loss as good as the final
  model, but its bets lost 6–9% in volume. Trained on noisy results it "finds" value in longshots.
- **Level only** (calibration + rounding, no drift): about 0% on 200 bets, fewer bets, same conclusion.
- **Two-stage drift model** (final): best closing-line value at every threshold.
- **11 drift variants** on 3 folds — Huber vs L2, bigger and smaller trees, favourites only, 90%/85% tiers only,
  tier weighting, no path features, no clock features, 3-seed bagging. All landed within noise of each other
  (closing-line EV +0.3% to +0.8% at EV > 0). Dropping the path features cut explained drift variance from 3.1%
  to 2.1% but barely changed the bets. I kept the plain bagged L2 version rather than pick a winner by chance.
- **Rounding-corrected probabilities**: the main new feature. No measurable log-loss change; it mostly makes the
  path features less noisy.
- **Does the close under-react to its own path?** No consistent effect. Given the closing price, the move since
  open, the last move and the last 3 hours add nothing in the 90% tier (|z| < 1.5); the last move has a small
  effect in the 80% tier (z = 2.4) and the opposite sign in the 85% tier (z = −3.8).
- **Rounding slack vs results at the close**: the direction is right in the 80% tier (favourites ≤ 1.5: −13% with
  the price rounded down, −6% rounded up) but it is not statistically detectable in the 90% tier alone.
- Not tried: neural sequence models (earlier work found them no better, and the path adds only 0.0002 log-loss
  here), features from other matches' prices at the same time, staking or bet-timing optimisation.

## Facts checked
Confirmed: three overround tiers, constant through a market's life (centres 1.111, about 1.171 rather than 1.176,
and 1.25); 0.05 grid below 9.5; flat loss 22–23%; favourite-longshot bias; closing more accurate than opening;
in-play rows 0.06%. One difference from the brief: markets are rarely withdrawn at 15-minute resolution — only
0.3% of price segments follow a gap.

## Honest expectation live
About 0% (−1% to +1%), 4–6 bets a week at odds around 1.25, with long losing stretches: a quarter at −20% happened
in the test. Staking it is a way to bet Winner at roughly no cost, not an income.

## What would actually beat the margin
The predictable part of Winner's line movement explains about 3% of its variance; the rest is news Winner's feed
has not received yet. To capture it you need the source the feed is following:
- **Pinnacle (or Betfair Exchange) 1X2 odds, timestamped to the minute, for the same matches**, joined to
  Winner's snapshots. The bet is simply: Winner's price > sharp fair price / 0.98. Earlier analysis showed about
  3% of minor-league outcomes are at some point priced more than 3% above their closing value, worth about +11%;
  a sharp feed is what identifies them in real time.
- Second: confirmed line-up times (about 60 minutes before kickoff) for the 90% and 85% tiers.
- Useful from Winner itself: a faster scrape (1–2 minutes instead of 15) around those moments, since the median
  mispriced quote lasts about an hour.

## Reproduce
`python backtest.py dev && python report.py dev`; `python backtest.py final && python report.py final`.
Saved outputs: `exp/report_dev.txt`, `exp/report_final.txt`. The spec was frozen in `PREREGISTRATION.md` before
the test was run. The shipped model is retrained on all data to 2026-11-07, so Mar–Nov 2026 is now used up for
this model as well; the next clean check is matches after 7 Nov 2026.
