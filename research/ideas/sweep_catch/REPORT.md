# Idea 2: catching sweeps with deep resting orders: GO for a paper pilot

*Research only. Saved from the agent's final message, 2026-11-21, with my independent spot-check added.*

## Verdict
Resting limit buys about 20¢ below the Polymarket mid return +38% to +52% per $ in every period, with 95% CIs above zero. Each order must also be at least 6¢ below Winner's fair price. We buy only when a big seller sweeps an empty order book, and the price snaps back within about 5 minutes in about 90% of fills.

The edge is real per $ but small in dollars:
- **Fills:** about 9 a month now (21 a month in train), with a median of about 25 shares.
- **Profit:** about $400 a month at 1,000 shares per order.
- **Collateral:** Polymarket very likely ties up the full cost of every open buy order. Quoting every game would lock $125k–280k, so the practical version quotes only in the last 3 hours before kickoff: about $5–13k locked for about $190–270 a month.

## Rules (chosen on train)
1. **Market:** every matched soccer 1X2 market, both sides of each outcome. Buying Yes catches dumps; buying No catches pumps.
2. **When to quote:** while the mid c of our token is between 0.08 and 0.92, from 72 hours before kickoff until 10 minutes before. The "last 3 hours" variant quotes only from 3 hours before kickoff.
3. **Buy order:** a limit buy at L = floor(c − 0.20), only if L ≤ Winner's fair price − 0.06 (gap ≥ 10¢ for the hourly bot).
4. **Refresh:** every 1 minute (continuous bot) or every 60 minutes (hourly bot). Both perform about the same.
5. **After a fill:**
   - Don't re-quote that outcome and side for 60 minutes.
   - Post a limit sell at floor(c).
   - If it hasn't sold 5 minutes before kickoff, sell 1¢ under the last real bid print, paying the fee. If there is no such print, the position is held to the result (12–29% of fills).

## Results (1,000-share orders; fills only when a real print went strictly through our price)

| Rule | Period | Fills/month | Return per $ [95% CI] | $/month | Capital locked |
|---|---|---|---|---|---|
| Continuous, 20¢ | train | 21.3 | +52% [+46, +59] | $782 | $126k |
| | val | 8.7 | +51% [+34, +86] | $405 | $239k |
| | test | 8.8 | +39% [+29, +57] | $387 | $277k |
| Hourly, 20¢ | train | 21.3 | +47% [+39, +56] | $747 | ~$126k |
| | val | 7.0 | +52% [+37, +80] | $435 | ~$239k |
| | test | 9.4 | +38% [+28, +52] | $389 | ~$277k |
| Last 3h only | train | 3.4 | +56% [+44, +74] | $188 | $5.3k |
| | val | 3.0 | +45% [+30, +81] | $266 | $10.7k |
| | test | 1.9 | +43% (2 games) | $251 | $13k |

- **Shallow orders lose.** At 2–5¢ below the mid the result is −2% to −9%.
- **The Winner filter is essential.** Without it, orders are negative down to 12¢ deep.
- **Wallet history adds nothing.**
- **The money is concentrated.** In val and test, the top 5 fills make 80–96% of the dollars. The return stays at +32% to +51% even with the best game removed.
- **Fills are getting rarer:** 43–64 a month in late 2025, about 10 a month in 2026.

## My spot-check (continuous rule, 253 fills)
- **Entries:** every fill has a real print strictly below our price within ±60 seconds of the fill (100%).
- **Exits:** 79% have a real buy print above the exit target before kickoff, and 65% have enough volume to sell the whole position.
- **Profit source:** most of the profit comes from fills that hit their target, which are real prints in both directions (train +52%, val +60%, test +44%). Fills held to the result are a small share of the dollars.
- **Look-ahead:** the mid used to place each order is 2 minutes old, so the strategy only uses information it would have had.

## Risks and open questions
- **Collateral:** Polymarket probably locks the full cost of open orders. To verify.
- **Competition:** other bots may now quote these deep empty levels. Paper trading would show this.
- **Small samples:** val and test have only 26 and 14 fills, and the dollars are lumpy.
- **Partial exits:** about a third of exits might only be partly sold at the target.
- **Search size:** 432k grid combinations were tried. The edge rises smoothly with depth, so it looks structural rather than overfit.

## Live
- An hourly Lambda is enough. Each run takes about 50–300 API calls.
- No LLM is needed.
- Start with the last-3h variant at 100–300 shares per order.

## Files
- Scripts `s01`–`s08`.
- Outputs in `out/`: `final_summary.csv`, `final_fills.parquet`, `collateral.csv` and the charts.

---

# Production tuning (2026-11-22, `s10_detail.py`, `s11_eval.py`): the version that went live as `sweep_v1`

`s10` replays every candidate fill again, with more choices than the first search:
- **Depth:** deeper levels up to 35¢, plus depth relative to the price and hybrids of the two.
- **Refresh cadence:** 1, 5, 15, 30 and 60 minutes.
- **Winner data lag:** 1, 30, 60 and 180 minutes.
- **Fill-detection delay before the sell order is placed:** 0 to 60 minutes.
- **Sell targets:** six options.
- **Fill size:** the shares that traded through our price in the same transaction.
- **Sell size:** capped by the shares real buyers bought above the target.
- **Money tied up in resting orders:** tracked for every setting.

`s11` applies the filters, the cooldown and the one-fill-per-refresh rule afterwards. Choices were made on train, checked on validation (Jul–Sep), and the frozen rule was run once on the test period (Oct 1 – Nov 18). The test period had already been seen in the first search.

## What each detail does
| Detail | Finding | Choice |
|---|---|---|
| Depth | Return per $ rises steadily with depth, while $/month stays about flat. Depth relative to the price (for example 50% of c) beats a fixed 20¢. | Two levels: A = min(c−15¢, 0.6c), B = min(c−20¢, 0.5c) |
| Quoting window | Most capital-efficient in the last 2–3 h (lineup news). Quoting from 72 h earns about 2× the $ but ties up 20–25× the money. | 3 h to 10 min before kickoff |
| Refresh cadence | 1–15 min are equal; 30–60 min lose about 5 points of return on train | Every 5 min |
| Fill-detection delay | Doesn't matter even at 60 min: the price returns to its pre-dump level and stays there | — |
| Winner data lag | Doesn't matter up to 3 h, so Winner's 3-hourly data is enough | — |
| Winner gap filter | Essential for shallow levels; rarely matters for the deep levels | Kept at ≥ 6¢ as a safety check |
| Sell target | floor(pre-dump price) ≈ that minus 1¢ ≈ Winner fair; selling halfway back is much worse | floor(c) |
| Cooldown | Barely matters | 60 min |
| Sides | Buying "No" (catching spikes) earns most of it; buying "Yes" was weak in validation | Both |
| Stop quoting | 2 minutes before kickoff adds fills, but they end up held to the result | 10 min; sell leftovers at kick − 5 min |
| Order size | Fills are capped by the volume that went through our price (median about 45 shares), so profit grows much more slowly than size | 500 per level, recording the volume so results can be rescaled |

## The frozen rule (2 levels × 500 shares, both sides, last 3 h, 5-minute refresh)
| Period | Fills | Games | Fills/month | Return per $ [95% CI] | $/month |
|---|---|---|---|---|---|
| Train | 68 | 35 | 6.8 | +72% [+50, +87] | $221 |
| Validation | 17 | 9 | 5.7 | +69% [+59, +77] | $212 |
| Test (once) | 5 | 2 | 3.1 | +100% (2 games) | $149 |

- **Months:** 13 of 14 were positive.
- **Exits:** 54% of fills sold at the target and 22% were held to the result (train).
- **Money tied up:** about $4–8k at these sizes.
- **Samples are small:** validation and test have only 9 and 2 games, so the live paper record is the real test.

## Production check
- I replayed the production code (`lambdas/predict_winner_model/sweep.py`) on a past test game, using the real trade feed and price history.
- It reproduced the research fills: same prices, same fill sizes (30 and 500 shares) and the same +$160 exit.
