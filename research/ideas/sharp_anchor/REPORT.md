# Idea 4: Pinnacle / Betfair as the sharp anchor — no-go

*Research only. Saved from the research agent's final message, 2026-11-21.*

## Verdict
Polymarket does not drift toward the sharp bookmakers; it is the other way round. Betfair Exchange closes about 90% of the gap to Polymarket by kickoff, while Polymarket closes about 5% of the gap to Betfair.

- **Best rule (Betfair gap ≥ 2 points, sell before kickoff):** −9.1% / −7.9% / −7.9% per $ on train / validation / test.
- **With no spread and no fee:** only +1.2% to +1.8%, which is the most the rule could ever earn.

## Coverage
- **Matching:** 8,345 football-data.co.uk games matched to Polymarket.
- **Pinnacle data stops in October 2025 – January 2026,** so only 884 games have it, all in train. Pinnacle closed its public feed in 2025.
- **Betfair Exchange was used instead** (4,153 games).
- **The odds' collection time is uncertain.** Football-data only says "Tuesday or Friday afternoon"; the test assumes 18:00 London, and the odds look roughly 5–7 hours old by then.

## Results (return per $, 95% CI)

| Rule | Train | Validation | Test |
|---|---|---|---|
| Sell before kickoff, Betfair gap ≥ 2 points | −9.1% [−10.4, −7.9] (227 trades) | −7.9% [−9.7, −6.1] (79) | −7.9% [−9.8, −6.1] (82) |
| Same rule, no fee | −3.3% | −2.2% | −1.8% |
| Same rule, zero cost | +1.2% | +1.7% | +1.8% |
| Hold to the result, gap ≥ 1 point | −4.6% [−13.5, +3.8] | −9.1% [−26, +9] | −5.0% [−22, +14] |

- **Limit-order entry, counting only trades strictly through our price:**
  - about 10% of orders fill;
  - filled orders drift −0.6 points against us;
  - the result is −1% to −4%.

## Which market is sharpest (log loss)
- **At close:** Polymarket and Betfair tie (0.98895 vs 0.98890). Both beat the market average (0.9910) and Winner (0.9938).
- **Polymarket vs Pinnacle:** Polymarket's close beat Pinnacle's close on 1,327 games.
- **Earlier, at the time the bookmaker odds were collected:** Polymarket already beats Betfair.
- **Combining the anchors:** adds nothing out of sample.

## Live
- **No free live Pinnacle odds.**
- **Betfair:** the API needs a funded account, and the live key costs about £299.
- **No role for an LLM:** this is a numeric rule.

**Recommendation:** no-go.

## Files
Scripts `s01`–`s06`. Outputs in `out/`: `drift.csv`, `sharpness.csv`, `strategy_grid.csv` and charts.
