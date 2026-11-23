# Pre-registration — oddsflow (written 2026-11-13, before any look at kickoff >= 2026-03-01 in this work)

All exploration, feature design, model choice and threshold choice used matches with kickoff < 2026-03-01 only
(`exp/common.py` and `backtest.py dev` never load later matches). Caveat: earlier projects in this repo have
already looked at Mar–Nov 2026, and the general design here (calibration + rounding + closing-line drift) follows
what those projects found, so this is a strong check, not a perfectly clean one.

## Frozen model
- Code: `features.py`, `table.py`, `model_lib.py` as of this file's timestamp.
- Features: odds history and timing only (list `FEATS` in `features.py`). No team or league identity. The payout
  tier is inferred from the overround of the prices themselves.
- Stage 1: LightGBM regression (3 seeds, 600 rounds, 31 leaves, min leaf 500, L2) of
  `closing logit − current logit` (rounding-corrected, margin-free), on all soccer 1X2 decision points.
- Stage 2: pooled logistic calibration (cubic in logit, tier intercepts/slopes, draw/away terms) fitted on closing
  prices vs results; applied to `current logit + predicted drift`; probabilities renormalised to sum to 1.
- Trained ONCE on all soccer 1X2 matches with kickoff < 2026-03-01. No refit during the test.
- Universe: every soccer 1X2 market, no league filter.

## Frozen betting rule
1 unit on an outcome the first time `ev = p_model × odds − 1 > threshold` at a snapshot where the market is
actually present (every 15-minute snapshot is checked), once per (market, outcome).

- PRIMARY:   threshold  0.00   (dev walk-forward Oct 2024–Feb 2026: 426 bets, ROI −0.8%, 90% CI −5.0..+3.3;
                                calibrated EV at the closing line +0.5%; beat the close 71%)
- Secondary: threshold +0.01   (dev: 276 bets, +2.2%, CI −2.6..+6.9; closing-line EV +1.0%)
- Secondary: threshold −0.01   (dev: 566 bets, −1.7%, CI −5.6..+2.0; closing-line EV 0.0%)

## Test
Kickoff 2026-03-01 .. 2026-11-07, `python backtest.py final` then `python report.py final -0.01 0 0.01`, run once,
reported unchanged.

## Expectation stated in advance
True ROI of the primary rule: about 0% (plausible range −2%..+1.5%), 4–6 bets a week, all favourites at odds
1.05–1.45 in the 90%-payout tier, most placed days before kickoff. With ~200 test bets the 90% CI will be about
±5 points, so the test cannot distinguish 0% from the −0.6% / −1.1% of the earlier models; the closing-line
figures (share of bets beating the close, calibrated EV at the close) are the sharper check.
Predicted EV is expected to rank only weakly above 0.
