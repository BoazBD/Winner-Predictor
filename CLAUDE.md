# Notes for AI agents

Winner (Israeli bookmaker) odds → betting models, paper-traded against Polymarket.
- **Pipeline:** AWS Lambdas in il-central-1, writing to Firestore.
- **Site:** Flask on GCP Cloud Run (`site/`).

## Production systems (all paper trading, no real money)
- **`predict_winner_model` Lambda** (`lambdas/predict_winner_model/lambda_function.py`, every 3 h): models value_v1, slack_v1, oddsflow_v1, move_v1 and longshot_v1, plus Polymarket matching (team names are translated with Bedrock Haiku 4.5).
- **`sweep_catch` Lambda** (`lambdas/predict_winner_model/sweep.py`, every 5 min): **sweep_v1 "Panic catcher"**. Full context: **`lambdas/predict_winner_model/SWEEP_V1.md`**.

## Rules
- Never commit, print or share `service-account.json` (Firestore credentials).
- Don't change the existing models' strategies unless asked.
- Commit only when asked.
- **Research hold-out:** all data through 2026-11-18 has been used. Clean evaluation = kickoffs after 2026-11-18.
- **Polymarket data pitfall:** round prices to 4 decimal places before fill comparisons (float32 artefacts once faked an edge).
- Research lives in `research/` (one folder per study, with `REPORT.md`). Large data and caches there are not in git.
