# Design: Bedrock (Claude) matching of Winner games to Polymarket

Status: proposal, not built.

## Goal
Make money (paper trading for now) by comparing our models' probabilities for Winner soccer 1X2 games with live
Polymarket prices. Bets may be placed any time before kickoff (days ahead is fine). The owner wants the
Winner→Polymarket game matching done by an LLM instead of the current heuristic (polymarket.py: kickoff ±12 min +
Hebrew/English consonant-skeleton name similarity), because Winner names are Hebrew and ~99% untranslated.

## Current system (for context)
- One Docker Lambda `predict_winner_model` (il-central-1, 2 GB, ~1 min/run), EventBridge every 3 h.
- Run: Athena `api_odds` (Winner odds history, scraped every 15 min) → 3 models (value_v1, slack_v1, oddsflow_v1)
  → Polymarket Gamma API (upcoming soccer 3-way moneyline events, ~550 games / 8 days) → match → paper bet when
  p_model × fee-adjusted Polymarket odds − 1 > 3% (first trigger per model/game/outcome) → Firestore → settle from
  Athena `results`. Website (Cloud Run) only reads Firestore.
- Only big-league games (31 competitions) are used: ~25–60 open games at a time.
- Heuristic matcher today: ~22/25 big games matched, no known false matches after tightening.

## Proposal
LLM decides only "which Polymarket event is this Winner game (and is home/away swapped)". Code keeps the
betting math, de-dup, Firestore writes and settlement (deterministic, auditable, cheap).

Per run:
1. Winner odds + 3 models: unchanged.
2. Fetch Polymarket games: unchanged.
3. Matching:
   - Firestore `poly_matches/{winner_game_id}` caches confirmed matches; a game is matched once and reused.
   - For each new big-league game, code builds a shortlist: Polymarket events whose start time is within ±3 h of
     the Winner kickoff (usually 5–30 events).
   - One Bedrock call per run (Claude Haiku-class), all new games + shortlists in, structured JSON out:
     {winner_game_id, event_id | null, swapped, confidence, reason}. "No match" is retried next run until kickoff
     (Polymarket lists some games late).
4. Code validation of each answer: event_id must be in that game's shortlist; Winner and Polymarket kickoff times
   must agree within 15 min (sanity check that it is the same game, NOT a limit on when we bet); confidence
   threshold; existing guard |p_model − Polymarket mid| ≤ 0.15. The old heuristic matcher runs as a second opinion;
   disagreements are logged.
5. Paper bets, settlement, site: unchanged; each bet also stores the LLM's reason. Confirmed matches also yield
   English team names (`team_names` collection) the site can display.

Failure: if Bedrock fails, skip the Polymarket step for that run (no fallback guessing); models still run.
AWS: enable model access in Bedrock; add bedrock:InvokeModel to the Lambda role; call Bedrock in a region that
offers the model (il-central-1 may not). Only team names, leagues and kickoff times are sent.
Cost: ~8 calls/day, few dozen games each, caching makes most runs tiny → expected a few $/month (to verify).
Rollout: 2–3 days shadow mode (LLM matches recorded, heuristic still drives bets), hand-check disagreements,
then switch.

Rejected: letting the agent run everything (read odds, decide bets, write records): costlier, slower, not
reproducible, errors indistinguishable from signal in a paper test meant to measure the models.
