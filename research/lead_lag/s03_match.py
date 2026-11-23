"""Step 3: match Winner games to Polymarket events with the live matcher (lambdas/predict_winner_model/polymarket.py),
imported read-only. Output data/matches.parquet (one row per matched game with both venues' metadata)."""
import os, sys, json
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "lambdas", "predict_winner_model")))
import polymarket as pm  # noqa: E402

D = os.path.join(HERE, "data")
games = pd.read_parquet(os.path.join(D, "winner_games.parquet"))
poly = pd.read_parquet(os.path.join(D, "poly_events.parquet"))
names_f = os.path.join(D, "team_names.json")
names = json.load(open(names_f)) if os.path.exists(names_f) else {}

games["kick_utc"] = pd.to_datetime(games.kick_utc)
games = games[(games.kick_utc >= poly.start.min() - pd.Timedelta("1D")) & (games.kick_utc <= poly.start.max() + pd.Timedelta("1D"))]
out = []
for day, g in games.groupby(games.kick_utc.dt.date):
    p = poly[(poly.start >= pd.Timestamp(day) - pd.Timedelta("1h")) & (poly.start <= pd.Timestamp(day) + pd.Timedelta("25h"))]
    if len(p):
        out.append(pm.match(g[["unique_id", "home", "away", "kick_utc"]], p.reset_index(drop=True), names))
m = pd.concat(out).sort_values("score", ascending=False).drop_duplicates("event_id").drop_duplicates("unique_id")
m = m.merge(games, on="unique_id").merge(poly, on="event_id")
m["kick_diff_min"] = (m.start - m.kick_utc).dt.total_seconds() / 60
m.to_parquet(os.path.join(D, "matches.parquet"))
print("winner games in range", len(games), "poly events", len(poly), "matched", len(m), "swapped", m.swapped.sum())
print(m.kick_diff_min.describe())
print(m.groupby(m.kick_utc.dt.to_period("M")).size().to_string())
