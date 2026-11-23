"""s08 - Collect headline numbers into out/summary.json: LLM cost, per-run/per-month projections, results.

Projection for live: an hourly run covering ~60 top games/week; per run the screen sees only NEW headlines
(dedup), Sonnet only for material ones. Cost per call measured from logs/llm_usage.csv.
"""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
u = pd.read_csv(os.path.join(HERE, "logs/llm_usage.csv"))
by = u.groupby("stage").agg(calls=("usd", "size"), usd=("usd", "sum"), in_tok=("in_tok", "sum"), out_tok=("out_tok", "sum"))
by["usd_per_call"] = by.usd / by.calls
E = pd.read_csv(os.path.join(HERE, "out/events.csv"))
H = pd.read_parquet(os.path.join(HERE, "data/headlines.parquet"))
s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet"))
n_games = len(s)
scr = by.loc["screen_normal"]
usd_screen_per_game = scr.usd / H.unique_id.nunique()
usd_judge_per_event = by.loc["judge"].usd_per_call if "judge" in by.index else 0
events_per_game = len(E) / n_games
per_game = usd_screen_per_game * H.unique_id.nunique() / n_games + events_per_game * usd_judge_per_event
summary = dict(
    llm_cost_by_stage=by.round(5).reset_index().to_dict("records"),
    total_llm_usd=float(u.usd.sum()),
    sample_games=n_games, games_with_headlines=int(H.unique_id.nunique()), headlines=len(H),
    material_events=len(E), sonnet_signals=int(E.sonnet_sig.sum()),
    usd_per_game_all_in=float(per_game),
    usd_per_month_at_250_games=float(per_game * 250),
    note="hourly runs re-screen only new headlines, so cost/month ~ games/month x cost/game; ~$0.0X per hourly run",
)
import numpy as np
rng = np.random.default_rng(3)
es = pd.read_csv(os.path.join(HERE, "out/event_study_rows.csv"))
drift = {}
for st, g in es.groupby("set"):
    for col in ["60", "180", "kick"]:
        gm = g.groupby("unique_id")[col].agg(["sum", "count"])
        b = [(lambda x: x["sum"].sum() / x["count"].sum())(gm.sample(len(gm), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
        drift[f"{st} | to {col}"] = dict(mean_c=float(g[col].mean()), ci=[float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))], n=int(g[col].notna().sum()))
summary["drift_cents_in_predicted_direction"] = drift
tr = pd.read_csv(os.path.join(HERE, "out/trading.csv"))
months = (s.kick_utc.max() - s.kick_utc.min()).days / 30.4 - 18 / 30.4   # minus the Sep 21-Oct 8 data gap
tr["trades_per_month"] = tr.trades / months
tr["usd_per_month_at_100"] = tr.trades_per_month * 100 * tr.ret_per_dollar
summary["trading"] = tr.round(4).to_dict("records")
summary["months_covered"] = months
E["month"] = pd.to_datetime(E.t_seen).dt.to_period("M").astype(str)
summary["events_by_month"] = E.groupby("month").size().to_dict()
summary["sonnet_signal_by_conf"] = E[E.sonnet_sig].groupby("conf").size().to_dict()
for f in ["placebo.json", "hindsight_check.json", "source_stats.json", "lineup_window.json"]:
    p = os.path.join(HERE, "out", f)
    if os.path.exists(p):
        summary[f[:-5]] = json.load(open(p))
json.dump(summary, open(os.path.join(HERE, "out/summary.json"), "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str)[:4000])
