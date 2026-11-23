"""Step 8: print the markdown tables quoted in REPORT.md (out/tables.md)."""
import os, json
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(HERE, "out")
R = json.load(open(os.path.join(O, "results.json")))
L = []
nm = {"winner": "Winner", "poly": "Polymarket"}


def md(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(rows)


def pct(x):
    return f"{100 * x:.0f}%"


wf = pd.DataFrame(R["who_first"])
t = pd.DataFrame({"Major move on": wf.trigger.map(nm), "size": (wf.thr * 100).astype(int).astype(str) + "+ pts",
                  "moves (games)": wf.n.astype(str) + " (" + wf.games.astype(str) + ")",
                  "Winner first": wf["Winner first"], "same 15-min window": wf["same 15-min window"],
                  "Polymarket first": wf["Polymarket first"], "other venue never followed": wf["other never followed"]})
L += ["## who_first", md(t), ""]
t = pd.DataFrame({"Major move on": wf.trigger.map(nm), "size": (wf.thr * 100).astype(int).astype(str) + "+",
                  "Winner first": wf["chance_Winner first"].map(pct), "same window": wf["chance_same 15-min window"].map(pct),
                  "Polymarket first": wf["chance_Polymarket first"].map(pct), "never": wf["chance_other never followed"].map(pct)})
L += ["## chance", md(t), ""]
cf = pd.read_csv(os.path.join(O, "conditional_follow.csv"))
cf["Major move on"] = cf.trigger.map(nm); cf["size"] = (cf.thr * 100).astype(int).astype(str) + "+"
for c in ("share_of_moves", "follow_within_15m", "follow_within_60m", "follow_within_180m", "follow_within_360m", "follow_before_kick"):
    cf[c] = cf[c].map(pct)
L += ["## conditional_follow", md(cf[["kind", "Major move on", "size", "n", "share_of_moves", "follow_within_15m", "follow_within_60m",
                                    "follow_within_180m", "follow_within_360m", "follow_before_kick"]]), ""]
lt = pd.DataFrame(R["lead_times"])
lt["Major move on"] = lt.trigger.map(nm)
L += ["## lead_times", md(lt[["Major move on", "thr", "measure", "n", "q10", "q25", "q50", "q75", "q90", "share_within_15", "share_neg"]].round(2)), ""]
tr = pd.DataFrame(R["trading"])
_ci = lambda v, m: "[" + ", ".join(f"{m * float(x):.{1 if m == 100 else 2}f}" for x in v) + "]"
tr["gross pts"] = tr.gross_move_pts.round(2).astype(str) + " " + tr.gross_ci.map(lambda v: _ci(v, 1))
tr["ROI/$ (kickoff price)"] = (100 * tr.roi_clv).round(1).astype(str) + "% " + tr.roi_clv_ci.map(lambda v: _ci(v, 100))
tr["ROI/$ realised"] = (100 * tr.roi_realized).round(1).astype(str) + "% " + tr.roi_real_ci.map(lambda v: _ci(v, 100))
for c in ("roi_clv_no_fee", "roi_clv_no_cost"):
    tr[c] = (100 * tr[c]).round(1).astype(str) + "%"
L += ["## trading", md(tr[["cadence_min", "thr", "n", "games", "gross pts", "move_60m_pts", "move_360m_pts", "ROI/$ (kickoff price)",
                          "roi_clv_no_fee", "roi_clv_no_cost", "ROI/$ realised", "hit_rate"]].round(2)), ""]
rc = pd.DataFrame(R["trading_recent"])
rc["ROI/$"] = (100 * rc.roi_clv).round(1).astype(str) + "% " + rc.roi_clv_ci.map(lambda v: _ci(v, 100))
rc["ROI/$ gap>4"] = (100 * rc.roi_clv_gap4).round(1).astype(str) + "% " + rc.roi_clv_gap4_ci.map(lambda v: _ci(v, 100))
L += ["## trading recent (Apr-Nov 2026)", md(rc[["cadence_min", "thr", "n", "gross_move_pts", "ROI/$", "roi_real", "n_gap4", "ROI/$ gap>4"]].round(3)), ""]
bd = pd.read_csv(os.path.join(O, "breakdowns.csv"))
for v in ("winner", "poly"):
    for dim in ("tier", "ttk", "size", "lvl", "vol_bucket", "dow", "hour_il"):
        x = bd[(bd.trigger == v) & (bd.dim == dim)].copy()
        for c in ("winner_first", "same", "poly_first", "never", "gap_closing"):
            x[c] = x[c].map(pct)
        L += [f"## breakdown {v} {dim}", md(x.drop(columns=["trigger", "dim"]).round(2)), ""]
tb = pd.read_csv(os.path.join(O, "trading_breakdowns.csv"))
for dim in ("tier", "ttk", "vol_bucket", "gap_bucket", "month"):
    x = tb[tb.dim == dim].copy(); x["roi_clv"] = (100 * x.roi_clv).round(1); x["roi_real"] = (100 * x.roi_real).round(1)
    L += [f"## trading breakdown {dim}", md(x.drop(columns=["dim"]).round(2)), ""]
mo = pd.read_csv(os.path.join(O, "monthly.csv"))
L += ["## monthly", md(mo.round(3)), ""]
gc = pd.read_csv(os.path.join(O, "gap_closing.csv"))
L += ["## gap_closing", md(gc.round(3)), ""]
L += ["## xcorr", json.dumps(R["xcorr"], indent=1), "## ecm", json.dumps(R["ecm"], indent=1, default=str)]
L += ["## counts", json.dumps({k: R[k] for k in ("games_matched", "games_good", "events", "events_games", "events_dropped_not_persistent", "tiers", "vol_buckets")}, indent=1)]
open(os.path.join(O, "tables.md"), "w").write("\n".join(L))
print("\n".join(L))
