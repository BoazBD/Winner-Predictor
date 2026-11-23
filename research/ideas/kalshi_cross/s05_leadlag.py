"""s05 - lead/lag: does the Kalshi book lead Polymarket prints, and is it tradable on Polymarket only?

Polymarket-only trading (Kalshi used purely as a public data signal - the only legal use for a non-US trader).
Per pre-kick Polymarket YES_j print at time t (price q = ask we could have paid):
  gap  = Kalshi mid_j (last candle <= t, two-sided, spread <= 3c, <= 30 min old) - q
Entry rule: buy YES_j on Poly at q (+ taker fee) when gap >= G   (Kalshi says Poly's ask is cheap).
Also the mirror: a Polymarket NO_j print at q is a YES_j-short at (1-q); enter when (1-q) - kalshi_mid_j >= G.
Exit (all before kickoff, no result risk) after H minutes (or at kick-5 min if earlier):
  sell = the first opposite-side Polymarket print at/after exit time within 30 min
         (for a YES_j holder: a NO_j print at q2 means a resting YES_j bid at 1-q2 -> we sell there, taker fee)
  If none prints -> marked as 'no exit' and valued at Kalshi bid_j at exit time minus 1c (pessimistic proxy).
Fees: Polymarket 0.05*p*(1-p) on entry and exit.
Baseline: the same exit rule for all prints (G = -inf) - shows the round-trip cost.
Parameters (G, H) chosen on train (< 2026-07-01); reported on val (Jul-Sep 2026) and test (Oct 1 - Nov 18 2026).
CIs: 95% bootstrap over games.
Outputs: data/leadlag_trades.parquet, out/leadlag_grid_train.csv, out/leadlag_result.json, out/leadlag_predict.csv
"""
import json
import numpy as np, pandas as pd

R4 = lambda x: np.round(np.asarray(x, dtype="float64"), 4)
fP = lambda p: 0.05 * p * (1 - p)
x = pd.read_parquet("data/arb_prints.parquet")
x = x[x.clean & x.kbid.notna() & (x.kask - x.kbid <= 0.03 + 5e-5) & (x.kbid > 0) & (x.kask < 1)].copy()
# orientation: long YES_j if yes print, short YES_j (= long NO_j) if no print
x["side"] = np.where(x.yes, 1, -1)
x["px_yes"] = R4(np.where(x.yes, x.q, 1 - x.q))           # YES_j-equivalent price we transact at
x["sig"] = x.side * (x.kmid_j - x.px_yes)                   # >0: Kalshi says our side is cheap on Poly

# opposite-side prints (exit liquidity), per (unique_id, j, yes)
m = pd.read_parquet("../../edge/data/master.parquet", columns=["unique_id", "k", "yes", "q", "ts", "rel_min"])
m = m[m.unique_id.isin(x.unique_id.unique()) & (m.rel_min < -1)]
sw = pd.read_parquet("data/matches.parquet", columns=["unique_id"]).merge(
    pd.read_parquet("../../lead_lag/data/games_meta.parquet", columns=["unique_id", "swapped"]), on="unique_id")
m = m.merge(sw, on="unique_id")
m["j"] = np.where(m.swapped & (m.k != 1), 2 - m.k, m.k)
m["q"] = R4(m.q)
out = []
for H in (15, 60, 180):
    e = x.copy()
    e["exit_ts"] = np.minimum(e.ts + H * 60, e.ts - (e.rel_min + 5) * 60)   # never later than kick-5min
    e["exit_ts"] = e.exit_ts.astype("int64")
    e = e[e.exit_ts > e.ts]
    e["opp_yes"] = ~e.yes
    e = e.sort_values("exit_ts")
    mm = m.rename(columns={"yes": "opp_yes", "q": "q2", "ts": "ts2", "rel_min": "rel2"}).sort_values("ts2")
    e = pd.merge_asof(e, mm[["unique_id", "j", "opp_yes", "q2", "ts2"]], left_on="exit_ts", right_on="ts2",
                      by=["unique_id", "j", "opp_yes"], direction="forward", tolerance=1800)
    # we hold token T (YES_j if yes else NO_j) bought at q; an opposite print at q2 => bid for T is 1-q2
    e["sell"] = 1 - e.q2
    e["exit_ok"] = e.q2.notna() & (e.ts2 < e.ts - (e.rel_min + 2) * 60)
    e["pnl"] = np.where(e.exit_ok, e.sell - fP(e.sell) - e.q - fP(e.q), np.nan)
    e["ret"] = e.pnl / (e.q + fP(e.q))
    e["H"] = H
    out.append(e[["unique_id", "split", "series", "j", "yes", "q", "size", "ts", "rel_min", "sig", "kmid_j", "px_yes",
                  "exit_ok", "pnl", "ret", "H"]])
T = pd.concat(out, ignore_index=True)
T.to_parquet("data/leadlag_trades.parquet")

# 1) predictive check: does the gap predict the Poly move?  (mid-to-mid style, before costs)
T["move"] = T.pnl + fP(T.q) + fP(1 - T.pnl)                    # rough gross move (fees added back)
pred = []
for (s, H), z in T[T.exit_ok].groupby(["split", "H"]):
    z = z.assign(b=pd.cut(z.sig, [-1, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 1]))
    g = z.groupby("b", observed=True).agg(n=("pnl", "size"), mean_pnl=("pnl", "mean"), mean_ret=("ret", "mean"))
    g["split"], g["H"] = s, H
    pred.append(g.reset_index())
pred = pd.concat(pred); pred["b"] = pred.b.astype(str)
pred.to_csv("out/leadlag_predict.csv", index=False)
print(pred[pred.split == "train"].to_string())


def boot(z, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = z.groupby("unique_id").agg(p=("pnl", "sum"), c=("q", "size"))
    if len(g) < 5:
        return (np.nan, np.nan)
    ids = rng.integers(0, len(g), (n, len(g)))
    P, C = g.p.to_numpy()[ids].sum(1), g.c.to_numpy()[ids].sum(1)
    return tuple(np.percentile(P / C, [2.5, 97.5]))


# 2) grid on train: one trade per (game, j, side) - first qualifying print - to avoid counting the same idea 100x
rows = []
for H in (15, 60, 180):
    for G in (0.01, 0.02, 0.03, 0.04, 0.06):
        for mx in (120, 1440):
            z = T[(T.H == H) & (T.sig >= G - 5e-5) & (-T.rel_min <= mx) & (T.q >= 0.1) & (T.q <= 0.9)]
            z = z.sort_values("ts").drop_duplicates(["unique_id", "j", "yes"])
            for s in ("train", "val", "test"):
                y = z[z.split == s]
                yo = y[y.exit_ok]
                rows.append(dict(H=H, G=G, max_min=mx, split=s, signals=len(y), exit_rate=y.exit_ok.mean(),
                                 games=y.unique_id.nunique(), mean_pnl=yo.pnl.mean(), mean_ret=yo.ret.mean(),
                                 hit=(yo.pnl > 0).mean(), ci=boot(yo)))
gr = pd.DataFrame(rows)
gr.to_csv("out/leadlag_grid.csv", index=False)
tr = gr[(gr.split == "train") & (gr.signals >= 30)].sort_values("mean_pnl", ascending=False)
print(tr.head(12).to_string())
best = tr.iloc[0]
sel = gr[(gr.H == best.H) & (gr.G == best.G) & (gr.max_min == best.max_min)]
print("chosen on train:\n", sel.to_string())
json.dump(dict(chosen=dict(H=int(best.H), G=float(best.G), max_min=int(best.max_min)),
               results=sel.to_dict("records")), open("out/leadlag_result.json", "w"), indent=1, default=str)
