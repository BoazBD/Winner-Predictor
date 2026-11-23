import numpy as np, pandas as pd, glob, os
pd.set_option("display.width", 250, "display.max_rows", 200)
t = pd.read_parquet("data/master.parquet", columns=["event_id","q","size","rel_min","won","start"])
p = t[(t.rel_min >= 115) & (t.q <= 0.02)].copy()          # taker dumped a ~sure winner to a resting bid at s = 1-q
p["s"] = 1 - p.q; p["usd"] = p["size"] * p.s; p["pnl"] = p["size"] * ((1 - p.won) - p.s)
p["sb"] = pd.cut(p.s, [0.98, 0.99, 0.995, 0.9975, 0.9985, 1.0]); p["tb"] = pd.cut(p.rel_min, [115, 125, 135, 150, 180, 360, 1e9])
p["split"] = np.where(p.start < "2026-07-01", "train", np.where(p.start < "2026-10-01", "val", "test"))
a = p.groupby(["tb", "sb"], observed=True).agg(n=("usd", "size"), games=("event_id", "nunique"), usd=("usd", "sum"), pnl=("pnl", "sum"), lost_trades=("won", "sum"), lost_usd=("usd", lambda x: x[p.loc[x.index, "won"] == 1].sum()))
a["ret_bp"] = 1e4 * a.pnl / a.usd; print(a.round(1).to_string())
x = p[(p.rel_min >= 135)]
print(x.groupby("split").agg(usd=("usd", "sum"), pnl=("pnl", "sum"), lostgames=("event_id", lambda e: e[x.loc[e.index, "won"] == 1].nunique()), games=("event_id", "nunique")))
days = (t.start.max() - pd.Timestamp("2025-09-01")).days; print("usd/day since Sep25 (>=135m):", x[x.start >= "2025-09-01"].usd.sum() / days)
# S4 momentum on mids
rows = []
fs = sorted(glob.glob("../lead_lag/data/series/*.npz"))[::3]
for f in fs:
    z = np.load(f); P = z["P"]; n = len(P)
    for back in (1440, 720, 360, 180, 60):
        i = n - back
        if i - 180 < 0 or np.isnan(P[i - 180]).any(): continue
        rows.append(np.c_[np.full(3, back), P[i] - P[i - 60], P[i] - P[i - 180], P[min(i + 60, n - 1)] - P[i], P[n - 1] - P[i], P[i]])
r = pd.DataFrame(np.vstack(rows), columns=["back", "d60", "d180", "f60", "fk", "p"]); r = r[(r.p > .1) & (r.p < .9)]
for c in ("d60", "d180"):
    r["b"] = pd.cut(r[c], [-1, -.05, -.03, -.015, -.005, .005, .015, .03, .05, 1])
    print(c); print(r.groupby("b", observed=True).agg(n=("fk", "size"), f60=("f60", "mean"), fk=("fk", "mean")).round(4).T.to_string())
