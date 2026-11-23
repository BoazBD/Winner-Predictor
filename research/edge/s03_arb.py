"""Cross-venue arbitrage: Winner (slow) vs Polymarket (fast), using REAL Polymarket prints.
A) bet outcome k at Winner odds o, and buy 'No k' on Polymarket at q (a real taker bought No k at q then):
   outlay 1 + o*(q+fee), payoff o whatever happens.   edge = o / (1 + o*(q+fee)) - 1
B) buy 'Yes k' on Polymarket at q and bet the two other outcomes at Winner: outlay q+fee+1/oa+1/ob per 1 payoff.
Winner odds must be the same in the snapshot before AND after the print (so they really were on offer)."""
import numpy as np, pandas as pd
t = pd.read_parquet("data/master.parquet", columns=["event_id","unique_id","k","yes","q","size","ts","rel_min","start","poly_volume","tier"])
t = t[(t.rel_min < -2)].copy(); t["t"] = pd.to_datetime(t.ts, unit="s").astype("datetime64[us]")
w = pd.read_parquet("../lead_lag/data/winner_snap.parquet"); w = w[w.unique_id.isin(t.unique_id.unique())].sort_values(["unique_id", "run_utc"])
for c in ("r1", "r2", "r3"): w["n" + c] = w.groupby("unique_id")[c].shift(-1)
w["nrun"] = w.groupby("unique_id").run_utc.shift(-1)
w = w.sort_values("run_utc"); t = t.sort_values("t")
m = pd.merge_asof(t, w, left_on="t", right_on="run_utc", by="unique_id", tolerance=pd.Timedelta("20min"))
m = m[m.r1.notna() & (m.nrun - m.run_utc < pd.Timedelta("20min")) & (m.r1 == m.nr1) & (m.r2 == m.nr2) & (m.r3 == m.nr3)]
R = m[["r1", "r2", "r3"]].to_numpy(); k = m.k.to_numpy(); o = R[np.arange(len(m)), k]
fee = 0.05 * m.q * (1 - m.q); c = (m.q + fee).to_numpy()
inv = 1 / R; oth = inv.sum(1) - inv[np.arange(len(m)), k]
m["o"] = o
m["edge"] = np.where(m.yes, 1 / (c + oth) - 1, o / (1 + o * c) - 1)
m["kind"] = np.where(m.yes, "B", "A")
m["outlay"] = np.where(m.yes, m["size"] * (c + oth), m["size"] * (1 / o + c))     # $ needed to mirror the whole print
m[["event_id","unique_id","k","yes","q","size","t","rel_min","start","poly_volume","tier","o","edge","kind","outlay","run_utc"]].to_parquet("data/arb.parquet")
print(len(m), "prints with live Winner odds;", m.unique_id.nunique(), "games")
for th in (0.0, 0.01, 0.02, 0.05):
    x = m[m.edge > th]
    print(th, "prints", len(x), "games", x.unique_id.nunique(), "kind", x.kind.value_counts().to_dict(), "outlay$", round(x.outlay.sum()), "profit$", round((x.outlay * x.edge).sum()), "median edge", round(x.edge.median(), 4))
