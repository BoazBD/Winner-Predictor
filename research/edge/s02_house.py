"""'Be the house' on longshots: rest a limit SELL on a longshot's Yes share (= limit buy of its No) before kickoff.
Order placed at T0 hours before kickoff at a = mid + delta (rounded to 1c) if mid in [lo, hi]; filled only if a REAL
taker later bought that outcome at a price > a (through) or >= a (touch) before kickoff; held to the result.
P&L per share = a - won_yes ; capital per share = 1 - a ; maker fee = 0."""
import numpy as np, pandas as pd, os, sys
t = pd.read_parquet("data/master.parquet", columns=["event_id","unique_id","k","yes","q","size","ts","rel_min","won","start","poly_volume","tier"])
t = t[(t.rel_min < 0) & t.yes]                       # pre-match prints where a taker ended up long Yes of k
g = t.groupby("unique_id")
won = t.drop_duplicates(["unique_id", "k"]).set_index(["unique_id", "k"]).won
meta = t.drop_duplicates("unique_id").set_index("unique_id")[["start", "poly_volume", "tier", "event_id"]]
rows = []
for n, (uid, d) in enumerate(g):
    f = f"../lead_lag/data/series/{uid}.npz"
    if not os.path.exists(f): continue
    z = np.load(f); P = z["P"]; t0 = int(z["t0"]); kick = int(z["kick"])
    tm = d.ts.values / 60.0
    for T0 in (48, 24, 6, 1):
        i = kick - T0 * 60 - t0
        if i < 0 or i >= len(P): continue
        for k in range(3):
            m = float(P[i, k])
            if not (0.03 <= m <= 0.30): continue
            x = d[(d.k == k) & (tm > kick - T0 * 60) & (tm < kick)]
            for delta in (0.0, 0.01, 0.02):
                a = round(m + delta + 1e-9, 2)
                thr = x[x.q > a + 1e-9]; tch = x[x.q >= a - 1e-9]
                rows.append((uid, k, T0, delta, m, a, len(thr) > 0, float((thr["size"]).sum()), len(tch) > 0, float(tch["size"].sum()),
                             float(thr.rel_min.min()) if len(thr) else np.nan))
r = pd.DataFrame(rows, columns=["uid", "k", "T0", "delta", "mid", "a", "f_thr", "sz_thr", "f_tch", "sz_tch", "fill_rel"])
r["won"] = won.reindex(pd.MultiIndex.from_frame(r[["uid", "k"]])).values
r = r.join(meta, on="uid")
r.to_parquet("data/house.parquet"); print(len(r), r.uid.nunique())
