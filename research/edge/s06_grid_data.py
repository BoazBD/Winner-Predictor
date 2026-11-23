"""Candidate longshot orders on a fine grid. One row per (game, outcome, T0 hours before kickoff) with mid in [1c, 25c]:
mid, Winner's margin-free probability at that time, sum of the 3 mids, $ traded on the event before T0, and the highest
price at which a real taker bought that outcome between T0 and kickoff (max_q) -> fill test for any order price a:
strict fill = max_q > a, touch fill = max_q >= a.  Result from Polymarket's resolution."""
import numpy as np, pandas as pd, os
g = pd.read_parquet("../lead_lag/data/games_meta.parquet"); g = g[g.ok].copy()
fetched = set(pd.read_parquet("../scalp/data/trades_events.parquet").event_id)
g = g[g.event_id.isin(fetched)]
t = pd.read_parquet("data/master.parquet", columns=["unique_id","k","yes","q","size","ts","rel_min"])
t = t[t.rel_min < 0]; t["m"] = t.ts / 60.0; t["usd"] = t.q * t["size"]
T = {u: (d.m.values, d.k.values, d.yes.values, d.q.values, d.usd.values) for u, d in t.groupby("unique_id")}
rows = []
for r in g.itertuples():
    try: op = [float(r.outcome_price0), float(r.outcome_price1), float(r.outcome_price2)]
    except Exception: continue
    if sorted(op) != [0, 0, 1]: continue
    f = f"../lead_lag/data/series/{r.unique_id}.npz"
    if not os.path.exists(f): continue
    z = np.load(f); P = z["P"]; t0 = int(z["t0"]); kick = int(z["kick"]); wt = z["wt"]; Q = z["Q"]
    tr = T.get(r.unique_id)
    for T0 in (72, 48, 24, 12, 9, 6, 3, 1):
        i = kick - T0 * 60 - t0
        if i < 0 or i >= len(P) or np.isnan(P[i]).any(): continue
        now = kick - T0 * 60; j = np.searchsorted(wt, now) - 1
        for k in range(3):
            m = float(P[i, k])
            if not (0.01 <= m <= 0.25): continue
            kk = (2 - k) if r.swapped else k
            mx = np.nan; vol = 0.0; n_after = 0
            if tr is not None:
                a = (tr[0] > now) & (tr[0] < kick) & (tr[1] == k) & tr[2]
                if a.any(): mx = float(tr[3][a].max()); n_after = int(a.sum())
                vol = float(tr[4][tr[0] <= now].sum())
            rows.append((r.unique_id, k, T0, m, float(Q[j, k]) if j >= 0 else np.nan, float(P[i].sum()), vol, mx, n_after, op[kk], r.start, r.tier, r.series, r.poly_volume))
d = pd.DataFrame(rows, columns=["uid", "k", "T0", "mid", "w_prob", "psum", "vol_before", "max_q", "n_after", "won", "start", "tier", "series", "final_vol"])
d.to_parquet("data/grid.parquet"); print(len(d), d.uid.nunique(), d.groupby("T0").size().to_dict())
