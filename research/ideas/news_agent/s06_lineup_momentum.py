"""s06 - What is left after the lineup release for a fast (no-LLM) watcher?

Per league, locate the lineup spike minute L (max of 3-min smoothed mean |1-min change| of the favourite's
YES-implied trade price, searched in [-95,-40]). Then for every game and every outcome k:
  signal  = p(L+d) - p(L-3)   (the market's own initial reaction, d = detection delay 2/5/10/15/60 min)
  if |signal| >= thr: buy outcome k (if up) or the NO side (if down) at p(L+d) +/- half-spread, pay taker fee,
  sell at p(kick-3) at the bid, pay taker fee again.  Return per $ staked.
This is the best case for any news agent in the lineup window: it reacts to the same information instantly
(the price itself reveals it) with no LLM error. If this is negative, an LLM reading lineups later cannot win.
Half-spread = 0.5c (games here all have >$250k volume; see lead_lag/s00_spreads.py). Taker fee 0.05*p*(1-p).
95% CI by bootstrapping games. Outputs: out/lineup_spikes.csv, out/lineup_momentum.csv
"""
import os
import numpy as np, pandas as pd, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
HS = 0.005
rng = np.random.default_rng(0)
s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet"))
ids = ",".join(f"'{u}'" for u in s.unique_id)
t = duckdb.sql(f"""select unique_id, k, case when yes then q else 1-q end as py, rel_min
                   from '{R}/edge/data/master.parquet' where unique_id in ({ids}) and rel_min between -180 and 0""").df()
t["py"] = t.py.astype("float64").round(4)
t["m"] = np.floor(t.rel_min).astype(int)
P = t.groupby(["unique_id", "k", "m"]).py.median().unstack().reindex(columns=range(-180, 0)).ffill(axis=1)
P = P[P[-170].notna()]                                  # need a price before the window
lg = s.set_index("unique_id").league
# spike per league using the favourite rows
fav = P[-150].groupby(level=0).idxmax()
F = P.loc[fav.values]
F.index = F.index.get_level_values(0)
D = F.diff(axis=1).abs()
spk = []
for L_, g in D.groupby(lg.reindex(D.index)):
    pr = g.mean().rolling(3, center=True).mean()
    L = int(pr.loc[-95:-40].idxmax())
    spk.append(dict(league=L_, n=len(g), L=L, peak_c=pr[L] * 100, base_c=pr.loc[-170:-120].mean() * 100))
spk = pd.DataFrame(spk); spk.to_csv(os.path.join(HERE, "out/lineup_spikes.csv"), index=False); print(spk)
Lmap = spk.set_index("league").L
fee = lambda p: 0.05 * p * (1 - p)
rows = []
for d in [2, 5, 10, 15, 30, 60]:
    for thr in [0.01, 0.02, 0.03]:
        res = []
        for (u, k), r in P.iterrows():
            L = Lmap.get(lg.get(u))
            if L is None or L + d > -4:
                continue
            p0, pe, px = r[L - 3], r[L + d], r[-3]
            if np.isnan([p0, pe, px]).any():
                continue
            sig = pe - p0
            if abs(sig) < thr:
                continue
            if sig > 0:   # buy YES
                cost = pe + HS + fee(pe + HS); proceeds = px - HS - fee(px - HS)
            else:         # buy NO
                q, qx = 1 - pe, 1 - px
                cost = q + HS + fee(q + HS); proceeds = qx - HS - fee(qx - HS)
            res.append((u, (proceeds - cost) / cost, np.sign(sig) * (px - pe)))
        if not res:
            continue
        df = pd.DataFrame(res, columns=["u", "ret", "drift"])
        gm = df.groupby("u").ret.agg(["sum", "count"])
        bs = [gm.sample(len(gm), replace=True, random_state=int(rng.integers(1e9))).pipe(lambda x: x["sum"].sum() / x["count"].sum())
              for _ in range(1000)]
        rows.append(dict(delay_min=d, thr=thr, trades=len(df), games=len(gm), drift_after_entry_c=df.drift.mean() * 100,
                         ret_per_dollar=df.ret.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, "out/lineup_momentum.csv"), index=False)
print(out.round(4).to_string())
