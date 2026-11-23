"""Step 4b: reality check of TAKER entries priced at 'mid + half-spread'. For each signal, find the first real taker
BUY print of the share we want within 30 minutes and compare its price with the modelled entry price.
Grouped by book coherence (sum of the 3 Yes mids at the signal). Train + validation signals only.
Output out/entry_reality.csv"""
import os
import numpy as np, pandas as pd
from common import load_all, HERE

S, M = load_all()
S = S[(S.split != "test") & S.x0.between(0.15, 0.85) & S.fam.isin(["F1", "F2", "F3", "F4", "F6"])]
S = S[(S.fam != "F6") | (S.thr == 24.0)]
T = pd.read_parquet(os.path.join(HERE, "data", "trades.parquet"))
fetched = set(pd.read_parquet(os.path.join(HERE, "data", "trades_events.parquet")).event_id)
S = S[S.event_id.isin(fetched)]
T = T[T.event_id.isin(set(S.event_id))]
T["mn"] = T.ts / 60.0
by = {k: v for k, v in T.groupby(["event_id", "k"])}
S["hs"] = np.where(S.poly_volume < 1e4, 0.02, np.where(S.poly_volume < 5e4, 0.01, np.where(S.poly_volume < 2.5e5, 0.0075, 0.005)))
rows = []
for s in S.itertuples():
    t = by.get((s.event_id, s.k))
    px = np.nan
    if t is not None:
        a = s.t_min + 1
        tb = t.yes_buy.to_numpy() if s.dir > 0 else ~t.yes_buy.to_numpy()
        q = tb & (t.mn.to_numpy() > a) & (t.mn.to_numpy() <= a + 30)
        if q.any():
            p = t.yes_px.to_numpy()[np.where(q)[0][0]]
            px = p if s.dir > 0 else 1 - p
    rows.append((s.sid, s.fam, s.thr, s.psum, s.x0, s.hs, px))
R = pd.DataFrame(rows, columns=["sid", "fam", "thr", "psum", "x0", "hs", "px"])
R["book"] = pd.cut(R.psum, [0, 0.98, 1.05, 1.15, 9], labels=["<0.98", "0.98-1.05 (coherent)", "1.05-1.15", ">1.15"])
R["has_print"] = R.px.notna()
R["excess_c"] = (R.px - (R.x0 + R.hs)) * 100
out = R.groupby(["fam", "book"], observed=True).agg(signals=("sid", "size"), with_real_buy_30m=("has_print", "mean"),
                                                   median_excess_cents=("excess_c", "median"), mean_excess_cents=("excess_c", "mean"))
out.to_csv(os.path.join(HERE, "out", "entry_reality.csv"))
pd.set_option("display.width", 200)
print(out.round(3).to_string())
x = R[(R.fam == "F2") & (R.thr == 0.08)]
print("F2 >=8pt:", x.groupby("book", observed=True).agg(n=("sid", "size"), print=("has_print", "mean"), exc=("excess_c", "median")).round(3))
