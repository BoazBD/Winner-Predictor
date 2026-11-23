"""All real Polymarket trades (pre-match, in-play, post-game) of the matched games, with the final result.
Each trade is expressed as 'a taker BOUGHT token X at price q' (a taker selling Yes at p == buying No at 1-p).
Columns: event_id, k (outcome 0/1/2 in Polymarket order), yes (bought token is the Yes of k), q, size (shares),
ts, rel_min (minutes since listed start), won (bought token paid 1), kick, vol, tier, series."""
import glob, json, numpy as np, pandas as pd
g = pd.read_parquet("../lead_lag/data/games_meta.parquet"); g = g[g.ok].copy()
tm = json.load(open("../scalp/data/tokmap.json"))
res = {}
for r in g.itertuples():
    op = [r.outcome_price0, r.outcome_price1, r.outcome_price2]
    try: op = [float(x) for x in op]
    except Exception: continue
    if sorted(op) != [0, 0, 1]: continue          # keep cleanly resolved games only
    for k in range(3):
        kk = (2 - k) if r.swapped else k           # tokmap k is in Winner orientation
        res[(r.event_id, kk)] = op[k]
t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("../scalp/cache/trades/*.parquet"))], ignore_index=True).drop_duplicates()
m = t.token.map(tm); t = t[m.notna()].copy(); m = m[m.notna()]
t["event_id"] = [x[0] for x in m]; t["k"] = [x[1] for x in m]; yes = np.array([x[2] for x in m]); st = np.array([x[3] for x in m])
ywon = pd.Series(list(zip(t.event_id, t.k))).map(res).values
keep = ~np.isnan(ywon.astype(float)); t = t[keep]; yes = yes[keep]; st = st[keep]; ywon = ywon[keep].astype(float)
buy = (t.side == "BUY").values
t["yes"] = np.where(buy, yes, ~yes)               # token the taker ended up long
t["q"] = np.where(buy, t.price, 1 - t.price)
t["won"] = np.where(t.yes, ywon, 1 - ywon)
t["rel_min"] = (t.ts.values - st) / 60
meta = g.set_index("event_id")[["start", "poly_volume", "tier", "series", "unique_id", "fees0"]]
t = t.join(meta, on="event_id")
t = t[["event_id", "unique_id", "k", "yes", "q", "size", "ts", "rel_min", "won", "start", "poly_volume", "tier", "series", "fees0"]]
t.to_parquet("data/master.parquet")
print(len(t), t.event_id.nunique(), "pre", (t.rel_min < 0).mean().round(3), "inplay<125", ((t.rel_min >= 0) & (t.rel_min < 125)).mean().round(3), "post", (t.rel_min >= 125).mean().round(3))
print(t.start.min(), t.start.max())
