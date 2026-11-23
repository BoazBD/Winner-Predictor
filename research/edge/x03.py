import numpy as np, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 500)
t = pd.read_parquet("data/master.parquet", columns=["event_id","k","yes","q","size","ts","rel_min","won","start","poly_volume","tier","series"])
t = t[(t.rel_min < -5) & (t.rel_min > -72*60)]
t["split"] = np.where(t.start < "2026-07-01", "train", np.where(t.start < "2026-10-01", "val", "test"))
t["fee"] = 0.05 * t.q * (1 - t.q); t["cost"] = t.q + t.fee; t["ret"] = (t.won - t.cost) / t.cost
def boot(d, n=2000):
    g = d.groupby("event_id").ret.agg(["sum", "size"]).to_numpy(); r = np.random.default_rng(0)
    i = r.integers(0, len(g), (n, len(g))); e = g[i, 0].sum(1) / g[i, 1].sum(1); return np.percentile(e, [2.5, 97.5])
def rule(lo, hi, hmax, hmin=5/60, d=t):
    x = d[(d.q >= lo) & (d.q <= hi) & (d.rel_min >= -hmax*60) & (d.rel_min <= -hmin*60)]
    return x.sort_values("ts").drop_duplicates(["event_id", "k", "yes"])     # first chance per token
rows = []
for lo, hi in [(.80,.85),(.85,.90),(.90,.93),(.93,.96),(.96,.98),(.85,.97),(.88,.97),(.90,.98),(.03,.15),(.15,.30),(.30,.70),(.70,.85)]:
    for hmax in (3, 24, 72):
        b = rule(lo, hi, hmax)
        for s in ("train", "val", "test"):
            x = b[b.split == s]
            if len(x) < 20: continue
            ci = boot(x)
            rows.append((lo, hi, hmax, s, len(x), x.event_id.nunique(), x.q.mean(), x.won.mean(), x.ret.mean(), ci[0], ci[1]))
r = pd.DataFrame(rows, columns=["lo","hi","hmax","split","bets","games","q","won","ret","ci_lo","ci_hi"])
print(r.round(4).to_string())
b = rule(.85, .97, 24); b["type"] = np.where(b.yes, "Yes-fav", np.where(b.k == 1, "No-draw", "No-team"))
b["volb"] = pd.cut(b.poly_volume, [0, 1e4, 5e4, 2.5e5, 1e6, 1e12]); b["month"] = b.start.dt.to_period("M")
for c in ("type", "tier", "volb", "month"):
    print(b.groupby(c, observed=True).agg(bets=("ret","size"), games=("event_id","nunique"), q=("q","mean"), won=("won","mean"), ret=("ret","mean")).round(4).to_string())
