import numpy as np, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 300)
t = pd.read_parquet("data/master.parquet", columns=["event_id","q","size","rel_min","won","start"])
t["split"] = np.where(t.start < "2026-07-01", "train", np.where(t.start < "2026-10-01", "val", "test"))
t["phase"] = pd.cut(t.rel_min, [-4320, -5, 100, 135, 1e9], labels=["pre-match", "in-play", "late 100-135m", "post 135m+"])
t["pb"] = pd.cut(t.q, [0, .01, .02, .05, .08, .15, .85, .95, .99, 1.0])
t["sz"] = t["size"].clip(upper=2000)                 # cap a print at 2000 shares so whales don't dominate
t["mp"] = t.sz * (t.q - t.won); t["mc"] = t.sz * (1 - t.q)      # maker profit / maker capital (it holds the other side)
def f(x):
    g = x.groupby("event_id")[["mp", "mc"]].sum().to_numpy()
    if len(g) < 30: return pd.Series(dict(games=len(g)))
    r = np.random.default_rng(0); i = r.integers(0, len(g), (500, len(g))); e = g[i, 0].sum(1) / g[i, 1].sum(1)
    return pd.Series(dict(games=len(g), cap_usd=g[:, 1].sum(), roi=g[:, 0].sum() / g[:, 1].sum(), lo=np.percentile(e, 2.5), hi=np.percentile(e, 97.5), taker_q=np.average(x.q, weights=x.sz), won=np.average(x.won, weights=x.sz)))
o = t.dropna(subset=["phase"]).groupby(["phase", "pb", "split"], observed=True).apply(f).unstack()
print(o.round(4).to_string())
