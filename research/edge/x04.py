import numpy as np, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 500)
r = pd.read_parquet("data/house.parquet")
r["split"] = np.where(r.start < "2026-07-01", "train", np.where(r.start < "2026-10-01", "val", "test"))
r["mb"] = pd.cut(r.mid, [0.03, 0.08, 0.15, 0.22, 0.30], include_lowest=True)
def stat(x):
    # return per $ at risk: sum(a - won) / sum(1 - a), one share per filled order
    if len(x) == 0: return pd.Series(dict(n=0))
    g = pd.DataFrame(dict(u=x.uid.values, p=(x.a - x.won).values, c=(1 - x.a).values)).groupby("u").sum().to_numpy()
    rng = np.random.default_rng(0); i = rng.integers(0, len(g), (1000, len(g))); e = g[i, 0].sum(1) / g[i, 1].sum(1)
    return pd.Series(dict(n=len(x), games=len(g), a=x.a.mean(), won=x.won.mean(), ret=g[:, 0].sum() / g[:, 1].sum(), lo=np.percentile(e, 2.5), hi=np.percentile(e, 97.5)))
for fill in ("f_thr", "f_tch"):
    print("=====", fill)
    o = r.groupby(["T0", "delta", "mb", "split"], observed=True).apply(lambda x: pd.concat([pd.Series(dict(orders=len(x), fillrate=x[fill].mean())), stat(x[x[fill] & x.won.notna()])]))
    print(o[o.index.get_level_values("delta").isin([0.0, 0.01])].round(4).to_string())
