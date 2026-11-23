import numpy as np, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 500)
t = pd.read_parquet("data/master.parquet")
t["fee"] = 0.05 * t.q * (1 - t.q); t["cost"] = t.q + t.fee
t["ret"] = (t.won - t.cost) / t.cost; t["usd"] = t["size"] * t.q
t["split"] = np.where(t.start < "2026-07-01", "train", np.where(t.start < "2026-10-01", "val", "test"))
t["phase"] = pd.cut(t.rel_min, [-1e9, -1440, -180, -5, 0, 50, 65, 100, 125, 180, 1e9], labels=["pre>24h", "pre3-24h", "pre<3h", "pre<5m", "1st half", "HT", "2nd half", "end~", "post<3h", "post>3h"])
t["pb"] = pd.cut(t.q, [0, .05, .15, .3, .5, .7, .85, .93, .97, .99, .995, .9985, 1.0])
tr = t[t.split == "train"]
def tab(d):
    a = d.groupby(["phase", "pb"], observed=True).agg(n=("ret", "size"), games=("event_id", "nunique"), usd=("usd", "sum"), q=("q", "mean"), won=("won", "mean"), ret=("ret", "mean"))
    a["ret_usd"] = d.assign(x=d.ret * d.usd * (d.cost / d.q)).groupby(["phase", "pb"], observed=True).x.sum() / d.assign(c=d["size"] * d.cost).groupby(["phase", "pb"], observed=True).c.sum()
    return a
print(tab(tr).round(4).to_string())
