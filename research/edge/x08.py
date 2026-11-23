import numpy as np, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 300)
exec(open("x07.py").read().split("S = sys.argv")[0])
g = pd.read_parquet("../lead_lag/data/games_meta.parquet").set_index("unique_id")
c = d[(d.mid >= .02) & (d.mid <= .10) & d.T0.isin([3, 6, 9, 12])]
dr = c[c.k == 1]
print(dr[["mid", "w_prob", "psum", "vol_before", "max_q", "won"]].describe().round(3).T.to_string())
print(dr.drop_duplicates("uid").join(g[["title", "series"]], on="uid", rsuffix="_g")[["title", "series", "mid", "w_prob", "psum", "won", "start"]].sample(12, random_state=1).to_string())
print(dr.drop_duplicates("uid").series.value_counts().head(8).to_dict())
for s in ("train", "val", "test"):
    for nm, x in (("draw", dr), ("team", c[c.k != 1])):
        print(s, nm, ev(x[x.split == s], 0))
