import numpy as np, pandas as pd, sys
pd.set_option("display.width", 250, "display.max_rows", 300)
d = pd.read_parquet("data/grid.parquet")
d["split"] = np.where(d.start < "2026-07-01", "train", np.where(d.start < "2026-10-01", "val", "test"))
d["base"] = np.ceil(d.mid * 100 - 1e-6) / 100
def ev(x, delta, fill="strict"):
    a = (x.base + delta).round(2); f = (x.max_q > a + 1e-9) if fill == "strict" else (x.max_q >= a - 1e-9)
    y = x[f]; a = a[f]
    if len(y) < 15: return dict(orders=len(x), fills=len(y))
    g = pd.DataFrame(dict(u=y.uid.values, p=(a - y.won).values, c=(1 - a).values)).groupby("u").sum().to_numpy()
    r = np.random.default_rng(0); i = r.integers(0, len(g), (600, len(g))); e = g[i, 0].sum(1) / g[i, 1].sum(1)
    return dict(orders=len(x), fills=len(y), fr=round(len(y) / len(x), 2), a=round(a.mean(), 3), won=round(y.won.mean(), 4), roi=round(g[:, 0].sum() / g[:, 1].sum(), 4), lo=round(np.percentile(e, 2.5), 4), hi=round(np.percentile(e, 97.5), 4),
                per_order=round(g[:, 0].sum() / len(x) * 100, 3))
S = sys.argv[1] if len(sys.argv) > 1 else "train"
x = d[d.split == S] if S != "all" else d
x["mb"] = pd.cut(x.mid, [.01, .02, .03, .04, .05, .06, .08, .10, .12, .15, .20, .25], include_lowest=True)
print("== by price bucket x delta (T0=6)")
for dl in (-0.01, 0, 0.01, 0.02):
    print("delta", dl); print(pd.DataFrame({str(k): ev(v, dl) for k, v in x[x.T0 == 6].groupby("mb", observed=True)}).T.to_string())
c = x[(x.mid >= .02) & (x.mid <= .10)]
print("== by T0 (mid 2-10c)"); 
for dl in (0, 0.01): print("delta", dl); print(pd.DataFrame({k: ev(v, dl) for k, v in c.groupby("T0")}).T.to_string())
c6 = c[c.T0.isin([3, 6, 9, 12])]
for name, key in [("type", np.where(c6.k == 1, "draw", "team")), ("tier", c6.tier), ("vol_before", pd.cut(c6.vol_before, [-1, 1000, 5000, 20000, 100000, 1e12]).astype(str)),
                  ("gap mid-winner", pd.cut(c6.mid - c6.w_prob, [-1, -.02, -.005, .005, .02, 1]).astype(str)), ("psum", pd.cut(c6.psum, [0, .99, 1.01, 1.03, 1.06, 9]).astype(str)),
                  ("touch-vs-strict n_after", pd.cut(c6.n_after, [-1, 0, 3, 20, 1e9]).astype(str))]:
    print("==", name, "(mid 2-10c, T0 3-12, delta 0)"); print(pd.DataFrame({k: ev(v, 0) for k, v in c6.groupby(key)}).T.to_string())
