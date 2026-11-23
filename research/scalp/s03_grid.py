"""Step 3: grid search on TRAIN (kickoff < 2026-07-01); report every config's VALIDATION (Jul-Sep 2026) result.
The TEST period (kickoff >= 2026-10-01) is dropped before anything is computed (see s05_final.py).
Usage: python s03_grid.py mid      (midpoint fills, c=1 'book moved through our price')
       python s03_grid.py trades   (real trade prints, mode 'through' = a print strictly better than our limit)
Selection metric: game-clustered lower bound of return per $ staked: mean(pnl/entry price) - 2*SE,
needs >= 150 trades in >= 100 games on train. Finalist per family = best train lower bound.
Outputs out/grid_<src>.csv, out/finalists_<src>.csv, out/n_configs_<src>.txt
"""
import os, sys, itertools
import numpy as np, pandas as pd
from common import load_all, EXITS, HERE

SRC = sys.argv[1] if len(sys.argv) > 1 else "mid"
S, M = load_all()
if SRC == "mid":
    M = M[M.c == 1].drop(columns="uid")
else:
    M = pd.read_parquet(os.path.join(HERE, "data", "sims_trades.parquet"))
    M = M[M["mode"] == "through"]
S = S[S.split != "test"]                                   # test never enters the search
S["fav_ok"] = (S.x0 >= 0.15) & (S.x0 <= 0.85)
S["mid_px"] = (S.x0 >= 0.25) & (S.x0 <= 0.75)
S["far"] = S.htk > 12
S["vol_mid"] = (S.series_vol >= 1e4) & (S.series_vol <= 2.5e5)
S["nontop"] = S.tier != "top"
S["coh"] = S.psum.between(0.98, 1.05)
FILTERS = {
    "time": {"all": None, ">12h": "far", "<=12h": "~far"},
    "price": {"15-85c": "fav_ok", "25-75c": "mid_px"},
    "vol": {"all": None, "series 10k-250k": "vol_mid"},
    "tier": {"all": None, "non-top": "nontop"},
    "book": {"all": None, "coherent": "coh"},
}
FCOLS = ["fav_ok", "mid_px", "far", "vol_mid", "nontop", "coh"]
X = M.merge(S[["sid", "uid", "fam", "thr", "k", "dir", "split", "x0"] + FCOLS], on="sid", how="inner")
X = X[X.fav_ok & ~((X.fam == "F6") & (X.thr != 24.0))]
for e in EXITS:
    X[e.replace("pnl", "ret")] = X[e] / X.pe


def score(df, col, nmin=150, gmin=100):
    g = df.groupby("uid")[col].agg(["sum", "size"])
    n = g["size"].sum()
    if n < nmin or len(g) < gmin:
        return None
    m = g["sum"].sum() / n
    r = g["sum"] - m * g["size"]
    se = np.sqrt((r ** 2).sum()) / n
    return m, se, n, len(g)


rows = []
n_cfg = 0
for (fam, thr, ent, d), A in X.groupby(["fam", "thr", "ent", "d"]):
    extra = [None]
    if fam == "F6":
        extra += [(kk, dd) for kk in (0, 1, 2) for dd in (1, -1)]
    fl = {c: A[c].to_numpy() for c in FCOLS}
    kk_, dd_ = A.k.to_numpy(), A.dir.to_numpy()
    for ex, tf, pf, vf, trf, bf, xt in itertools.product(EXITS, *[FILTERS[f] for f in ("time", "price", "vol", "tier", "book")], extra):
        n_cfg += 1
        q = np.ones(len(A), bool)
        for name, val in (("time", tf), ("price", pf), ("vol", vf), ("tier", trf), ("book", bf)):
            col = FILTERS[name][val]
            if col:
                q &= ~fl[col.strip("~")] if col.startswith("~") else fl[col]
        if xt:
            q &= (kk_ == xt[0]) & (dd_ == xt[1])
        B = A[q]
        base = dict(fam=fam, thr=thr, ent=ent, d=d, exit=ex, time=tf, price=pf, vol=vf, tier=trf, book=bf, sub=str(xt))
        for sp, nmin, gmin in (("train", 150, 100), ("val", 30, 20)):
            Fs = B[(B.split == sp)]
            sc = score(Fs, ex.replace("pnl", "ret"), nmin, gmin) if len(Fs) else None
            if sc is None:
                rows.append(dict(base, split=sp, n=len(Fs)))
                continue
            m, se, n, ng = sc
            rows.append(dict(base, split=sp, n=n, games=ng, ret=m, se=se, lb=m - 2 * se,
                             hit=Fs.hit.mean(), cents=Fs[ex].mean() * 100))
print("configs", n_cfg, flush=True)
G = pd.DataFrame(rows)
open(os.path.join(HERE, "out", f"n_configs_{SRC}.txt"), "w").write(str(n_cfg))
key = ["fam", "thr", "ent", "d", "exit", "time", "price", "vol", "tier", "book", "sub"]
W = G.pivot_table(index=key, columns="split", values=["ret", "lb", "n", "hit", "cents", "games"], aggfunc="first")
W.columns = [f"{a}_{b}" for a, b in W.columns]
W = W.reset_index()
W.to_csv(os.path.join(HERE, "out", f"grid_{SRC}.csv"), index=False)
fin = W.dropna(subset=["lb_train"]).sort_values("lb_train", ascending=False).groupby("fam").head(1)
fin.to_csv(os.path.join(HERE, "out", f"finalists_{SRC}.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(fin[key + ["n_train", "ret_train", "lb_train", "hit_train", "n_val", "ret_val", "lb_val", "hit_val"]].to_string())
top = W.dropna(subset=["lb_train"]).sort_values("lb_train", ascending=False).groupby("fam").head(8)
print(top[key + ["n_train", "ret_train", "lb_train", "hit_train", "n_val", "ret_val", "lb_val"]].to_string())
# how well does train rank carry to validation? (rank correlation over configs with both)
both = W.dropna(subset=["ret_train", "ret_val"])
print("configs with train & val:", len(both), " Spearman(train ret, val ret) =", round(both.ret_train.corr(both.ret_val, method="spearman"), 3))
print("share of configs positive on val among train-lb>0:", round(float((both[both.lb_train > 0].ret_val > 0).mean()), 3),
      "n =", int((both.lb_train > 0).sum()))
