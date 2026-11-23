"""s03: rule search on TRAIN only (kickoff < 2026-07-01), honest 'through' fills; validation printed alongside.

Rule = refresh R x depth d x side filter x liquidity filter (usd printed on the game in the 6 h before the fill, known
live from the public trade feed) x price band of our token at the order time x time-to-kick window x Winner filter
(gap = Winner fair of our token - our buy price L; 'away from fair' = gap > 0) x exit (target T, deadline H; H='hold'
is excluded from the search because the owner wants no result risk).
Metric per rule: return per $ staked = sum(pnl per share)/sum(L) (1 share per fill). Selection score = game-clustered
lower bound mean_ret - 2*SE (SE from per-game sums, ratio estimator), requires >= 150 fills in >= 100 games on train.
Outputs: out/grid.csv (every rule: train n/ret/lb, val n/ret), out/n_rules.txt, out/top_rules.csv
"""
import os, itertools
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
VAL0 = pd.Timestamp("2026-07-01").timestamp(); TEST0 = pd.Timestamp("2026-10-01").timestamp()
TEST1 = pd.Timestamp("2026-11-19").timestamp()

SIDES = {"both": None, "short_k": -1, "long_k": 1}
LIQ = (0, 100, 1000, 5000)
BAND = {"08-92": (0.08, 0.92), "15-85": (0.15, 0.85), "25-75": (0.25, 0.75)}
TIME = {"all": (0, 10 ** 6), "<=12h": (0, 720), ">12h": (720, 10 ** 6)}
WIN = {"none": -9, "gap>=0": 0.0, "gap>=3c": 0.03, "gap>=6c": 0.06, "gap>=10c": 0.10}
# exits: limit-sell target x (240-min deadline then taker at mid-hs [unverified price], or VERIFIED liquidation at
# kick 'vk' = 1c under the last real bid-hit print in the final 2 h, else hold to result)
EXITS = [f"{t}_{h}" for t in ("tmid", "t1", "t2", "t3", "tw") for h in ("h240", "vk")]


def load():
    f = pd.read_parquet(os.path.join(D, "fills.parquet"))
    f = f[f["mode"] == "through"].drop(columns="mode")
    f["per"] = np.where(f.kick_ts < VAL0, "train", np.where(f.kick_ts < TEST0, "val",
                                                            np.where(f.kick_ts < TEST1, "test", "after")))
    f["gap"] = (f.wf - f.L).astype(float)
    f["g"] = f.g.fillna(0.0)          # 0 = 'mid' family (L = c - d); >0 = anchored L = min(c-d, Winner fair - g)
    return f


def stats(pnl, stake, gid, ng):
    """ratio mean and clustered SE over games (gid = factorized game ids, ng games)."""
    sp = np.bincount(gid, pnl, ng); ss = np.bincount(gid, stake, ng)
    used = ss > 0
    sp, ss = sp[used], ss[used]
    G = len(sp)
    if G < 2:
        return np.nan, np.nan, G
    r = sp.sum() / ss.sum()
    e = sp - r * ss
    se = np.sqrt(G / (G - 1) * (e ** 2).sum()) / ss.sum()
    return r, se, G


def masks(x):
    for (sn, sv), lq, (bn, (b0, b1)), (tn, (m0, m1)), (wn, wv) in itertools.product(
            SIDES.items(), LIQ, BAND.items(), TIME.items(), WIN.items()):
        m = (x.liq6.to_numpy() >= lq) & (x.c.to_numpy() >= b0) & (x.c.to_numpy() <= b1) \
            & (x.mtk.to_numpy() > m0) & (x.mtk.to_numpy() <= m1)
        if sv is not None:
            m &= x.s.to_numpy() == sv
        if wv > -1:
            m &= np.nan_to_num(x.gap.to_numpy(), nan=-1) >= wv
        yield dict(side=sn, liq=lq, band=bn, time=tn, win=wn), m


def main():
    f = load()
    rows = []
    for (R, d, gq), x in f[f.per.isin(["train", "val"])].groupby(["R", "d", "g"]):
        tr = x[x.per == "train"]; va = x[x.per == "val"]
        gtr, ugtr = pd.factorize(tr.uid); gva, ugva = pd.factorize(va.uid)
        mtr = list(masks(tr)); mva = list(masks(va))
        for (spec, a), (_, b) in zip(mtr, mva):
            if a.sum() < 150:
                continue
            for e in EXITS:
                r, se, G = stats(tr[e].to_numpy(float)[a], tr.L.to_numpy(float)[a], gtr[a], len(ugtr))
                if G < 100:
                    continue
                rv, sev, Gv = stats(va[e].to_numpy(float)[b], va.L.to_numpy(float)[b], gva[b], len(ugva)) \
                    if b.sum() > 1 else (np.nan, np.nan, 0)
                rows.append(dict(R=R, d=d, g=gq, **spec, exit=e, n_tr=int(a.sum()), g_tr=G, ret_tr=r, lb_tr=r - 2 * se,
                                 n_va=int(b.sum()), ret_va=rv, lb_va=rv - 2 * sev if Gv > 1 else np.nan))
    g = pd.DataFrame(rows).sort_values("lb_tr", ascending=False)
    n_total = len(f.groupby(["R", "d", "g"])) * len(SIDES) * len(LIQ) * len(BAND) * len(TIME) * len(WIN) * len(EXITS)
    g.to_csv(os.path.join(OUT, "grid.csv"), index=False)
    open(os.path.join(OUT, "n_rules.txt"), "w").write(
        f"rules enumerated {n_total}; with >=150 train fills in >=100 games: {len(g)}\n")
    top = g.head(30)
    top.to_csv(os.path.join(OUT, "top_rules.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(open(os.path.join(OUT, "n_rules.txt")).read())
    print(top.round(4).to_string())
    print("share of evaluated rules with train ret>0:", round(float((g.ret_tr > 0).mean()), 3),
          " lb>0:", round(float((g.lb_tr > 0).mean()), 3))
    print("of rules with train lb>0, val ret>0 share:", round(float((g[g.lb_tr > 0].ret_va > 0).mean()), 3))
    print("train-ret vs val-ret spearman over rules:", round(float(g.ret_tr.corr(g.ret_va, method="spearman")), 3))


if __name__ == "__main__":
    main()
