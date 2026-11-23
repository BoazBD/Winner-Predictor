"""s04: pick the final rule(s) on TRAIN only with a money criterion, then look at validation (no test here).

Realistic size: our order is S shares; at a fill we get q = min(S, shares that printed strictly through our level in
the 60 s after the first through-print) - those sellers would have hit our better bid first. S = 1000 shares.
Rule space (all 'mid' family, plus the Winner-anchored family from s02):
  R (refresh min) x d (depth below mid) x Winner gap filter x price band x side x time-to-kick x liquidity before x exit,
  exits restricted to VERIFIED prices: limit sell at target (pre-sweep level 'tmid', Winner fair 'tw', or L+3c) and,
  if unfilled, verified liquidation before kick ('vk'); also the 240-min deadline variant (taker at mid-hs) for info.
Selection: maximise train $/month subject to train $-weighted clustered lower bound (ret - 2 SE) > 0 and >= 100 fills
in >= 80 games.  Done separately for R=1 (continuous bot) and R=60 (hourly bot).
Outputs: out/select_grid.csv, out/selected.json
"""
import os, json, itertools
import numpy as np, pandas as pd
from s03_grid import load, OUT

S = 1000
MONTHS = {"train": 10.0, "val": 3.0, "test": 1.6}     # Sep 2025-Jun 2026, Jul-Sep 2026, Oct 1-Nov 18 2026
DS = (0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
GAP = {"none": -9, "gap>=0": 0.0, "gap>=3c": 0.03, "gap>=6c": 0.06, "gap>=10c": 0.10}
BAND = {"08-92": (0.08, 0.92), "15-85": (0.15, 0.85), "25-75": (0.25, 0.75)}
SIDE = {"both": 0, "short_k": -1, "long_k": 1}
TIME = {"all": (0, 10 ** 6), "<=12h": (0, 720), ">12h": (720, 10 ** 6)}
LIQ = (0, 100, 1000)
EXITS = ("tmid_vk", "tw_vk", "t3_vk", "tmid_h240", "tw_h240")


def wstats(pnl, stake, gid, ng):
    sp = np.bincount(gid, pnl, ng); ss = np.bincount(gid, stake, ng)
    u = ss > 0; sp, ss = sp[u], ss[u]; G = len(sp)
    if G < 2 or ss.sum() <= 0:
        return np.nan, np.nan, G
    r = sp.sum() / ss.sum(); e = sp - r * ss
    return r, np.sqrt(G / (G - 1) * (e ** 2).sum()) / ss.sum(), G


def main():
    f = load()
    f = f[f.per.isin(["train", "val"])]
    f["q"] = np.minimum(S, f.sh_thr.astype(float))
    rows = []
    for (R, d, g), x in f.groupby(["R", "d", "g"]):
        if g == 0 and d not in DS:
            continue
        gid, ug = pd.factorize(x.uid)
        isv = (x.per == "val").to_numpy()
        c = x.c.to_numpy(); gap = np.nan_to_num(x.gap.to_numpy(), nan=-1); s = x.s.to_numpy(); mtk = x.mtk.to_numpy()
        liq = x.liq6.to_numpy(); q = x.q.to_numpy(); L = x.L.to_numpy()
        for (gn, gv), (bn, (b0, b1)), (sn, sv), (tn, (m0, m1)), lq in itertools.product(
                GAP.items(), BAND.items(), SIDE.items(), TIME.items(), LIQ):
            if g > 0 and gn != "none":
                continue
            m = (gap >= gv) & (c >= b0) & (c <= b1) & (mtk > m0) & (mtk <= m1) & (liq >= lq)
            if sv:
                m &= s == sv
            mt, mv = m & ~isv, m & isv
            if mt.sum() < 100:
                continue
            for e in EXITS:
                p = x[e].to_numpy(float)
                r, se, G = wstats((p * q)[mt], (L * q)[mt], gid[mt], len(ug))
                if G < 80:
                    continue
                rv, sev, Gv = wstats((p * q)[mv], (L * q)[mv], gid[mv], len(ug)) if mv.sum() > 1 else (np.nan,) * 3
                rows.append(dict(fam="anch" if g > 0 else "mid", R=R, d=d, g=g, gap=gn, band=bn, side=sn, time=tn,
                                 liq=lq, exit=e, n_tr=int(mt.sum()), ret_tr=r, lb_tr=r - 2 * se,
                                 usdm_tr=float((p * q)[mt].sum()) / MONTHS["train"],
                                 n_va=int(mv.sum()), ret_va=rv, usdm_va=float((p * q)[mv].sum()) / MONTHS["val"]))
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(OUT, "select_grid.csv"), index=False)
    sel = {}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("rules evaluated:", len(g))
    for R in (1, 60):
        z = g[(g.R == R) & (g.lb_tr > 0) & g.exit.str.endswith("_vk")].sort_values("usdm_tr", ascending=False)
        print(f"\nR={R}: top by train $/month (verified exits), n candidates {len(z)}")
        print(z.head(12).round(4).to_string())
        sel[f"R{R}"] = z.iloc[0].to_dict()
        print("val $/month of top-50 train rules: median", round(float(z.head(50).usdm_va.median()), 1),
              " share>0", round(float((z.head(50).usdm_va > 0).mean()), 2))
    print("\nshare of all rules with val $/month > 0 among train-lb>0 rules:",
          round(float((g[g.lb_tr > 0].usdm_va > 0).mean()), 3), " vs among train-lb<=0:",
          round(float((g[g.lb_tr <= 0].usdm_va > 0).mean()), 3))
    json.dump(sel, open(os.path.join(OUT, "selected.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
