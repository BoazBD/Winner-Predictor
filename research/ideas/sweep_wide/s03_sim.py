"""s03: replay sweep_v1's rules (without the Winner check) on downloaded markets.
Per market and side s (+1 = token0, -1 = token1), every 5 min from kick-3h to kick-10min: c = token price 2 min ago
(prices-history), band 8-92c, two bids L = floor(min(c - dm, c (1 - f))) for A (15c, 40%) and B (20c, 50%), L >= 2c,
each S shares, live for 5 min. Fill = a maker bid on our token hit strictly below L (q = shares through L in that tx,
cap S). Exit: sell at T = max(floor(c), L + 1c) to real buyers above T until kick-5min (volume shared between our
positions in fill order); rest sold 1c under the last bid-hit print of the final 2 h, fee 0.05 p(1-p); else held to
the result. Cooldown 60 min per (side, level).
Usage: python s03_sim.py <sample.parquet> <out.parquet> [S]
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
LEV = {"A": (0.15, 0.40), "B": (0.20, 0.50)}; TOL = 5e-5


def fl(v):
    return np.floor(v * 100 + 1e-6) / 100


def sim(row, S):
    f = os.path.join(C, f"{row.cond}.npz")
    if not os.path.exists(f):
        return []
    z = np.load(f); k0 = int(row.kick_ts)
    if not int(z["ok"]) or not len(z["pt"]):
        return []
    ts, ypx, lg, sz, tx, pt, pp = z["ts"], z["ypx"], z["long"], z["size"].astype(float), z["tx"], z["pt"], z["pp"]
    o = np.argsort(ts, kind="stable"); ts, ypx, lg, sz, tx = ts[o], ypx[o], lg[o], sz[o], tx[o]
    out = []
    for s in (1, -1):
        px = ypx if s > 0 else np.round(1 - ypx, 4)
        bid = lg if s > 0 else ~lg
        res = row.res0 if s > 0 else row.res1
        cool = {}; used = {}
        for t in range(k0 - 180 * 60, k0 - 10 * 60, 300):
            i = np.searchsorted(pt, t - 120, side="right") - 1
            if i < 0 or pt[i] < t - 1800:
                continue
            c = round(float(pp[i]) if s > 0 else 1 - float(pp[i]), 4)
            if not 0.08 <= c <= 0.92:
                continue
            for lv, (dm, fr) in LEV.items():
                L = float(fl(min(c - dm, c * (1 - fr))))
                if L < 0.02 or cool.get(lv, 0) > t:
                    continue
                w = bid & (px < L - TOL) & (ts >= t) & (ts < min(t + 300, k0 - 600))
                if not w.any():
                    continue
                j = int(np.flatnonzero(w)[0]); ft = int(ts[j])
                thr = float(sz[w & (tx == tx[j])].sum()); q = min(S, thr)
                cool[lv] = ft + 3600
                T = max(float(fl(c)), round(L + 0.01, 2))
                a = ~bid & (px > T + TOL) & (ts > ft) & (ts <= k0 - 300)
                vol = float(sz[a].sum()); k = round(T, 2)
                qx = min(q, max(vol - used.get(k, 0.0), 0.0)); used[k] = used.get(k, 0.0) + qx
                rest = q - qx
                b = bid & (ts > ft) & (ts >= k0 - 7200) & (ts <= k0 - 300)
                if rest <= 1e-9:
                    how, val = "target", 0.0
                elif b.any():
                    vb = max(float(px[b][-1]) - 0.01, 0.001); how, val = "liq", vb - 0.05 * vb * (1 - vb)
                else:
                    how, val = "held", float(res)
                out.append(dict(cond=row.cond, s=s, lvl=lv, t=t, ft=ft, mtk=(k0 - ft) / 60, c=c, L=L, T=T, through=thr, q=q,
                                qx=qx, how=how, pnl=qx * (T - L) + rest * (val - L), stake=q * L))
    return out


if __name__ == "__main__":
    s = pd.read_parquet(sys.argv[1]); S = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
    rows = []
    for r in s.itertuples():
        rows += sim(r, S)
    d = pd.DataFrame(rows)
    ok = sum(int(np.load(os.path.join(C, f"{c}.npz"))["ok"]) for c in s.cond if os.path.exists(os.path.join(C, f"{c}.npz")))
    print("markets", len(s), "window reached", ok)
    if len(d):
        d = d.merge(s[["cond", "series", "winner", "volume", "start", "title"]], on="cond")
        d.to_parquet(sys.argv[2])
