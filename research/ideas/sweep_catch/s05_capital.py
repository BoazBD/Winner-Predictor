"""s05: how much collateral do the resting deep orders lock up, and where are fills densest per locked dollar?

Polymarket requires the full cost of every open BUY order to be covered by the wallet balance (assumed: the CLOB
rejects orders beyond balance minus open orders), so a strategy that rests 1,000-share bids 20c under the mid on every
outcome of every game in the next 72 h ties up a lot of USDC that mostly never fills.
For the selected rule family (mid-based, depth d, Winner gap filter applied AT PLACEMENT: L <= Winner fair - gap) we
replay the order book of OUR orders minute by minute: collateral = L per share for every minute an order rests
(R=1 refresh).  Buckets: hours to kick (0-3,3-6,6-12,12-24,24-48,48-72), side, liquidity before (usd printed on the
game in the previous 6 h: <100, 100-1k, 1k-10k, >10k).
Output: out/collateral.csv (share-minutes and $-minutes of collateral per period x bucket x d)
"""
import os, json
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq
from s02_sim import LL, D, LAG, floor_c

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DS = (0.10, 0.12, 0.15, 0.20)
GAPS = (0.0, 0.06)
TB = np.array([0, 180, 360, 720, 1440, 2880, 4321])
PERS = ["train", "val", "test", "after"]
LB = np.array([-1, 100, 1000, 10000, 1e15])


def per(kick_ts):
    return "train" if kick_ts < pd.Timestamp("2026-07-01").timestamp() else "val" if kick_ts < pd.Timestamp(
        "2026-10-01").timestamp() else "test" if kick_ts < pd.Timestamp("2026-11-19").timestamp() else "after"


def game(args):
    uid, ts, usd, kick_ts = args
    try:
        z = np.load(os.path.join(LL, "series", f"{uid}.npz"))
    except FileNotFoundError:
        return None
    t0 = int(z["t0"]); P = np.round(z["P"].astype(np.float64), 4); wt = z["wt"]; Q = z["Q"].astype(np.float64)
    n = len(P)
    m = np.arange(n)
    ci = m - LAG
    okm = (ci >= 0) & (m <= n - 10)
    # trailing 6 h usd per minute
    mm = ts // 60 - t0
    u = np.zeros(n + 1)
    np.add.at(u, np.clip(mm, 0, n), usd)
    cu = np.cumsum(u)[:n]
    liq = cu - np.r_[np.zeros(360), cu[:-360]] if n > 360 else cu
    liq = np.r_[0, liq[:-1]]                                        # strictly before the minute
    wi = np.searchsorted(wt, t0 + m - 1, side="right") - 1
    tb = np.digitize(n - m, TB) - 1; lb = np.digitize(liq, LB) - 1
    pr = PERS.index(per(kick_ts))
    acc = np.zeros((2, len(DS), len(GAPS), 2, len(TB) - 1, len(LB) - 1))   # [min|usdmin, d, gap, side, tb, lb]
    cell = tb * (len(LB) - 1) + lb
    ncell = (len(TB) - 1) * (len(LB) - 1)
    for k in range(3):
        for si, s in enumerate((1, -1)):
            X = P[:, k] if s > 0 else 1 - P[:, k]
            w = np.where(wi >= 0, (Q[np.clip(wi, 0, None), k] if s > 0 else 1 - Q[np.clip(wi, 0, None), k]), np.nan)
            c = np.where(okm, X[np.clip(ci, 0, n - 1)], np.nan)
            v = okm & ~np.isnan(c) & (c >= 0.08) & (c <= 0.92)
            for di, d in enumerate(DS):
                L = floor_c(np.nan_to_num(c) - d)
                for gi_, gp in enumerate(GAPS):
                    q = v & (L >= 0.02) & (np.nan_to_num(w, nan=-1) - L >= gp) & (tb >= 0) & (tb < len(TB) - 1)
                    if not q.any():
                        continue
                    acc[0, di, gi_, si] += np.bincount(cell[q], minlength=ncell).reshape(len(TB) - 1, len(LB) - 1)
                    acc[1, di, gi_, si] += np.bincount(cell[q], L[q], minlength=ncell).reshape(len(TB) - 1, len(LB) - 1)
    return pr, acc


def tasks():
    pf = pq.ParquetFile(os.path.join(D, "tape.parquet"))
    buf = None
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["unique_id", "ts", "ypx", "size", "mlong", "kick_ts"]).to_pandas()
        if buf is not None:
            t = pd.concat([buf, t], ignore_index=True)
        lu = t.unique_id.iloc[-1]; buf = t[t.unique_id == lu]; t = t[t.unique_id != lu]
        for uid, g in t.groupby("unique_id", sort=False):
            usd = (g["size"] * np.where(g.mlong, g.ypx, 1 - g.ypx)).to_numpy()
            yield uid, g.ts.to_numpy(), usd, int(g.kick_ts.iloc[0])
    if buf is not None and len(buf):
        g = buf
        yield g.unique_id.iloc[0], g.ts.to_numpy(), (g["size"] * np.where(g.mlong, g.ypx, 1 - g.ypx)).to_numpy(), int(
            g.kick_ts.iloc[0])


def main():
    tot = np.zeros((len(PERS), 2, len(DS), len(GAPS), 2, len(TB) - 1, len(LB) - 1))
    with Pool(2) as pool:
        for r in pool.imap_unordered(game, tasks(), chunksize=16):
            if r is not None:
                tot[r[0]] += r[1]
    rows = []
    for idx in np.ndindex(len(PERS), len(DS), len(GAPS), 2, len(TB) - 1, len(LB) - 1):
        p, di, gi_, si, a, b = idx
        rows.append(dict(per=PERS[p], d=DS[di], gap=GAPS[gi_], s=(1, -1)[si], tb=a, lb=b,
                         min=tot[p, 0, di, gi_, si, a, b], usdmin=tot[p, 1, di, gi_, si, a, b]))
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(OUT, "collateral.csv"), index=False)
    months = {"train": 10.0, "val": 3.0, "test": 1.6}
    x = c[np.isclose(c.d, 0.2) & np.isclose(c.gap, 0.06) & c.per.isin(months)]
    avg = x.groupby("per").usdmin.sum() / pd.Series(months) / (30 * 24 * 60)
    print("avg $ locked per 1-share order set (d=0.20, gap6, all games/outcomes/sides):", avg.round(1).to_dict())


if __name__ == "__main__":
    main()
