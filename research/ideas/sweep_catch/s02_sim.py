"""s02: replay resting DEEP limit orders against the real print tape (every game, outcome, side, parameter combo).

Strategy being replayed (per game g, outcome k, side s):
  * every R minutes (refresh) the bot reads the Polymarket mid c of OUR token (s=+1: Yes of k; s=-1: No of k, i.e. a
    short of k) from the 1-minute mid series, LAG=2 minutes stale (no look-ahead: the mid used is from before the
    window), and rests a limit BUY of our token at L = floor_to_cent(c - d). It quotes only while 0.08 <= c <= 0.92,
    from 72 h before kickoff until 10 min before kickoff.
  * Fill rule ('through', the honest one): a MAKER print of a taker SELLING our token at a price strictly below L
    (x < L - 5e-5). By price priority our order at L would have been hit before that print -> we buy at L (not at the
    worse sweep price). 'touch' (x <= L + 5e-5, i.e. also prints AT our price, needs queue priority) = optimistic.
  * After a fill the same (g,k,s) is not re-quoted for COOL=60 min (avoid stacking on one crash).
  * Exit: a resting limit SELL at target T, filled when a taker BUYS our token at a price strictly above T after the
    fill second.  T in {pre-sweep bid-ish level floor(c), L+1c, L+2c, L+3c}.  If T is not hit by the deadline
    H in {60 min, 240 min, 5 min before kick}, sell as taker at mid - half-spread (scalp/out/hs_model.json by game
    volume bucket) paying the taker fee 0.05 p (1-p); or 'hold' = keep to the result (game-result risk);
    or 'vk' = VERIFIED liquidation: sell 1c below the last real bid-hit print of our token in the last 2 h before
    kick (fee paid); if there is none, hold to the result (vk_res=1). T also includes 'tw' = Winner fair (floored).
Per fill we record pnl per share for every (T, H), marks of the mid after the fill, Winner's fair value of our token
at the fill (latest snapshot >= 1 min old), the sweeper (taker wallet, tx usd, levels hit, sweep depth), liquidity
before the fill (prints / usd in the previous 6 h on the game), minutes to kickoff.
Output: data/fills.parquet   (one row per fill per combo R,d,mode)
"""
import os, json, time
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
R_ = os.path.join(HERE, "..", "..")
LL = os.path.join(R_, "lead_lag", "data")
LAG, COOL, TOL = 2, 60, 5e-5
RS = (1, 5, 15, 60)
DS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
MODES = ("through", "touch")
GS = (0.04, 0.06, 0.08, 0.10, 0.12)
# (d, g): g=None -> 'mid' family L = c - d ; g given -> 'anchored' family L = min(c - d, Winner fair - g)
COMBOS = [(d, None) for d in DS] + [(d, g) for d in (0.02, 0.04) for g in GS]
TS_ = ("tmid", "t1", "t2", "t3", "tw")
HS_ = ("h60", "h240", "hk", "hold", "vk")
HSM = json.load(open(os.path.join(R_, "scalp", "out", "hs_model.json")))["hs"]


def fee(p):
    return 0.05 * p * (1 - p)


def hs_of(vol):
    return HSM["<10k"] if vol < 1e4 else HSM["10-50k"] if vol < 5e4 else HSM["50-250k"] if vol < 2.5e5 else HSM[">250k"]


def floor_c(v):
    return np.floor(np.asarray(v) * 100 + 1e-6) / 100


def sim_game(args):
    uid, g, meta = args
    try:
        z = np.load(os.path.join(LL, "series", f"{uid}.npz"))
    except FileNotFoundError:
        return []
    t0, kick = int(z["t0"]), int(z["kick"])
    P = np.round(z["P"].astype(np.float64), 4)
    wt, Q = z["wt"].astype(np.float64), z["Q"].astype(np.float64)
    n = len(P)
    hs = hs_of(meta["poly_volume"])
    won = meta["won"]
    ts = g.ts.to_numpy(); mm = ts // 60 - t0
    ok = (mm >= 0) & (mm < n)
    g = g[ok]; ts = ts[ok]; mm = mm[ok]
    if not len(g):
        return []
    K = g.k.to_numpy(); Y = g.ypx.to_numpy(); ML = g.mlong.to_numpy(); SZ = g["size"].to_numpy().astype(float)
    TX = g.tx.to_numpy(); TW = g.twid.to_numpy(); TU = g.tusd.to_numpy().astype(float)
    usd_all = SZ * np.where(ML, Y, 1 - Y)
    cum_usd = np.cumsum(usd_all)
    out = []
    last = n - 5
    for k in range(3):
        for s in (1, -1):
            X = P[:, k] if s > 0 else np.round(1 - P[:, k], 4)
            wk = Q[:, k] if s > 0 else 1 - Q[:, k]
            ent = (K == k) & (ML == (s > 0))
            ext = (K == k) & (ML == (s < 0))
            xe = np.round(Y[ent] if s > 0 else 1 - Y[ent], 4); te = ts[ent]; me = mm[ent]
            sze, txe, twe, tue = SZ[ent], TX[ent], TW[ent], TU[ent]
            ie = np.flatnonzero(ent)
            xx = np.round(Y[ext] if s > 0 else 1 - Y[ext], 4); tx_ = ts[ext]
            if not len(xe):
                continue
            for R in RS:
                ci = (me // R) * R - LAG
                good = (ci >= 0) & (me <= n - 10)
                c = np.where(good, X[np.clip(ci, 0, n - 1)], np.nan)
                good &= ~np.isnan(c) & (c >= 0.08) & (c <= 0.92)
                # Winner fair of our token at order placement (latest snapshot >= 1 min before the refresh)
                wpi = np.searchsorted(wt, t0 + ci + LAG - 1, side="right") - 1
                wpl = np.where(wpi >= 0, wk[np.clip(wpi, 0, None)], np.nan) if len(wt) else np.full(len(ci), np.nan)
                for d, gg in COMBOS:
                    if gg is None:
                        L = floor_c(c - d)
                    else:
                        L = floor_c(np.fmin(c - d, wpl - gg))
                        L = np.where(np.isnan(wpl), np.nan, L)
                    for mode in MODES:
                        if gg is not None and mode != "through":
                            continue
                        with np.errstate(invalid="ignore"):
                            cond = good & ~np.isnan(L) & ((xe < L - TOL) if mode == "through" else (xe <= L + TOL)) & (L >= 0.02)
                        cand = np.flatnonzero(cond)
                        if not len(cand):
                            continue
                        nxt = -1
                        for j in cand:
                            if me[j] < nxt:
                                continue
                            nxt = me[j] + COOL
                            Lj, cj, tf, mf = L[j], c[j], te[j], me[j]
                            # sweep stats: qualifying prints within 60 s (shares through our level), depth
                            w60 = (te >= tf) & (te <= tf + 60)
                            thr = w60 & (xe < Lj - TOL)
                            sh_thr = float(sze[thr].sum())
                            intx = txe == txe[j]
                            xmin = float(xe[w60].min())
                            lev = len(np.unique(xe[intx]))
                            # Winner fair of our token (snapshot >= 1 min old)
                            wi = np.searchsorted(wt, t0 + mf - 1, side="right") - 1
                            wf = float(wk[wi]) if wi >= 0 else np.nan
                            wage = float(t0 + mf - wt[wi]) if wi >= 0 else np.nan
                            # liquidity before: usd printed on the game in prior 6 h
                            gi = ie[j]
                            a6 = np.searchsorted(ts, tf - 6 * 3600)
                            liq6 = float(cum_usd[gi - 1] - (cum_usd[a6 - 1] if a6 > 0 else 0)) if gi > 0 else 0.0
                            # exits
                            after = tx_ > tf
                            row = dict(uid=uid, k=k, s=s, R=R, d=d, g=(np.nan if gg is None else gg), mode=mode, ts=int(tf), mf=int(mf),
                                       mtk=int(n - mf), c=cj, L=Lj, xmin=xmin, sh_thr=sh_thr, lev=lev,
                                       twid=int(twe[j]), tusd=float(tue[j]), wf=wf, wage=wage, liq6=liq6,
                                       won=(won[k] if s > 0 else 1 - won[k]), hs=hs)
                            for lab, off in (("x5", 5), ("x15", 15), ("x60", 60)):
                                q = min(mf + off, last)
                                row[lab] = X[q] - Lj if mf + off <= last else np.nan
                            row["xc"] = X[last] - Lj
                            # verified liquidation at kick: last REAL print where a taker sold our token (= a bid that
                            # existed) in [kick-120, kick-5] min and after the fill; we sell there 1c worse, pay the
                            # taker fee. No such print -> hold to the result (flagged by vk_res).
                            vw = (te > tf) & (me >= n - 120) & (me <= last)
                            if vw.any():
                                vb = max(float(xe[vw][-1]) - 0.01, 0.001)
                                vk_pnl = vb - fee(vb) - Lj; row["vk_res"] = 0
                            else:
                                vk_pnl = row["won"] - Lj; row["vk_res"] = 1
                            for tl in TS_:
                                if tl == "tmid":
                                    T = max(Lj + 0.01, floor_c(cj))
                                elif tl == "tw":
                                    T = max(Lj + 0.01, floor_c(wf)) if not np.isnan(wf) else Lj + 0.01
                                else:
                                    T = Lj + int(tl[1]) / 100
                                T = round(float(T), 4)
                                hit = after & (xx > T + TOL)
                                hm = (tx_[hit][0] // 60 - t0) if hit.any() else 10 ** 9
                                for hl in HS_:
                                    dl = {"h60": mf + 60, "h240": mf + 240}.get(hl, last)
                                    dl = min(dl, last)
                                    if hm <= dl:
                                        p = T - Lj
                                    elif hl == "hold":
                                        p = row["won"] - Lj
                                    elif hl == "vk":
                                        p = vk_pnl

                                    else:
                                        xb = X[dl]
                                        bid = max(xb - hs, 0.001) if not np.isnan(xb) else np.nan
                                        p = bid - fee(bid) - Lj
                                    row[f"{tl}_{hl}"] = p
                                row[f"{tl}_hit"] = int(hm - mf) if hm < 10 ** 9 else -1
                            out.append(row)
    return out


def tasks(tape, meta):
    pf = pq.ParquetFile(tape)
    buf = None
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i).to_pandas()
        if buf is not None:
            t = pd.concat([buf, t], ignore_index=True)
        lastu = t.unique_id.iloc[-1]
        buf = t[t.unique_id == lastu]
        t = t[t.unique_id != lastu]
        for uid, g in t.groupby("unique_id", sort=False):
            if uid in meta:
                yield uid, g, meta[uid]
    if buf is not None and len(buf):
        uid = buf.unique_id.iloc[0]
        if uid in meta:
            yield uid, buf, meta[uid]


def main():
    t0 = time.time()
    gm = pd.read_parquet(os.path.join(LL, "games_meta.parquet"))
    gm = gm[gm.ok == True]
    meta = {}
    for r in gm.itertuples():
        try:
            op = [float(x) for x in (r.outcome_price0, r.outcome_price1, r.outcome_price2)]
        except Exception:
            continue
        if sorted(op) != [0, 0, 1]:
            continue
        wonW = [np.nan] * 3
        for k in range(3):
            wonW[(2 - k) if r.swapped else k] = op[k]
        meta[r.unique_id] = dict(poly_volume=float(r.poly_volume or 0), won=wonW)
    rows = []
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(sim_game, tasks(os.path.join(D, "tape.parquet"), meta), chunksize=4)):
            rows.extend(r)
            if i % 1000 == 0:
                print(i, len(rows), round(time.time() - t0), "s", flush=True)
    f = pd.DataFrame(rows)
    kt = pd.read_parquet(os.path.join(D, "tape.parquet"), columns=["unique_id", "kick_ts"]).drop_duplicates("unique_id")
    f = f.merge(kt.rename(columns={"unique_id": "uid"}), on="uid", how="left")
    for c in f.columns:
        if f[c].dtype == np.float64 and c not in ("c", "L", "xmin", "wf", "d", "g"):
            f[c] = f[c].astype(np.float32)
    f.to_parquet(os.path.join(D, "fills.parquet"))
    print("fills", len(f), "games", f.uid.nunique(), "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
