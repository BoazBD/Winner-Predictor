"""s10: detailed replay for tuning the production version of the sweep catcher (paper pilot).

Differences from s02 (which picked the rule):
  * deeper levels (d up to 0.35) and a relative-depth family (L = floor(c * (1 - f)));
  * refresh cadences R in {1, 5, 15, 30, 60} min;
  * EVERY candidate entry (first through-print of each tx while an order at L would be resting) is kept; the
    1-fill-per-refresh-slot rule and the cooldown are applied later (s11) AFTER the filters, so filters and
    cooldowns can be varied without re-running;
  * Winner fair of our token at several data lags (production reads the latest Winner scrape from Athena);
  * fill size = shares printed strictly through L in the same transaction (and within 60 s);
  * the EXIT can only be placed when the bot has seen the fill: detection delay dl in {0, 1, 5, 15, 30, 60} min,
    targets T in {floor(c), floor(c)-1c, floor(c)-2c, floor(c)-3c, L + (c-L)/2, Winner fair}; per (T, dl) the
    minute the target is first hit (a taker bought our token strictly above T) and the shares printed above T
    until kick-5 (= the most a resting sell at T could have sold);
  * liquidation of what is left 5 min before kick: 1c under the last real bid-hit print in the final 2 h (fee
    paid), else hold to the result; also stored: plain hold-to-result.
Locked capital: for every refresh slot in which an order is live, L * R share-minutes are added, bucketed by
period / R / d / hours-to-kick / Winner gap bucket / price band (out/s10_lock.parquet).
Output: data/s10_cand.parquet
"""
import os, json, time
from multiprocessing import Pool
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "out")
LL = os.path.join(HERE, "..", "..", "lead_lag", "data")
LAG, TOL = 2, 5e-5
RS = (1, 5, 15, 30, 60)
DMID = (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35)
FREL = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
# hybrid: L = floor(min(c - dm, c * (1 - f))), encoded d = dm * 100 + f (e.g. 15.4 = 15c and 40%)
FHYB = [round(dm * 100 + f, 2) for dm in (0.10, 0.12, 0.15, 0.20) for f in (0.30, 0.40, 0.50)]
FAMS = [("mid", d) for d in DMID] + [("rel", f) for f in FREL] + [("hyb", h) for h in FHYB]


def level(fam, d, c):
    if fam == "mid":
        return floor_c(c - d)
    if fam == "rel":
        return floor_c(c * (1 - d))
    dm, f = np.floor(d) / 100, round(d - np.floor(d), 2)
    return floor_c(np.fmin(c - dm, c * (1 - f)))
WLAGS = (1, 30, 60, 180)            # minutes: Winner snapshot must be at least this old
DLS = (0, 1, 5, 15, 30, 60)
TS_ = ("tm", "tm1", "tm2", "tm3", "half", "tw")
VAL0 = pd.Timestamp("2026-07-01").timestamp(); TEST0 = pd.Timestamp("2026-10-01").timestamp()


def fee(p):
    return 0.05 * p * (1 - p)


def floor_c(v):
    return np.floor(np.asarray(v) * 100 + 1e-6) / 100


def per_of(kick):
    return 0 if kick < VAL0 else 1 if kick < TEST0 else 2


def sim_game(args):
    uid, g, meta = args
    try:
        z = np.load(os.path.join(LL, "series", f"{uid}.npz"))
    except FileNotFoundError:
        return [], None
    t0 = int(z["t0"]); P = np.round(z["P"].astype(np.float64), 4)
    wt, Q = z["wt"].astype(np.float64), z["Q"].astype(np.float64)
    n = len(P); won = meta["won"]; per = per_of(meta["kick"])
    ts = g.ts.to_numpy(); mm = ts // 60 - t0
    ok = (mm >= 0) & (mm < n)
    g = g[ok]; ts = ts[ok]; mm = mm[ok]
    if not len(g):
        return [], None
    K = g.k.to_numpy(); Y = g.ypx.to_numpy(); ML = g.mlong.to_numpy(); SZ = g["size"].to_numpy().astype(float)
    TX = g.tx.to_numpy(); TU = g.tusd.to_numpy().astype(float)
    usd_all = SZ * np.where(ML, Y, 1 - Y); cum_usd = np.cumsum(usd_all)
    last = n - 5
    out = []; lock = np.zeros((len(RS), len(FAMS), 73 * 8 * 2))
    for k in range(3):
        for s in (1, -1):
            X = P[:, k] if s > 0 else np.round(1 - P[:, k], 4)
            wk = Q[:, k] if s > 0 else 1 - Q[:, k]
            ent = (K == k) & (ML == (s > 0)); ext = (K == k) & (ML == (s < 0))
            xe = np.round(Y[ent] if s > 0 else 1 - Y[ent], 4); te = ts[ent]; me = mm[ent]
            sze, txe, tue = SZ[ent], TX[ent], TU[ent]; ie = np.flatnonzero(ent)
            xx = np.round(Y[ext] if s > 0 else 1 - Y[ext], 4); tx_ = ts[ext]; sx = SZ[ext]
            mx_ = tx_ // 60 - t0
            # Winner fair of our token at each minute, for each data lag
            def wf_at(minute, lagm):
                wi = np.searchsorted(wt, t0 + minute - lagm, side="right") - 1
                return np.where(wi >= 0, wk[np.clip(wi, 0, None)], np.nan) if len(wt) else np.full(np.shape(minute), np.nan)
            for R in RS:
                # ---- locked capital: one order per refresh slot (start minute sl), mid at sl - LAG
                sl = np.arange(0, n - 10, R)
                csl = X[np.clip(sl - LAG, 0, n - 1)]; csl[sl - LAG < 0] = np.nan
                wsl = wf_at(sl, 30)
                hb = ((n - sl) // 60).astype(int)                    # hours to kick bucket
                for fam, d in FAMS:
                    Lsl = level(fam, d, csl)
                    with np.errstate(invalid="ignore"):
                        live = ~np.isnan(csl) & (csl >= 0.05) & (csl <= 0.95) & (Lsl >= 0.02)
                    if live.any():
                        gap = wsl[live] - Lsl[live]
                        gb = np.where(np.isnan(gap), 0, np.clip(np.floor(gap * 100 / 3), -1, 6) + 1).astype(int)  # 3c buckets, 0=-inf..
                        bb = ((csl[live] >= 0.08) & (csl[live] <= 0.92)).astype(int)
                        idx = (np.minimum(hb[live], 72) * 8 + gb) * 2 + bb
                        lock[RS.index(R), FAMS.index((fam, d))] += np.bincount(idx, Lsl[live] * R, 73 * 8 * 2)
                if not len(xe):
                    continue
                slot = (me // R) * R
                ci = slot - LAG
                good = (ci >= 0) & (me <= n - 2)
                c = np.where(good, X[np.clip(ci, 0, n - 1)], np.nan)
                with np.errstate(invalid="ignore"):
                    good &= ~np.isnan(c) & (c >= 0.05) & (c <= 0.95)
                for fam, d in FAMS:
                    L = level(fam, d, c)
                    with np.errstate(invalid="ignore"):
                        cond = good & (xe < L - TOL) & (L >= 0.02)
                    cand = np.flatnonzero(cond)
                    seen = set()
                    for j in cand:
                        if txe[j] in seen:
                            continue
                        seen.add(txe[j])
                        Lj, cj, tf, mf = float(L[j]), float(c[j]), int(te[j]), int(me[j])
                        intx = txe == txe[j]
                        sh_tx = float(sze[intx & (xe < Lj - TOL)].sum())
                        w60 = (te >= tf) & (te <= tf + 60)
                        sh60 = float(sze[w60 & (xe < Lj - TOL)].sum())
                        gi = ie[j]; a6 = np.searchsorted(ts, tf - 6 * 3600)
                        liq6 = float(cum_usd[gi - 1] - (cum_usd[a6 - 1] if a6 > 0 else 0)) if gi > 0 else 0.0
                        wonk = won[k] if s > 0 else 1 - won[k]
                        row = dict(uid=uid, per=per, k=k, s=s, R=R, fam=fam, d=d, ts=tf, mf=mf, slot=int(slot[j]),
                                   mtk=int(n - mf), c=cj, L=Lj, xmin=float(xe[w60].min()), sh_tx=sh_tx, sh60=sh60,
                                   tusd=float(tue[j]), liq6=liq6, won=wonk, vol=meta["vol"], tier=meta["tier"])
                        for lg in WLAGS:
                            row[f"wf{lg}"] = float(wf_at(np.array([slot[j]]), lg)[0])   # known at order placement
                        for lab, off in (("x5", 5), ("x15", 15), ("x60", 60)):
                            row[lab] = X[min(mf + off, last)] - Lj if mf + off <= last else np.nan
                        # liquidation at kick-5: 1c under the last real bid-hit print in [kick-120, kick-5] after fill
                        vw = (te > tf) & (me >= n - 120) & (me <= last)
                        if vw.any():
                            vb = max(float(xe[vw][-1]) - 0.01, 0.001)
                            row["vk"] = vb - fee(vb) - Lj; row["vk_res"] = 0
                        else:
                            row["vk"] = wonk - Lj; row["vk_res"] = 1
                        row["hold"] = wonk - Lj
                        wf = row["wf30"]
                        for tl in TS_:
                            if tl == "tm":
                                T = floor_c(cj)
                            elif tl.startswith("tm"):
                                T = floor_c(cj) - int(tl[2]) / 100
                            elif tl == "half":
                                T = floor_c(Lj + (cj - Lj) / 2)
                            else:
                                T = floor_c(wf) if not np.isnan(wf) else floor_c(cj)
                            T = round(max(float(T), Lj + 0.01), 4)
                            row[f"T_{tl}"] = T
                            above = (xx > T + TOL) & (mx_ <= last)
                            for dl in DLS:
                                m = above & (tx_ >= tf + dl * 60) & (tx_ > tf)
                                if m.any():
                                    row[f"h_{tl}_{dl}"] = int(mx_[m][0] - mf); row[f"v_{tl}_{dl}"] = float(sx[m].sum())
                                else:
                                    row[f"h_{tl}_{dl}"] = -1; row[f"v_{tl}_{dl}"] = 0.0
                        out.append(row)
    return out, (per, lock)


def tasks(tape, meta):
    pf = pq.ParquetFile(tape)
    buf = None
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i).to_pandas()
        if buf is not None:
            t = pd.concat([buf, t], ignore_index=True)
        lastu = t.unique_id.iloc[-1]
        buf = t[t.unique_id == lastu]; t = t[t.unique_id != lastu]
        for uid, g in t.groupby("unique_id", sort=False):
            if uid in meta:
                yield uid, g, meta[uid]
    if buf is not None and len(buf):
        uid = buf.unique_id.iloc[0]
        if uid in meta:
            yield uid, buf, meta[uid]


def main():
    t0 = time.time()
    gm = pd.read_parquet(os.path.join(LL, "games_meta.parquet")); gm = gm[gm.ok == True]
    kt = pd.read_parquet(os.path.join(D, "tape.parquet"), columns=["unique_id", "kick_ts"]).drop_duplicates("unique_id")
    kick = dict(zip(kt.unique_id, kt.kick_ts))
    meta = {}
    for r in gm.itertuples():
        try:
            op = [float(x) for x in (r.outcome_price0, r.outcome_price1, r.outcome_price2)]
        except Exception:
            continue
        if sorted(op) != [0, 0, 1] or r.unique_id not in kick:
            continue
        wonW = [np.nan] * 3
        for k in range(3):
            wonW[(2 - k) if r.swapped else k] = op[k]
        meta[r.unique_id] = dict(vol=float(r.poly_volume or 0), won=wonW, kick=kick[r.unique_id], tier=str(r.tier))
    rows = []; LK = np.zeros((3, len(RS), len(FAMS), 73 * 8 * 2))
    with Pool(4) as pool:
        for i, (r, lk) in enumerate(pool.imap_unordered(sim_game, tasks(os.path.join(D, "tape.parquet"), meta), chunksize=8)):
            rows.extend(r)
            if lk is not None:
                LK[lk[0]] += lk[1]
            if i % 1000 == 0:
                print(i, len(rows), round(time.time() - t0), "s", flush=True)
    f = pd.DataFrame(rows)
    for c in f.columns:
        if f[c].dtype == np.float64 and c not in ("c", "L", "d") and not c.startswith("T_") and not c.startswith("wf"):
            f[c] = f[c].astype(np.float32)
        if f[c].dtype == np.int64 and c not in ("ts",):
            f[c] = f[c].astype(np.int32)
    f.to_parquet(os.path.join(D, "s10_cand.parquet"))
    ii = np.indices(LK.shape).reshape(4, -1)
    lk = pd.DataFrame(dict(per=ii[0], R=np.array(RS)[ii[1]], fam=[FAMS[j][0] for j in ii[2]], d=[FAMS[j][1] for j in ii[2]],
                           hb=ii[3] // 16, gb=(ii[3] // 2) % 8 - 1, bb=ii[3] % 2, usdmin=LK.reshape(-1)))
    lk[lk.usdmin > 0].to_parquet(os.path.join(OUT, "s10_lock.parquet"))   # gb: -1 = no Winner, k>=0 = gap in [3(k-1), 3k) c
    print("cands", len(f), "games", f.uid.nunique(), "done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
