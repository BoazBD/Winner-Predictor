"""Step 4: execution with REAL trade prints (data/trades.parquet from s01; taker-side fills only).

(a) Spread calibration: effective half-spread = how far real taker prints are from the 1-minute midpoint
    (buy prints: price - mid; sell prints: mid - price), by final-volume bucket and hours to kickoff.
    -> out/spread_calibration.csv, and the half-spread model used below (median by volume bucket).
(b) Re-simulate every signal (F1-F4, F6 24 h; price 15-85c) with fills decided by real prints of the share bought:
    resting BUY at L fills when a taker SELL of that share prints at <= L ('touch') or < L ('through');
    resting SELL at S fills when a taker BUY prints at >= S ('touch') or > S ('through').
    'through100' additionally needs >= $100 of qualifying taker volume (cumulative) before counting the fill.
    For the No share: px = 1 - Yes price and buy/sell flags swap (a taker selling Yes = buying No's complement).
    Taker entry/exit use mid +/- calibrated half-spread + fee; 'taker_v' entry also needs a real taker BUY print of
    our share within 30 min and pays at least that price (guards against empty books with meaningless mids).
Output data/sims_trades.parquet (signal x entry x mode x d), out/spread_calibration.csv
"""
import os, json
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import HERE, LL, load_meta

FEE = 0.05
D_LIST = (0.01, 0.02, 0.03, 0.05, -1.0)
ENTRIES = ("taker", "taker_v", "bid60", "bid240", "mid60")
MODES = (("touch", 0.0), ("through", 0.0), ("through100", 100.0))
VB = [-1, 1e4, 5e4, 2.5e5, 1e12]
VB_LAB = ["<10k", "10-50k", "50-250k", ">250k"]


def fee(p):
    return FEE * p * (1 - p)


def load_series(uid):
    z = np.load(os.path.join(LL, "data", "series", f"{uid}.npz"))
    return int(z["t0"]), int(z["kick"]), z["P"].astype(np.float64)


# ------------------------------------------------------------------ (a) calibration
def calib_game(args):
    uid, tr = args
    t0, kick, P = load_series(uid)
    m = (tr.ts.to_numpy() // 60).astype(np.int64) - t0
    ok = (m >= 1) & (m < len(P))
    tr = tr[ok]; m = m[ok]
    mid = P[m - 1, tr.k.to_numpy()]               # mid of the minute before the print (no own-impact)
    hs_eff = np.where(tr.yes_buy, tr.yes_px - mid, mid - tr.yes_px)
    return pd.DataFrame(dict(uid=uid, hs_eff=hs_eff, mid=mid, mtk=kick - (t0 + m), usd=tr["size"].to_numpy() * tr.price.to_numpy()))


# ------------------------------------------------------------------ (b) re-simulation
def first_fill(ts, px, flag, sz, t_from, t_to, cond, need_usd):
    q = flag & (ts > t_from) & (ts <= t_to) & cond
    if not q.any():
        return -1.0
    idx = np.where(q)[0]
    if need_usd <= 0:
        return ts[idx[0]]
    cum = np.cumsum(sz[idx] * px[idx])
    h = np.where(cum >= need_usd)[0]
    return ts[idx[h[0]]] if h.size else -1.0


def sim_game(args):
    uid, sigs, tr, hs, settle = args
    t0, kick, P = load_series(uid)
    n = len(P)
    last = n - 5
    out = []
    for s in sigs.itertuples():
        k, dr, j = s.k, s.dir, s.j
        X = P[:, k] if dr > 0 else 1 - P[:, k]
        a = j + 1
        if a >= last or np.isnan(X[a]):
            continue
        x0 = X[a]
        T = tr[tr.k == k]
        ts = T.ts.to_numpy() / 60.0 - t0                 # minutes on the game grid
        px = T.yes_px.to_numpy() if dr > 0 else 1 - T.yes_px.to_numpy()
        tb = T.yes_buy.to_numpy() if dr > 0 else ~T.yes_buy.to_numpy()   # taker bought OUR share
        sz = T["size"].to_numpy()
        st = settle[k] if dr > 0 else 1 - settle[k]
        for ent in ENTRIES:
            for mode, need in MODES:
                strict = mode.startswith("through")
                if ent == "taker":
                    pe = min(x0 + hs, 0.999); fe = fee(pe); me = float(a)
                elif ent == "taker_v":      # taker, but only if a real taker BUY of our share printed within 30 min;
                    q = tb & (ts > a) & (ts <= a + 30)   # we pay at least that print's price
                    if not q.any():
                        out.append((s.sid, ent, mode, np.nan, False) + (np.nan,) * 9)
                        continue
                    i0 = np.where(q)[0][0]
                    pe = min(max(px[i0], x0 + hs), 0.999); fe = fee(pe); me = float(ts[i0])
                else:
                    L = np.floor((x0 - hs) * 100 + 1e-6) / 100 if ent.startswith("bid") else np.floor(x0 * 100 + 1e-6) / 100
                    if L <= 0:
                        continue
                    win = 60 if ent.endswith("60") else 240
                    cond = (px < L - 1e-9) if strict else (px <= L + 1e-9)
                    f = first_fill(ts, px, ~tb, sz, a, min(a + win, last), cond, need)
                    if f < 0:
                        out.append((s.sid, ent, mode, np.nan, False) + (np.nan,) * 9)
                        continue
                    me = f; pe = L; fe = 0.0
                for d in D_LIST:
                    if d < 0:
                        if s.fam not in ("F1", "F2") or np.isnan(s.w_fair):
                            continue
                        S = max(pe + 0.01, round(s.w_fair, 2))
                    else:
                        S = pe + d
                    if S >= 1:
                        continue
                    cond = (px > S + 1e-9) if strict else (px >= S - 1e-9)
                    h = first_fill(ts, px, tb, sz, me, last, cond, need)
                    win_pnl = S - pe - fe

                    def tx(mm):
                        mm = int(min(max(mm, a), last))
                        p_ = max(X[mm] - hs, 0.001)
                        return p_ - fee(p_) - pe - fe
                    hit = h >= 0
                    pk = win_pnl if hit else tx(last)
                    ph = win_pnl if hit else st - pe - fe
                    p180 = win_pnl if (hit and h <= me + 180) else tx(np.ceil(me + 180))
                    p720 = win_pnl if (hit and h <= me + 720) else tx(np.ceil(me + 720))
                    out.append((s.sid, ent, mode, d, True, me - j, pe, S, hit, (h - me) if hit else -1, pk, ph, p180, p720))
    return out


COLS = ["sid", "ent", "mode", "d", "filled", "me", "pe", "S", "hit", "t_hit", "pnl_kick", "pnl_hold", "pnl_t180", "pnl_t720"]

if __name__ == "__main__":
    g = load_meta()
    gm = pd.read_parquet(os.path.join(LL, "data", "games_meta.parquet"))
    gm = gm[gm.ok == True].set_index("unique_id")
    T = pd.read_parquet(os.path.join(HERE, "data", "trades.parquet"))
    ev2uid = dict(zip(g.event_id, g.uid))
    T["uid"] = T.event_id.map(ev2uid)
    T = T[T.uid.notna()]
    by = dict(tuple(T.groupby("uid")))
    print("trades", len(T), "games with trades", len(by), "of", len(g), flush=True)
    # ---- (a)
    with Pool(7) as pool:
        C = pd.concat(pool.map(calib_game, list(by.items()), chunksize=50), ignore_index=True)
    C = C.merge(g[["uid", "poly_volume", "split"]], on="uid")
    C["vb"] = pd.cut(C.poly_volume, VB, labels=VB_LAB)
    C["hb"] = pd.cut(C.mtk / 60, [-1, 1, 6, 24, 1e4], labels=["<1h", "1-6h", "6-24h", ">24h"])
    C = C[C.mid.between(0.05, 0.95)]
    cal = C.groupby(["vb", "hb"], observed=True).hs_eff.describe(percentiles=[.25, .5, .75])
    cal.to_csv(os.path.join(HERE, "out", "spread_calibration.csv"))
    cal_v = C.groupby("vb", observed=True).hs_eff.agg(["median", "mean", "size"])
    cal_v["usd_median"] = C.groupby("vb", observed=True).usd.median()
    print(cal_v); print(cal)
    # Prints are conditional on someone choosing to trade (spreads tend to be tight then), so they understate the
    # spread at a random signal time; keep the conservative volume model of s02 (live snapshots: 1-3c in thin books)
    # unless the prints say spreads are even wider.
    MODEL = {"<10k": 0.02, "10-50k": 0.01, "50-250k": 0.0075, ">250k": 0.005}
    HS = {lab: float(max(MODEL[lab], cal_v.loc[lab, "mean"])) for lab in VB_LAB}
    json.dump(dict(hs=HS, table=cal_v.reset_index().astype(str).to_dict("records")), open(os.path.join(HERE, "out", "hs_model.json"), "w"), indent=1)
    print("half-spread model", HS, flush=True)
    # ---- (b)
    S = pd.read_parquet(os.path.join(HERE, "data", "signals.parquet"))
    S = S[S.x0.between(0.15, 0.85) & ((S.fam != "F6") | (S.thr == 24.0))]
    fetched = set(pd.read_parquet(os.path.join(HERE, "data", "trades_events.parquet")).event_id)
    S = S[S.uid.isin(set(g[g.event_id.isin(fetched)].uid))]      # only games whose trade history was fetched
    empty = T.iloc[:0]
    jobs = []
    for uid, sg in S.groupby("uid"):
        r = gm.loc[uid]
        st = np.array([float(r.outcome_price0), float(r.outcome_price1), float(r.outcome_price2)])
        if r.swapped:
            st = st[::-1]
        vb = VB_LAB[int(np.searchsorted(VB, r.poly_volume, side="left")) - 1] if r.poly_volume > 0 else "<10k"
        jobs.append((uid, sg, by.get(uid, empty), HS[vb], st))
    rows = []
    with Pool(7) as pool:
        for i, r in enumerate(pool.imap_unordered(sim_game, jobs, chunksize=20)):
            rows += r
            if i % 2000 == 0:
                print(i, len(rows), flush=True)
    R = pd.DataFrame(rows, columns=COLS)
    for c in R.columns:
        if R[c].dtype == np.float64:
            R[c] = R[c].astype(np.float32)
    R.to_parquet(os.path.join(HERE, "data", "sims_trades.parquet"))
    print("rows", len(R))
