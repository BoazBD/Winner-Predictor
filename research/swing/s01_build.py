"""Swing research, step 1: one simulated order per (game, outcome, direction, decision time), labelled from REAL trades.

Decision times: every 30 min from 48 h to 30 min before kickoff (the lead/lag 15-min midpoint grid, every other row).
Directions: +1 = buy the outcome's Yes, -1 = buy its No. Everything is computed in "our token" prices: for a No we
mirror Yes-equivalent prices (p -> 1-p) and taker sides, so one code path handles both.

Order life (all maker orders, no fee except a forced exit):
  1. limit BUY at L = floor(mid) - x (x in X_C cents), resting W_ENTRY. Filled only if a real taker SOLD our token
     strictly below L in (t, t+W] (strict = a trade printed through our level; conservative on queue position).
  2. limit SELL at L + d (d in D_C cents) from the fill until E = kickoff - 5 min. Filled only if a real taker BOUGHT
     strictly above L + d.
  3. otherwise sell at market at E: bid = min(last real taker-sell price in [E-60m, E], mid(E) - 0.5c), or mid(E)-1c
     when nobody sold in that hour, minus the taker fee 0.05*p*(1-p). Never held into the game.
Trade prices carry float noise (0.85999999988 for 0.86), so they are rounded to 4 decimals before any comparison.
Labels: pnl_x{x}_d{d} in $ per share (0 when the buy did not fill), fill flags, and features known at t (no look-ahead).
Output: data/orders.parquet
"""
import os, numpy as np, pandas as pd, duckdb
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..")
X_C = (0, 1, 2)
D_C = (1, 2, 3, 5)
W_ENTRY = 2 * 3600
EXIT_BEFORE = 5 * 60
PMIN, PMAX = 0.10, 0.90
DELAY = int(os.environ.get('DELAY', 0))   # robustness: order starts this many seconds after the decision
QMIN = float(os.environ.get('QMIN', 0))    # robustness: shares that must trade through our level before we count a fill
OUT = os.environ.get('OUT', 'orders')


def fee(p):
    return 0.05 * p * (1 - p)


def load():
    c = duckdb.connect()
    meta = pd.read_parquet(os.path.join(R, "lead_lag/data/games_meta.parquet"),
                           columns=["unique_id", "ok", "tier", "league", "poly_volume"])
    meta = meta[meta.ok == True]
    G = c.sql("""select unique_id uid, mins_to_kick mtk, t_min, w0,w1,w2,p0,p1,p2
                 from read_parquet(?) where mins_to_kick <= 2895 order by uid, t_min""",
              params=[os.path.join(R, "lead_lag/data/grid.parquet")]).df()
    T = c.sql("""select unique_id uid, k, round(case when yes then q else 1-q end, 4) ypx, yes ybuy, size, ts
                 from read_parquet(?) where rel_min < 0 order by uid, ts""",
              params=[os.path.join(R, "edge/data/master.parquet")]).df()
    G = G[G.uid.isin(set(T.uid)) & G.uid.isin(set(meta.unique_id))].copy()
    for c in ("p0", "p1", "p2", "w0", "w1", "w2"):   # stored as float32: 0.86 -> 0.8600000143 would fake through-fills
        G[c] = G[c].astype("float64").round(4 if c[0] == "p" else 6)
    return G, T, meta.set_index("unique_id")


def first_after(ts, ok, t0, t1):
    """index of the first trade with ok & t0 < ts <= t1 for each (t0,t1) row; -1 if none. ts sorted."""
    if len(ts) == 0:
        return np.full(len(t0), -1)
    M = ok[None, :] & (ts[None, :] > t0[:, None]) & (ts[None, :] <= t1[:, None])
    i = M.argmax(1)
    return np.where(M.any(1), i, -1)


def game(args):
    uid, g, t, tier = args
    kick = int((g.t_min.iloc[0] + g.mtk.iloc[0]) * 60)
    E = kick - EXIT_BEFORE
    gm = g.t_min.to_numpy() * 60
    rows = []
    # decision points: every other grid row (30 min), 30 min .. 48 h before kickoff
    dec = g[(g.mtk >= 30) & (g.mtk <= 2880)].iloc[::2]
    if len(dec) == 0:
        return None
    # exit midpoint: the last grid row at or before E
    ge = g[gm <= E]
    if len(ge) == 0:
        return None
    for k in range(3):
        tk = t[t.k == k]
        if len(tk) == 0:
            continue                                            # no trades at all: an order could never fill
        ts_all = tk.ts.to_numpy(); ypx = tk.ypx.to_numpy(); yb = tk.ybuy.to_numpy(); sz = tk["size"].to_numpy()
        for dr in (1, -1):
            px = ypx if dr == 1 else 1 - ypx
            buy = yb if dr == 1 else ~yb                       # taker bought OUR token
            mid_all = g[f"p{k}"].to_numpy() if dr == 1 else 1 - g[f"p{k}"].to_numpy()
            w_all = g[f"w{k}"].to_numpy() if dr == 1 else 1 - g[f"w{k}"].to_numpy()
            mid_e = (ge[f"p{k}"].iloc[-1] if dr == 1 else 1 - ge[f"p{k}"].iloc[-1])
            sel = (ts_all > E - 3600) & (ts_all <= E) & ~buy
            bid_e = min(px[sel][-1], mid_e - 0.005) if sel.any() else mid_e - 0.01
            bid_e = max(bid_e, 0.0)
            exit_px = bid_e - fee(bid_e)
            t0 = dec.t_min.to_numpy() * 60
            mid = (dec[f"p{k}"] if dr == 1 else 1 - dec[f"p{k}"]).to_numpy()
            keep = (mid >= PMIN) & (mid <= PMAX)
            if not keep.any():
                continue
            t0, mid = t0[keep], mid[keep]
            d0 = dec[keep]
            out = dict(uid=uid, k=k, dir=dr, t=t0, mtk=d0.mtk.to_numpy(), mid=mid,
                       w=(d0[f"w{k}"] if dr == 1 else 1 - d0[f"w{k}"]).to_numpy(),
                       psum=(d0.p0 + d0.p1 + d0.p2).to_numpy(),
                       wsum=(d0.w0 + d0.w1 + d0.w2).to_numpy(),
                       rank=np.array([(d0[[f"p{j}" for j in range(3)]].to_numpy()[i] > d0[f"p{k}"].to_numpy()[i]).sum()
                                      for i in range(len(d0))]),
                       mid_exit=np.full(len(t0), mid_e), bid_exit=np.full(len(t0), bid_e))
            # --- features from the midpoint / Winner grid (strictly at or before t) ---
            for h, n in ((1, 4), (3, 12), (6, 24), (24, 96)):
                idx = np.searchsorted(gm, t0 - h * 3600, side="right") - 1
                ok = idx >= 0
                out[f"dmid_{h}h"] = np.where(ok, mid - mid_all[np.maximum(idx, 0)], np.nan)
                out[f"dw_{h}h"] = np.where(ok, out["w"] - w_all[np.maximum(idx, 0)], np.nan)
            vol6 = []
            for tt in t0:
                m = (gm > tt - 6 * 3600) & (gm <= tt)
                vol6.append(np.abs(np.diff(mid_all[m])).sum() if m.sum() > 1 else 0.0)
            out["absmove_6h"] = np.array(vol6)
            # --- features from trades before t ---
            cs_n = np.searchsorted(ts_all, t0, side="right")
            notional = sz * px
            cn = np.concatenate([[0], np.cumsum(notional)])
            cb = np.concatenate([[0], np.cumsum(notional * buy)])
            out["n_tr"] = cs_n
            out["lognot"] = np.log1p(cn[cs_n])
            for h in (1, 6):
                lo = np.searchsorted(ts_all, t0 - h * 3600, side="right")
                tot = cn[cs_n] - cn[lo]; bb = cb[cs_n] - cb[lo]
                out[f"n_{h}h"] = cs_n - lo
                out[f"not_{h}h"] = np.log1p(tot)
                out[f"buyshare_{h}h"] = np.where(tot > 0, bb / np.maximum(tot, 1e-9), np.nan)
            li = cs_n - 1
            out["last_gap"] = np.where(li >= 0, px[np.maximum(li, 0)] - mid, np.nan)
            out["since_last_h"] = np.where(li >= 0, (t0 - ts_all[np.maximum(li, 0)]) / 3600, np.nan)
            for nm, msk in (("lastbuy_gap", buy), ("lastsell_gap", ~buy)):
                ii = np.flatnonzero(msk)
                j = np.searchsorted(ts_all[ii], t0, side="right") - 1
                out[nm] = np.where(j >= 0, px[ii[np.maximum(j, 0)]] - mid, np.nan) if len(ii) else np.full(len(t0), np.nan)
            # --- labels ---
            t1 = np.minimum(t0 + DELAY + W_ENTRY, E)
            for x in X_C:
                L = np.round(np.floor(mid * 100 + 1e-6) / 100 - x / 100, 2)
                M = (~buy[None, :]) & (px[None, :] < L[:, None] - 5e-5) & (ts_all[None, :] > t0[:, None] + DELAY) & (ts_all[None, :] <= t1[:, None])
                R1 = np.cumsum(M * sz[None, :], 1) >= max(QMIN, 1e-12)
                fi = np.where(R1.any(1), R1.argmax(1), -1)
                filled = fi >= 0
                tf = np.where(filled, ts_all[np.maximum(fi, 0)] if len(ts_all) else 0, 0)
                out[f"fill_x{x}"] = filled
                out[f"tfill_x{x}"] = np.where(filled, tf, np.nan)
                for d in D_C:
                    tgt = L + d / 100
                    M2 = buy[None, :] & (px[None, :] > tgt[:, None] + 5e-5) & (ts_all[None, :] > tf[:, None]) & (ts_all[None, :] <= E)
                    hit = (np.cumsum(M2 * sz[None, :], 1) >= max(QMIN, 1e-12)).any(1) & filled
                    pnl = np.where(hit, d / 100, exit_px - L)
                    out[f"hit_x{x}_d{d}"] = hit
                    out[f"pnl_x{x}_d{d}"] = np.where(filled, pnl, 0.0)
                out[f"L_x{x}"] = L
            rows.append(pd.DataFrame(out))
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df["kick"] = kick
    df["tier_top"] = tier == "top"
    return df


def main():
    G, T, meta = load()
    print("games", G.uid.nunique(), "trades", len(T), flush=True)
    tg = dict(tuple(T.groupby("uid", sort=False)))
    args = [(u, g.reset_index(drop=True), tg[u].reset_index(drop=True), meta.tier.get(u)) for u, g in G.groupby("uid", sort=False)]
    del T
    with Pool(8) as p:
        parts = [r for r in p.imap_unordered(game, args, chunksize=20) if r is not None]
    df = pd.concat(parts, ignore_index=True)
    df["kick"] = pd.to_datetime(df.kick, unit="s")
    df.to_parquet(os.path.join(HERE, "data", OUT + ".parquet"))
    print("orders", len(df), "games", df.uid.nunique(), flush=True)
    for x in X_C:
        print("x", x, "fill", df[f"fill_x{x}"].mean().round(3),
              {d: round(df[df[f'fill_x{x}']][f'pnl_x{x}_d{d}'].mean() * 100, 3) for d in D_C})


if __name__ == "__main__":
    main()
