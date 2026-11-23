"""Step 6: from the per-game series, extract
  data/events.parquet  - major moves on either venue and how/when the other venue followed (+ random-time placebo)
  data/grid.parquet    - both venues sampled at every Winner snapshot (for cross-correlation / regressions)
  data/trades.parquet  - simulated "Winner moved, Polymarket has not" signals at 15 min / 1 h / 3 h check cadences
Definitions are documented in REPORT.md. Times are integer minutes on the game's UTC grid.
"""
import os, glob, zlib
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
WIN = 180          # move must happen within a trailing 3-hour window
REFRACT = 360      # one event per game & venue per 6 hours
THRS = (0.03, 0.05, 0.08)
HORIZONS = (0, 15, 60, 180, 360)
LOOKBACK = 360     # trading signal: Winner change over the last 6 h
CADENCES = (15, 60, 180)
SIG_THRS = (0.02, 0.03, 0.05)
RNG_SEED = 7
PATH_OFFS = np.arange(-360, 361, 15)


def roll_med5(P):
    return pd.DataFrame(P).rolling(5, center=True, min_periods=1).median().to_numpy(dtype=np.float32)


def first_cross(x, start, d, base, need):
    """first index j >= start with d*(x[j]-base) >= need, else -1"""
    seg = d * (x[start:] - base) >= need
    j = np.argmax(seg) if seg.size else 0
    return start + j if seg.size and seg[j] else -1


def detect(X, lo, thr):
    """moves of >= thr within WIN minutes; returns list of (i, k, d, tb, M). X: (n,3)"""
    n = len(X)
    df = pd.DataFrame(X)
    rmin = df.rolling(WIN + 1, min_periods=1).min().to_numpy()
    rmax = df.rolling(WIN + 1, min_periods=1).max().to_numpy()
    up = X - rmin; dn = rmax - X
    out = []
    i = lo
    while i < n:
        cand = np.where(np.nanmax(np.maximum(np.nan_to_num(up[i:], nan=0), np.nan_to_num(dn[i:], nan=0)), 1) >= thr)[0]
        if not cand.size:
            break
        i = i + cand[0]
        if np.nanmax(up[i]) >= np.nanmax(dn[i]):
            k = int(np.nanargmax(up[i])); d = 1
            w = X[i - WIN:i + 1, k]; tb = i - WIN + int(np.where(w == np.nanmin(w))[0][-1])
        else:
            k = int(np.nanargmax(dn[i])); d = -1
            w = X[i - WIN:i + 1, k]; tb = i - WIN + int(np.where(w == np.nanmax(w))[0][-1])
        out.append((i, k, d, tb, float(d * (X[i, k] - X[tb, k]))))
        i += REFRACT
    return out


def pack(z, shift, n, t0):
    """Venue series of game z on a reference minute grid t0..t0+n-1, where reference minute m corresponds to
    minute m+shift of game z (shift = kick_z - kick_ref aligns by minutes-to-kickoff; 0 for the game itself)."""
    t0z, P, wt, Q = int(z["t0"]), z["P"], z["wt"], z["Q"]
    mins = t0 + np.arange(n) + shift
    widx = np.searchsorted(wt, mins, side="right") - 1
    W = np.where(widx[:, None] >= 0, Q[np.clip(widx, 0, None)], np.nan).astype(np.float32)
    pi = mins - t0z
    okp = (pi >= 0) & (pi < len(P))
    Pa = np.full((n, 3), np.nan, dtype=np.float32); Pa[okp] = P[pi[okp]]
    return dict(W=W, P=Pa, Ps=roll_med5(Pa), widx=widx, wt=wt - shift)


def process(args):
    f, fpart = args
    uid = os.path.basename(f)[:-4]
    z = np.load(f)
    t0, kick = int(z["t0"]), int(z["kick"])
    n = len(z["P"])
    own = pack(z, 0, n, t0)
    W, P, Ps, widx, wt = own["W"], own["P"], own["Ps"], own["widx"], own["wt"]
    both = np.where(~np.isnan(W).any(1) & ~np.isnan(P).any(1))[0]
    if both.size < WIN + 60:
        return None
    s0 = both[0]
    zp = np.load(fpart)
    part = pack(zp, int(zp["kick"]) - kick, n, t0)      # another game's series, aligned by minutes-to-kickoff
    rng = np.random.default_rng(zlib.crc32(uid.encode()) + RNG_SEED)

    def wtimes(pk, j):
        """Winner true-change bounds for the value first visible at minute j: (prev snap, snap]"""
        a = pk["widx"][j]; w = pk["wt"]
        return (w[a - 1] if a >= 1 else w[a] - 15.0), w[a]

    # ---------------- events
    ev = []
    for venue, other in (("winner", "poly"), ("poly", "winner")):
        key = {"winner": "W", "poly": "Ps"}
        XA = own[key[venue]]
        for thr in THRS:
            for (i, k, d, tb, M) in detect(XA, s0 + WIN, thr):
                for kind in ("real", "placebo", "shuffled"):
                    pkb = part if kind == "shuffled" else own
                    XB = pkb[key[other]]
                    if kind == "placebo":       # same shape at a random minute of the same game
                        ip = int(rng.integers(s0 + WIN, n)); tbp = max(s0, ip - (i - tb))
                        ii, tbb = ip, tbp
                    else:
                        ii, tbb = i, tb
                    xa, xb = XA[:, k], XB[:, k]
                    if np.isnan(xb[tbb]):
                        continue
                    persist = d * (xa[min(ii + 30, n - 1)] - xa[tbb]) >= M / 2
                    ja = first_cross(xa, tbb, d, xa[tbb], M / 2) if kind != "placebo" else ii
                    jb = first_cross(xb, tbb, d, xb[tbb], M / 2)
                    pre_b = d * (xb[tbb] - xb[max(s0, tbb - 360)]) / M
                    row = dict(unique_id=uid, venue=venue, thr=thr, kind=kind, k=k, d=d, M=M, i=ii, tb=tbb,
                               t_min=t0 + ii, mins_to_kick=kick - (t0 + ii), level=float(xa[tbb]),
                               gap_before=float(d * (xb[tbb] - xa[tbb])), persist=bool(persist),
                               a_half=ja, b_half=jb, pre_b=float(pre_b))
                    # time bounds for the first-mover comparison, in minutes (absolute)
                    if jb >= 0 and ja >= 0:
                        if venue == "winner":
                            we, wl = wtimes(own, ja); tp = t0 + jb
                        else:
                            we, wl = wtimes(pkb, jb); tp = t0 + ja
                        row.update(w_early=float(we), w_late=float(wl), p_time=float(tp))
                    for h in HORIZONS:
                        j = ii + h
                        row[f"fb_{h}"] = float(d * (xb[j] - xb[tbb]) / M) if j < n else np.nan
                        row[f"fa_{h}"] = float(d * (xa[j] - xa[tbb]) / M) if j < n else np.nan
                    offs = ii + PATH_OFFS
                    okp = (offs >= s0) & (offs < n)
                    row["path_a"] = np.where(okp, d * (xa[np.clip(offs, 0, n - 1)] - xa[tbb]) / M, np.nan).astype(np.float32)
                    row["path_b"] = np.where(okp, d * (xb[np.clip(offs, 0, n - 1)] - xb[tbb]) / M, np.nan).astype(np.float32)
                    row["fb_kick"] = float(d * (xb[n - 1] - xb[tbb]) / M)
                    row["fa_kick"] = float(d * (xa[n - 1] - xa[tbb]) / M)
                    ev.append(row)

    # ---------------- grid at Winner snapshots (first minute at/after each snapshot)
    js = np.unique(np.ceil(wt - t0).astype(int))
    js = js[(js >= s0) & (js < n)]
    grid = pd.DataFrame(dict(unique_id=uid, j=js, mins_to_kick=kick - (t0 + js), t_min=t0 + js,
                             **{f"w{k}": W[js, k] for k in range(3)}, **{f"p{k}": P[js, k] for k in range(3)}))

    # ---------------- trading signals
    tr = []
    for cad in CADENCES:
        # check right after the Winner snapshot, at clock times divisible by the cadence (UTC)
        chk = js[((t0 + js) % cad) < 15] if cad > 15 else js
        chk = chk[chk - LOOKBACK >= s0]
        taken = set()
        for j in chk:
            for k in range(3):
                dW = W[j, k] - W[j - LOOKBACK, k]
                dP = P[j, k] - P[j - LOOKBACK, k]
                for thr in SIG_THRS:
                    if abs(dW) < thr:
                        continue
                    d = 1 if dW > 0 else -1
                    if d * dP >= 0.5 * abs(dW):
                        continue
                    key = (thr, k, d)
                    if key in taken:
                        continue
                    taken.add(key)
                    je = min(j + 1, n - 1)
                    tr.append(dict(unique_id=uid, cadence=cad, thr=thr, k=k, d=d, j=j, t_min=t0 + j,
                                   mins_to_kick=kick - (t0 + j), dW=float(dW), dP=float(dP),
                                   p_entry=float(P[je, k]), p_close=float(P[n - 1, k]), w_now=float(W[j, k]),
                                   w_close=float(W[n - 1, k]),
                                   **{f"p_{h}": float(P[j + h, k]) if j + h < n else float(P[n - 1, k])
                                      for h in (15, 60, 180, 360)}))
    return (pd.DataFrame(ev), grid, pd.DataFrame(tr))


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(D, "series", "*.npz")))
    perm = np.random.default_rng(RNG_SEED).permutation(len(files))
    partner = [None] * len(files)
    for a, b in zip(perm, np.roll(perm, -1)):      # each game paired with a random other game
        partner[a] = files[b]
    with Pool(7) as pool:
        res = [r for r in pool.imap_unordered(process, list(zip(files, partner)), chunksize=20) if r is not None]
    ev = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    grid = pd.concat([r[1] for r in res], ignore_index=True)
    tr = pd.concat([r[2] for r in res if len(r[2])], ignore_index=True)
    ev.to_parquet(os.path.join(D, "events.parquet")); grid.to_parquet(os.path.join(D, "grid.parquet"))
    tr.to_parquet(os.path.join(D, "trades.parquet"))
    print("games", len(res), "events", len(ev), "grid", len(grid), "trades", len(tr))
    print(ev[ev.kind == "real"].groupby(["venue", "thr"]).size())
