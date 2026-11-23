"""Feature code shared by training, backtests and the live API.

Only odds history (ratio1/2/3 over time) and timing are used. No team / league identity.

Pipeline:  raw rows --odds_history.soccer_1x2_history--> segments --seg_stats--> per-segment path state
           (segment, evaluation time) points --point_features--> wide market features
           --to_long--> one row per (point, outcome) with outcome-oriented features.
"""
import numpy as np
import pandas as pd

TIERS = np.array([1.1111, 1.1715, 1.25])           # Winner 1X2 overround tiers (payout ~90 / ~85 / 80 %), empirical centres
WINDOWS_H = (1, 3, 6, 12, 24, 48)                  # look-back windows for price change, hours
CHECK_H = (96, 72, 48, 36, 24, 18, 12, 8, 6, 4, 3, 2, 1.5, 1, 0.5)   # within-segment re-evaluation points
SNAP_MIN = 15.0


def grid_step(o):
    """Winner price grid: 0.05 below 9.5, 0.2 up to 10, 0.5 above."""
    return np.where(o < 9.55, 0.05, np.where(o < 10.0, 0.2, 0.5))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------------------------------------
# per-segment path state (everything known at the START of the segment)
# ----------------------------------------------------------------------------------------------------------
def seg_stats(seg):
    """seg sorted by (unique_id, seg_no). Adds price / path state columns. Returns a new frame."""
    s = seg.reset_index(drop=True).copy()
    n = len(s)
    O = s[["r1", "r2", "r3"]].values.astype(float)
    q = 1.0 / O
    ovr = q.sum(1)
    tier = (ovr > 1.14).astype(int) + (ovr > 1.21).astype(int)
    M = TIERS[tier]
    Lm = _logit(q / ovr[:, None])                                  # logit of multiplicative margin-free prob
    # Rounding-corrected ("latent") implied probabilities: prices are rounded to a grid but the unrounded implied
    # probabilities sum to the tier overround M, so the deviation (ovr - M) is rounding error. Allocate it to the
    # three prices in proportion to the squared width of each price's rounding interval (favourites absorb most).
    step = grid_step(O)
    wid = 1.0 / (O - step / 2) - 1.0 / (O + step / 2)
    lat = q - (ovr - M)[:, None] * wid ** 2 / (wid ** 2).sum(1, keepdims=True)
    lat = np.clip(lat, 1.0 / (O + step / 2), 1.0 / (O - step / 2))
    L = _logit(lat / lat.sum(1, keepdims=True))                    # logit of rounding-corrected margin-free prob
    sl = lat / q - 1.0                                             # value of the rounding: >0 = price rounded up
    u = 1.0 / np.clip(M[:, None] - (ovr[:, None] - q), 1e-3, None)  # price with zero rounding given the other two
    rr = O / u - 1.0                                               # rounding slack of each price
    mid = pd.factorize(s.unique_id)[0]
    first = np.r_[True, mid[1:] != mid[:-1]]
    ts = s.t_start.values.astype("datetime64[s]").astype(np.int64) / 3600.0      # hours
    te = s.t_end.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    dur = te - ts + SNAP_MIN / 60.0

    prevL = np.vstack([L[:1], L[:-1]])
    dL = np.where(first[:, None], 0.0, L - prevL)
    chg = (~first) & (np.abs(np.vstack([O[:1], O[:-1]]) - O).sum(1) > 1e-9)
    gapf = (s.gap_runs.values > 0) & ~first

    def gcum(x):
        return pd.Series(x).groupby(mid).cumsum().values

    s["tier"] = tier
    s["ovr"] = ovr
    s["dov"] = ovr - M
    s["ts_h"] = ts
    s["n_chg"] = gcum(chg.astype(float))
    s["n_gap"] = gcum(gapf.astype(float))
    s["gap_now"] = gapf.astype(float)
    s["gap_h"] = np.where(gapf, s.gap_runs.values * SNAP_MIN / 60.0, 0.0)
    open_ts = pd.Series(ts).groupby(mid).transform("first").values
    s["open_h"] = open_ts
    # time of the last price change (market open counts as a change)
    lc = pd.Series(np.where(chg | first, ts, np.nan)).groupby(mid).ffill().values
    s["lastchg_h"] = lc
    pc = pd.Series(np.where(chg | first, ts, np.nan))
    prev_lc = pc.groupby(mid).shift().groupby(mid).ffill().values       # change time before the latest one
    lc_prev = np.where(chg | first, prev_lc, np.nan)
    lc_prev = pd.Series(np.where(chg | first, np.nan_to_num(lc_prev, nan=-1.0), np.nan)).groupby(mid).ffill().values
    s["prev_quiet_h"] = np.where(lc_prev < 0, np.nan, lc - lc_prev)     # how long the previous price stood
    kick_h = s.kick.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    s["kick_h"] = kick_h
    s["kick_moved"] = (kick_h != pd.Series(kick_h).groupby(mid).transform("first").values).astype(float)
    cum_dur_prior = gcum(dur) - dur
    s["obs_prior_h"] = cum_dur_prior
    dmax = np.abs(dL).max(1)
    s["absmove_sum"] = gcum(dmax)
    for i in range(3):
        li = L[:, i]
        s[f"l{i}"] = li
        s[f"o{i}"] = O[:, i]
        s[f"rr{i}"] = rr[:, i]
        s[f"m{i}"] = Lm[:, i]
        s[f"sl{i}"] = sl[:, i]
        s[f"open{i}"] = pd.Series(li).groupby(mid).transform("first").values
        s[f"lmin{i}"] = pd.Series(li).groupby(mid).cummin().values
        s[f"lmax{i}"] = pd.Series(li).groupby(mid).cummax().values
        s[f"tw{i}"] = gcum(li * dur) - li * dur                    # time-weighted sum of earlier segments
        d = dL[:, i]
        s[f"nup{i}"] = gcum((chg & (d > 1e-9)).astype(float))
        s[f"ndn{i}"] = gcum((chg & (d < -1e-9)).astype(float))
        last_d = pd.Series(np.where(chg, d, np.where(first, 0.0, np.nan))).groupby(mid).ffill().values
        s[f"lastd{i}"] = last_d
        # move before the last one
        cd = pd.Series(np.where(chg, d, np.nan))
        idx = np.flatnonzero(chg)
        prev_d = np.full(n, np.nan)
        if len(idx):
            sub = pd.Series(d[idx]).groupby(mid[idx]).shift().values
            prev_d[idx] = np.nan_to_num(sub, nan=0.0)
        prev_d[first] = 0.0
        s[f"prevd{i}"] = pd.Series(prev_d).groupby(mid).ffill().values
        s[f"prr{i}"] = np.where(first, 0.0, np.r_[rr[:1, i], rr[:-1, i]])   # slack of the previous price
    s["_mid"] = mid
    return s


# ----------------------------------------------------------------------------------------------------------
# evaluation points
# ----------------------------------------------------------------------------------------------------------
def make_points(seg, runs, check_h=CHECK_H, closing=True):
    """Decision points for training / compact backtests: every segment start, plus the first snapshot of a
    segment after time-to-kickoff crosses each value in check_h, plus the last snapshot of the market.
    Returns frame: seg_row (index into seg), t (datetime64), kind (0 start, 1 checkpoint, 2 close)."""
    runs_h = runs.astype("datetime64[s]").astype(np.int64) / 3600.0
    ts = seg.t_start.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    te = seg.t_end.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    kick = seg.kick.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    rows = [np.arange(len(seg))]
    ridx = [seg.i_start.values.astype(np.int64)]
    kinds = [np.zeros(len(seg), np.int8)]
    for h in check_h:
        target = kick - h                                   # first run with run time >= target
        j = np.searchsorted(runs_h, target - 1e-9)
        ok = (j > seg.i_start.values) & (j <= seg.i_end.values)
        rows.append(np.flatnonzero(ok)); ridx.append(j[ok]); kinds.append(np.ones(ok.sum(), np.int8))
    if closing:
        last = np.flatnonzero(np.r_[seg.unique_id.values[1:] != seg.unique_id.values[:-1], True])
        ok = seg.i_end.values[last] > seg.i_start.values[last]
        rows.append(last[ok]); ridx.append(seg.i_end.values[last][ok].astype(np.int64))
        kinds.append(np.full(ok.sum(), 2, np.int8))
    p = pd.DataFrame({"seg_row": np.concatenate(rows), "run_idx": np.concatenate(ridx),
                      "kind": np.concatenate(kinds)})
    p = p.sort_values(["seg_row", "run_idx", "kind"]).drop_duplicates(["seg_row", "run_idx"], keep="last")
    p["t"] = runs[p.run_idx.values]
    return p.reset_index(drop=True)


def point_features(st, pts):
    """st = seg_stats(seg); pts has seg_row, t. Returns wide per-point frame (market-level + per-outcome cols)."""
    r = pts.seg_row.values
    t = pts.t.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    g = lambda c: st[c].values[r]
    f = pd.DataFrame({"seg_row": r, "t": pts.t.values})
    f["unique_id"] = st.unique_id.values[r]
    f["kick"] = st.kick.values[r]
    f["tier"] = g("tier"); f["ovr"] = g("ovr"); f["dov"] = g("dov")
    f["ttk_h"] = g("kick_h") - t
    f["open_h"] = t - g("open_h")
    f["age_h"] = t - g("lastchg_h")                       # time since last price change
    f["seg_age_h"] = t - g("ts_h")                        # time since this price appeared / reappeared
    f["prev_quiet_h"] = np.nan_to_num(g("prev_quiet_h"), nan=-1.0)
    f["n_chg"] = g("n_chg"); f["n_gap"] = g("n_gap")
    f["gap_now"] = g("gap_now"); f["gap_h"] = g("gap_h")
    f["chg_rate"] = f.n_chg / (f.open_h + 1.0) * 24.0
    f["absmove_sum"] = g("absmove_sum")
    f["kick_moved"] = g("kick_moved")
    kick_local = st.kick.values[r]
    f["kick_hod"] = pd.DatetimeIndex(kick_local).hour.values + pd.DatetimeIndex(kick_local).minute.values / 60.0
    f["kick_dow"] = pd.DatetimeIndex(kick_local).dayofweek.values
    f["snap_hod"] = pd.DatetimeIndex(pts.t.values).hour.values + pd.DatetimeIndex(pts.t.values).minute.values / 60.0
    obs = g("obs_prior_h") + (t - g("ts_h")) + SNAP_MIN / 60.0
    # as-of look-back: price at time t - w  (segment active then; market open price if before open)
    mid = st["_mid"].values
    key = mid.astype(np.float64) * 1e7 + (st.ts_h.values - st.ts_h.values.min())
    base = st.ts_h.values.min()
    for i in range(3):
        li = g(f"l{i}")
        f[f"l{i}"] = li; f[f"o{i}"] = g(f"o{i}"); f[f"rr{i}"] = g(f"rr{i}"); f[f"m{i}"] = g(f"m{i}"); f[f"sl{i}"] = g(f"sl{i}"); f[f"prr{i}"] = g(f"prr{i}")
        f[f"dopen{i}"] = li - g(f"open{i}")
        f[f"lastd{i}"] = g(f"lastd{i}"); f[f"prevd{i}"] = g(f"prevd{i}")
        f[f"frommin{i}"] = li - g(f"lmin{i}"); f[f"frommax{i}"] = li - g(f"lmax{i}")
        f[f"nup{i}"] = g(f"nup{i}"); f[f"ndn{i}"] = g(f"ndn{i}")
        f[f"dtw{i}"] = li - (g(f"tw{i}") + li * (t - g("ts_h") + SNAP_MIN / 60.0)) / obs
    lall = st[["l0", "l1", "l2"]].values
    for w in WINDOWS_H:
        qk = mid[r].astype(np.float64) * 1e7 + (t - w - base)
        j = np.searchsorted(key, qk, side="right") - 1
        j = np.clip(j, 0, len(key) - 1)
        before_open = (mid[j] != mid[r]) | (j < 0)
        for i in range(3):
            past = np.where(before_open, st[f"open{i}"].values[r], lall[j, i])
            f[f"d{w}h{i}"] = f[f"l{i}"].values - past
    return f


OWN = ["l", "m", "sl", "o", "rr", "prr", "dopen", "lastd", "prevd", "frommin", "frommax", "nup", "ndn", "dtw"] + \
      [f"d{w}h" for w in WINDOWS_H]
MKT = ["tier", "ovr", "dov", "ttk_h", "open_h", "age_h", "seg_age_h", "prev_quiet_h", "n_chg", "n_gap", "gap_now",
       "gap_h", "chg_rate", "absmove_sum", "kick_moved", "kick_hod", "kick_dow", "snap_hod"]


def to_long(f):
    """One row per (point, outcome). Columns: own_* for the outcome, a_* / b_* for the other two outcomes
    (a = the other non-draw side, or home when the outcome is the draw; b = the remaining one)."""
    parts = []
    others = {0: (2, 1), 1: (0, 2), 2: (0, 1)}
    for j in range(3):
        a, b = others[j]
        d = {"pt": np.arange(len(f)), "outcome": np.full(len(f), j, np.int8)}
        for c in MKT:
            d[c] = f[c].values
        for c in OWN:
            d[f"own_{c}"] = f[f"{c}{j}"].values
        for c in ("l", "sl", "dopen", "lastd", "d6h", "d24h"):
            d[f"a_{c}"] = f[f"{c}{a}"].values
            d[f"b_{c}"] = f[f"{c}{b}"].values
        parts.append(pd.DataFrame(d))
    L = pd.concat(parts, ignore_index=True)
    O = f[["o0", "o1", "o2"]].values
    rank = O.argsort(1, kind="stable").argsort(1, kind="stable")
    L["rank"] = np.concatenate([rank[:, 0], rank[:, 1], rank[:, 2]]).astype(np.int8)
    L["fav_o"] = np.tile(O.min(1), 3)
    return L


FEATS = ["outcome", "rank", "fav_o"] + MKT + [f"own_{c}" for c in OWN] + \
        [f"{s}_{c}" for c in ("l", "sl", "dopen", "lastd", "d6h", "d24h") for s in ("a", "b")]
LEVEL_FEATS = ["outcome", "rank", "fav_o", "tier", "ovr", "dov", "own_l", "own_m", "own_o", "own_sl", "a_l", "b_l", "a_sl", "b_sl"]


def make_points_full(seg, runs, rows=None):
    """Every snapshot of the given segment rows (default: all) as an evaluation point."""
    rows = np.arange(len(seg)) if rows is None else np.asarray(rows)
    n = (seg.i_end.values[rows] - seg.i_start.values[rows] + 1).astype(np.int64)
    r = np.repeat(rows, n)
    off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    idx = seg.i_start.values[r].astype(np.int64) + off
    return pd.DataFrame({"seg_row": r, "run_idx": idx, "kind": np.where(off == 0, 0, 1).astype(np.int8), "t": runs[idx]})


def make_points_latest(seg):
    """The latest snapshot of each market."""
    last = np.flatnonzero(np.r_[seg.unique_id.values[1:] != seg.unique_id.values[:-1], True])
    return pd.DataFrame({"seg_row": last, "run_idx": seg.i_end.values[last].astype(np.int64),
                         "kind": np.full(len(last), 2, np.int8), "t": seg.t_end.values[last]})
