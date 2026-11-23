"""s03: attach price marks (Polymarket midpoints from the 15-min grid, Winner orientation) to every pre-start trade row.

For a row of (unique_id, k) at time ts (all prices are the Yes price of outcome k, float64, rounded to 4dp):
  mb   = mid at the last grid point <= ts            (before the trade)
  m0   = mid at the first grid point >= ts           (just after the trade; includes its own impact)
  m1h, m3h, m6h = mid at the first grid point >= ts + 1/3/6 h (capped at the last pre-kick point)
  mc   = mid at the last grid point before kickoff   (~14 min before kick) = 'close'
  won  = outcome k won (1/0)
Grid covers the 72 h before kickoff, so older trades get NaN for mb/m0/mXh (mc and won are always there).
dir = +1 if the wallet got longer outcome k, else -1. Derived:
  mv_c = dir*(mc - m0)   move after the trade, to close   (what a follower who copies instantly at mid could get)
  mv_1h/3h/6h            same to +h
  clv  = dir*(mc - ypx)  the trade's own closing-line value (incl. spread it paid/earned)
  res  = dir*(won - ypx) result-based edge (per share, no fees)
Output data/trades_marked.parquet
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")


def load_grid():
    gr = pd.read_parquet(os.path.join(HERE, "..", "lead_lag", "data", "grid.parquet"),
                         columns=["unique_id", "t_min", "mins_to_kick", "p0", "p1", "p2"])
    gr = gr.sort_values(["unique_id", "t_min"]).reset_index(drop=True)
    for c in ["p0", "p1", "p2"]:
        gr[c] = gr[c].astype(np.float64).round(4)
    gi = pd.factorize(gr.unique_id)
    gr["gi"] = gi[0]
    return gr, pd.Index(gi[1])


def main():
    t = pd.read_parquet(os.path.join(D, "trades_pre.parquet"))
    gr, uids = load_grid()
    P = gr[["p0", "p1", "p2"]].to_numpy()
    key = gr.gi.to_numpy().astype(np.int64) * 10**8 + gr.t_min.to_numpy().astype(np.int64)
    last_idx = gr.groupby("gi").cumcount(ascending=False).to_numpy() == 0
    last_of_g = np.flatnonzero(last_idx)                      # index of last row per gi (gi sorted)
    first_of_g = np.r_[0, last_of_g[:-1] + 1]
    gi = uids.get_indexer(t.unique_id)
    ok = gi >= 0
    print("rows", len(t), "with grid game", ok.mean().round(3), flush=True)
    t = t[ok].reset_index(drop=True); gi = gi[ok].astype(np.int64)
    k = t.k.to_numpy().astype(np.int64)
    tm = t.ts.to_numpy() / 60.0
    lo, hi = first_of_g[gi], last_of_g[gi]

    def at_or_after(minutes):
        q = gi * 10**8 + np.ceil(minutes).astype(np.int64)
        i = np.searchsorted(key, q, side="left")
        bad = i < lo                                   # before grid start -> first point (only for +h marks)
        i = np.clip(i, lo, hi)
        return i, bad

    def before(minutes):
        q = gi * 10**8 + np.floor(minutes).astype(np.int64)
        i = np.searchsorted(key, q, side="right") - 1
        bad = i < lo
        return np.clip(i, lo, hi), bad

    i0, _ = at_or_after(tm)
    in_grid = tm >= gr.t_min.to_numpy()[lo] - 15         # trade within grid coverage
    ib, badb = before(tm)
    t["mb"] = np.where(in_grid & ~badb, P[ib, k], np.nan)
    t["m0"] = np.where(in_grid, P[i0, k], np.nan)
    for h in [1, 3, 6]:
        ih, _ = at_or_after(tm + 60 * h)
        t[f"m{h}h"] = np.where(in_grid, P[ih, k], np.nan)
    t["mc"] = P[hi, k]
    t["mins_close"] = gr.mins_to_kick.to_numpy()[hi]
    g = pd.read_parquet(os.path.join(HERE, "..", "lead_lag", "data", "games_meta.parquet"))
    g = g[g.ok == True].set_index("unique_id")
    # won in Winner orientation
    op = np.stack([g.outcome_price0, g.outcome_price1, g.outcome_price2], 1).astype(float)
    sw = g.swapped.to_numpy().astype(bool)
    opw = op.copy(); opw[sw] = op[sw][:, ::-1]
    gix = g.index.get_indexer(t.unique_id)
    t["won"] = opw[gix, k]
    t["kick_ts"] = ((pd.to_datetime(g.kick_utc) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy()[gix]
    t["tier"] = g.tier.to_numpy()[gix]
    t["series"] = g.series.to_numpy()[gix]
    t["poly_volume"] = g.poly_volume.to_numpy()[gix]
    d = np.where(t["long"], 1.0, -1.0)
    t["dir"] = d.astype(np.int8)
    t["mv_c"] = d * (t.mc - t.m0)
    for h in [1, 3, 6]:
        t[f"mv_{h}h"] = d * (t[f"m{h}h"] - t.m0)
    t["clv"] = d * (t.mc - t.ypx)
    t["res"] = d * (t.won - t.ypx)
    t["impact"] = d * (t.m0 - t.mb)
    for c in ["mb", "m0", "m1h", "m3h", "m6h", "mc", "mv_c", "mv_1h", "mv_3h", "mv_6h", "clv", "res", "impact"]:
        t[c] = t[c].astype(np.float32)
    t.to_parquet(os.path.join(D, "trades_marked.parquet"))
    tk = t[t.taker]
    print("marked", len(t), "in grid", round(float(t.m0.notna().mean()), 3))
    print("taker rows: mean mv_c", tk.mv_c.mean(), "clv", tk.clv.mean(), "res", tk.res.mean())
    print("maker rows: mean mv_c", t[~t.taker].mv_c.mean(), "clv", t[~t.taker].clv.mean())


if __name__ == "__main__":
    main()
