"""s05: turn taker rows into BETS and attach 'insider-like' features (no look-ahead), then measure which features
predict the later pre-kickoff move (research question 2).

Bet = consecutive TAKER rows of one wallet on one (game, outcome k, direction) with gaps <= 10 min.
Features (all computed from information available at the bet's last fill time):
  usd, n_tx, max_levels (distinct maker prices in a tx = book sweep), impact (mid move across the bet),
  vol_before (taker usd traded on the game before the bet), share = usd/(vol_before+usd),
  mins_to_kick, ypx (Yes price of k), dir,
  wallet history (from earlier rows only): age_days since first seen in our data, prior_games, prior_usd,
  fresh (prior_games == 0), series_conc (share of prior games in the same series),
  wallet walk-forward scores (move/clv/res/tmove, month-start, games kicked off before the month only),
  cluster_n (distinct OTHER wallets with a same-direction taker bet >= $200 on the same outcome in the 60 min before),
  cluster_fresh (of those, fresh wallets).
Targets: mid moves in the bet's direction from the mid at (bet end + delay) to the close (~14 min pre-kick), for
delays 0/15/30/60 min (grid is 15-min; 'mv_d0' = from the first grid point after the bet).
Output: data/bets.parquet, out/s05_univariate.csv
"""
import os
import numpy as np, pandas as pd
from common import D, OUT, period, boot_ci
from s03_marks import load_grid


def build_bets(t):
    tk = t[t.taker].sort_values(["wid", "unique_id", "k", "dir", "ts"]).copy()
    key = ["wid", "unique_id", "k", "dir"]
    new = (tk[key].ne(tk[key].shift())).any(axis=1) | (tk.ts.diff() > 600)
    tk["bet"] = new.cumsum()
    tk["pxw"] = tk.ypx * tk["size"]
    b = tk.groupby("bet").agg(wid=("wid", "first"), unique_id=("unique_id", "first"), event_id=("event_id", "first"),
                              k=("k", "first"), dir=("dir", "first"), ts0=("ts", "min"), ts=("ts", "max"),
                              usd=("usd", "sum"), shares=("size", "sum"), pxw=("pxw", "sum"), n_tx=("ts", "size"),
                              max_levels=("tx_levels", "max"), mb=("mb", "first"), kick_ts=("kick_ts", "first"),
                              mc=("mc", "first"), won=("won", "first"), tier=("tier", "first"),
                              series=("series", "first"), poly_volume=("poly_volume", "first"))
    b["ypx"] = (b.pxw / b.shares).round(4)
    return b.drop(columns="pxw").reset_index(drop=True)


def main():
    t = pd.read_parquet(os.path.join(D, "trades_marked.parquet"))
    t = t.sort_values("ts").reset_index(drop=True)
    b = build_bets(t)
    b["mins_to_kick"] = (b.kick_ts - b.ts) / 60
    b = b[b.mins_to_kick > 0]
    print("bets", len(b), flush=True)
    # game volume before the bet (all taker usd on the game)
    tk = t[t.taker][["unique_id", "ts", "usd"]].sort_values(["unique_id", "ts"])
    tk["cum"] = tk.groupby("unique_id").usd.cumsum() - tk.usd
    b = b.sort_values("ts0")
    b = pd.merge_asof(b, tk[["ts", "unique_id", "cum"]].sort_values("ts"), left_on="ts0", right_on="ts",
                      by="unique_id", direction="backward", suffixes=("", "_v")).drop(columns="ts_v")
    b["vol_before"] = b.cum.fillna(0); b = b.drop(columns="cum")
    b["share"] = b.usd / (b.vol_before + b.usd)
    # wallet history from earlier rows (any role) in our data
    life = t.groupby("wid").ts.min().rename("first_ts")
    b = b.join(life, on="wid")
    b["age_days"] = (b.ts0 - b.first_ts) / 86400
    # prior games: games of the wallet whose first row is before this bet's start
    wg = t.groupby(["wid", "unique_id"]).agg(g_ts=("ts", "min"), g_usd=("usd", "sum"), series=("series", "first")) \
        .reset_index().sort_values(["wid", "g_ts"])
    wg["prior_games"] = wg.groupby("wid").cumcount()
    wg["prior_usd"] = wg.groupby("wid").g_usd.cumsum() - wg.g_usd
    wg["ser_cnt"] = wg.groupby(["wid", "series"]).cumcount()
    b = b.merge(wg[["wid", "unique_id", "prior_games", "prior_usd", "ser_cnt"]], on=["wid", "unique_id"], how="left")
    b["fresh"] = b.prior_games == 0
    b["series_conc"] = np.where(b.prior_games > 0, b.ser_cnt / b.prior_games.clip(lower=1), np.nan)
    # walk-forward wallet scores (taker role)
    sc = pd.read_parquet(os.path.join(D, "scores_taker.parquet"))
    b["month"] = pd.to_datetime(b.kick_ts, unit="s").dt.to_period("M").dt.to_timestamp()
    sc = sc.rename(columns={"n": "s_n", "usd": "s_usd", "move": "s_move", "clv": "s_clv", "res": "s_res",
                            "tmove": "s_tmove"})
    b = b.merge(sc, on=["month", "wid"], how="left")
    b["s_n"] = b.s_n.fillna(0)
    # clusters: other wallets' same-direction bets >= $200 on same outcome in previous 60 min
    big = b[b.usd >= 200][["unique_id", "k", "dir", "ts0", "wid", "fresh"]].sort_values("ts0")
    cl_n = np.zeros(len(b)); cl_f = np.zeros(len(b))
    bb = b.reset_index(drop=True)
    grp = {kk: g for kk, g in big.groupby(["unique_id", "k", "dir"])}
    for kk, g in bb[bb.usd >= 100].groupby(["unique_id", "k", "dir"]):
        if kk not in grp:
            continue
        o = grp[kk]; ots = o.ts0.to_numpy(); ow = o.wid.to_numpy(); of = o.fresh.to_numpy()
        for i, ts0, w in zip(g.index, g.ts0.to_numpy(), g.wid.to_numpy()):
            m = (ots < ts0) & (ots >= ts0 - 3600) & (ow != w)
            if m.any():
                cl_n[i] = len(set(ow[m])); cl_f[i] = len(set(ow[m & of]))
    bb["cluster_n"] = cl_n; bb["cluster_fresh"] = cl_f
    b = bb
    # delayed marks
    gr, uids = load_grid()
    P = gr[["p0", "p1", "p2"]].to_numpy()
    key = gr.gi.to_numpy().astype(np.int64) * 10**8 + gr.t_min.to_numpy().astype(np.int64)
    last = np.flatnonzero(gr.groupby("gi").cumcount(ascending=False).to_numpy() == 0)
    first = np.r_[0, last[:-1] + 1]
    gi = uids.get_indexer(b.unique_id).astype(np.int64)
    lo, hi = first[gi], last[gi]
    k = b.k.to_numpy().astype(np.int64)
    in_grid = b.ts.to_numpy() / 60 >= gr.t_min.to_numpy()[lo] - 15
    d = b.dir.to_numpy().astype(float)
    mc = P[hi, k]
    for dl in [0, 15, 30, 60]:
        q = gi * 10**8 + np.ceil(b.ts.to_numpy() / 60 + dl).astype(np.int64)
        i = np.clip(np.searchsorted(key, q), lo, hi)
        md = np.where(in_grid & (gr.t_min.to_numpy()[np.clip(np.searchsorted(key, q), lo, hi)] <= gr.t_min.to_numpy()[hi]),
                      P[i, k], np.nan)
        # if the delayed time is after the last grid point, nothing is left to trade
        late = b.ts.to_numpy() / 60 + dl > gr.t_min.to_numpy()[hi]
        md = np.where(late, np.nan, md)
        b[f"m_d{dl}"] = md
        b[f"mv_d{dl}"] = d * (mc - md)
    b["impact"] = d * (b.m_d0 - b.mb)
    b["res"] = d * (b.won - b.ypx)
    b["period"] = period(b.kick_ts)
    print("all bets", len(b), "saved (usd>=100)", int((b.usd >= 100).sum()), flush=True)
    b = b[b.usd >= 100].reset_index(drop=True)
    b.to_parquet(os.path.join(D, "bets.parquet"))
    print("bets saved", len(b), "with grid", round(float(b.m_d0.notna().mean()), 3), flush=True)
    # univariate tables on bets >= $100 (train/val/test): mean move left after 0 and 60 min delay
    x = b[(b.usd >= 100) & b.m_d0.notna()].copy()
    feats = {
        "usd": pd.cut(x.usd, [100, 500, 2000, 10000, 1e9]),
        "share": pd.cut(x.share, [0, 0.01, 0.05, 0.2, 1.0001]),
        "mins_to_kick": pd.cut(x.mins_to_kick, [0, 45, 60, 75, 120, 360, 1440, 1e6]),
        "fresh": x.fresh,
        "age_days": pd.cut(x.age_days, [-1, 0.01, 1, 7, 30, 1e4]),
        "prior_games": pd.cut(x.prior_games, [-1, 0, 5, 20, 100, 1e6]),
        "series_conc": pd.cut(x.series_conc.fillna(-1), [-2, -0.5, 0.5, 0.8, 1.01]),
        "max_levels": pd.cut(x.max_levels, [0, 1, 2, 3, 5, 1e4]),
        "impact": pd.cut(x.impact, [-1, -0.005, 0.005, 0.02, 0.05, 1]),
        "cluster_n": pd.cut(x.cluster_n, [-1, 0, 1, 3, 1e4]),
        "cluster_fresh": pd.cut(x.cluster_fresh, [-1, 0, 1, 1e4]),
        "s_move": pd.cut(np.where(x.s_n >= 20, x.s_move, np.nan), [-1, -0.01, -0.003, 0.003, 0.01, 0.02, 1]),
        "s_res": pd.cut(np.where(x.s_n >= 20, x.s_res, np.nan), [-1, -0.05, 0, 0.05, 0.1, 1]),
        "ypx": pd.cut(x.ypx, [0, 0.15, 0.35, 0.65, 0.85, 1]),
        "tier": x.tier,
    }
    rows = []
    for f, cut in feats.items():
        x["bin"] = cut.astype(str)
        for (per, bn), g in x.groupby(["period", "bin"]):
            if len(g) < 50:
                continue
            r = dict(feature=f, bin=bn, period=per, bets=len(g), games=g.unique_id.nunique())
            for tg in ["mv_d0", "mv_d60", "res"]:
                gg = g[g[tg].notna()]
                m, lo_, hi_ = boot_ci(gg[tg], gg.unique_id, n=200)
                r[tg] = m; r[tg + "_lo"] = lo_; r[tg + "_hi"] = hi_
            rows.append(r)
    u = pd.DataFrame(rows)
    u.to_csv(os.path.join(OUT, "s05_univariate.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(u.pivot_table(index=["feature", "bin"], columns="period", values=["mv_d0", "mv_d60"]).round(4).to_string())


if __name__ == "__main__":
    main()
