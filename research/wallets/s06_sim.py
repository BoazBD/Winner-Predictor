"""s06: realistic trade simulation for every candidate bet (research question 4).

Universe: bets (s05) with usd >= MIN_USD, >= 30 min before kickoff, that can match at least one s07 rule family. For each bet we play the follower who sees it
after a detection DELAY (5, 15, 30, 60 min and 'U' = uniform 0-60 min, the hourly-cron case) and enters in the bet's
direction (COPY) or the opposite (FADE). Our token = Yes of k when going long k, No of k when going short.
All prices in OUR token's terms, float64 rounded to 4dp; tick 0.01; float-safe strict comparisons (1e-9).

Entry (a) TAKER: first real taker print in our direction within 30 min after detection (= someone really bought at
          the ask), we pay that price + fee 0.05*p*(1-p). No print -> no trade. capacity = shares bought by takers in
          our direction in those 30 min at <= our price + 1c.
      (b) MAKER j: limit buy at floor(token mid at detection) - j cents (j=0,1), resting 60 min (and never later than
          kickoff-20 min). Filled only if a real taker SELLS our token strictly below our limit (an opposite-direction
          taker print with price < limit). No fee. capacity = shares that traded strictly through.
Exit: limit sell at entry + d (d = 1,2,3,5 c; maker, no fee), filled only if a real taker BUYS our token strictly
      above it before kickoff-10 min; otherwise sell at the bid at ~kickoff-14 min = token close mid - half-spread
      (2c <$10k games, 1c <$50k, 0.75c <$250k, 0.5c above), paying the taker fee. 'inf' = no limit, just close.
      Never held into the game.
Output: data/sim.parquet: one row per (bet, side copy/fade, delay, entry) with cost, filled, capacity and the proceeds
        for every d.
"""
import os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from common import D, fee
from s03_marks import load_grid

MIN_USD = 200
DELAYS = [5, 15, 60, -1]              # -1 = uniform 0-60 (the hourly job)
DS = [0.01, 0.02, 0.03, 0.05]
EPS = 1e-9


def hs_of(vol):
    return np.where(vol < 1e4, 0.02, np.where(vol < 5e4, 0.01, np.where(vol < 2.5e5, 0.0075, 0.005)))


def sim_group(args):
    bets, pr, gt, gp, kick, mc, hs = args
    # pr: prints of this (game,k): ts, ypx, dir, size (taker rows), sorted by ts
    pts, pyp, pdir, psz = pr
    out = []
    rng = np.random.default_rng(int(bets.bet_id.iloc[0]) % 2**31)
    for b in bets.itertuples():
        for side in (1, -1):                       # 1 = copy, -1 = fade
            od = b.dir * side                      # our direction on outcome k
            tp = pyp if od == 1 else np.round(1 - pyp, 4)   # print price in our token terms
            same = pdir == od
            gmid = gp if od == 1 else np.round(1 - gp, 4)
            mc_t = mc if od == 1 else round(1 - mc, 4)
            close_bid = round(mc_t - hs, 4)
            close_proc = close_bid - float(fee(close_bid)) if close_bid > 0 else 0.0
            tend = kick - 1200
            for dl in DELAYS:
                dmin = rng.uniform(0, 60) if dl < 0 else dl
                te = b.ts + dmin * 60
                if te > tend:
                    continue
                gi = np.searchsorted(gt, te / 60.0, "left")
                if gi >= len(gt):
                    continue
                mid = gmid[gi]
                for entry in ("taker", "mk0", "mk1"):
                    if entry == "taker":
                        w1 = min(te + 1800, tend)
                        m = same & (pts >= te) & (pts <= w1)
                        if not m.any():
                            out.append((b.bet_id, side, dl, dmin, entry, 0, np.nan, np.nan, np.nan, 0.0) + (np.nan,) * 5)
                            continue
                        j = np.flatnonzero(m)[0]
                        pe = tp[j]; tf = pts[j]
                        if pe < 0.02 or pe > 0.98:
                            continue
                        cost = pe + float(fee(pe))
                        cap = psz[m & (tp <= pe + 0.01 + EPS)].sum()
                    else:
                        jj = 0 if entry == "mk0" else 1
                        L = np.floor(mid * 100 + 1e-6) / 100 - jj * 0.01
                        L = round(L, 2)
                        if L < 0.02 or L > 0.98:
                            continue
                        w1 = min(te + 3600, tend)
                        m = (~same) & (pts >= te) & (pts <= w1) & (tp < L - EPS)
                        if not m.any():
                            out.append((b.bet_id, side, dl, dmin, entry, 0, L, np.nan, np.nan, 0.0) + (np.nan,) * 5)
                            continue
                        j = np.flatnonzero(m)[0]
                        pe = L; tf = pts[j]; cost = L
                        cap = psz[m].sum()
                    # exits
                    after = same & (pts > tf) & (pts <= kick - 600)
                    procs = []
                    for d in DS:
                        X = round(pe + d, 4)
                        hit = after & (tp > X + EPS)
                        procs.append(X if (hit.any() and X < 1) else close_proc)
                    procs.append(close_proc)
                    out.append((b.bet_id, side, dl, dmin, entry, 1, pe, cost, tf, float(cap)) + tuple(procs))
    return out


def main():
    b = pd.read_parquet(os.path.join(D, "bets.parquet"))
    b["bet_id"] = b.index.astype(np.int64)
    if os.path.exists(os.path.join(D, "bets_ml.parquet")):
        b = b.merge(pd.read_parquet(os.path.join(D, "bets_ml.parquet")), on="bet_id", how="left")
    else:
        b["ml"] = np.nan
    b = b[(b.usd >= MIN_USD) & (b.mins_to_kick >= 30) & b.m_d0.notna()].copy()
    # only bets that can satisfy at least one rule family in s07 (keeps the sim small)
    sk = b.s_n >= 20
    wal = sk & ((b.s_move.abs() >= 0.003) | (b.s_res.abs() >= 0.02) | (b.s_clv >= 0.005) | (b.s_tmove >= 1.5))
    ins = (b.usd >= 500) & (b.fresh | (b.cluster_n >= 2) | (b.max_levels >= 3) | b.mins_to_kick.between(50, 80) |
                            (b.share >= 0.1) | ((b.prior_games >= 5) & (b.series_conc >= 0.8)))
    b = b[wal | ins | (b.usd >= 1000) | (b.ml.abs() >= 0.005)]
    print("universe bets", len(b), flush=True)
    t = pd.read_parquet(os.path.join(D, "trades_marked.parquet"),
                        columns=["unique_id", "k", "taker", "ts", "ypx", "dir", "size"])
    t = t[t.taker].sort_values("ts")
    gr, uids = load_grid()
    grg = {u: g for u, g in gr.groupby("unique_id")}
    prg = {kk: g for kk, g in t.groupby(["unique_id", "k"])}
    jobs = []
    for (u, k), g in b.groupby(["unique_id", "k"]):
        if (u, k) not in prg or u not in grg:
            continue
        p = prg[(u, k)]; gg = grg[u]
        pr = (p.ts.to_numpy().astype(float), p.ypx.to_numpy().astype(np.float64).round(4), p.dir.to_numpy(),
              p["size"].to_numpy().astype(float))
        gt = gg.t_min.to_numpy().astype(float)
        gp = gg[f"p{int(k)}"].to_numpy().astype(np.float64).round(4)
        jobs.append((g[["bet_id", "ts", "dir"]], pr, gt, gp, float(g.kick_ts.iloc[0]), float(gp[-1]),
                     float(hs_of(np.array([g.poly_volume.iloc[0]]))[0])))
    del prg, grg, t
    print("jobs", len(jobs), flush=True)
    res = []
    with ProcessPoolExecutor(7) as ex:
        for i, r in enumerate(ex.map(sim_group, jobs, chunksize=64)):
            res.extend(r)
            if i % 5000 == 0:
                print(i, len(res), flush=True)
    cols = ["bet_id", "side", "delay", "dmin", "entry", "filled", "pe", "cost", "tf", "cap"] + \
           [f"p_d{int(d * 100)}" for d in DS] + ["p_close"]
    s = pd.DataFrame(res, columns=cols)
    s["side"] = s.side.astype(np.int8); s["delay"] = s.delay.astype(np.int8); s["filled"] = s.filled.astype(np.int8)
    s["entry"] = s.entry.astype("category")
    for c in ["dmin", "pe", "cost", "cap"] + [f"p_d{int(d * 100)}" for d in DS] + ["p_close"]:
        s[c] = s[c].astype(np.float32)
    s.to_parquet(os.path.join(D, "sim.parquet"))
    print("sim rows", len(s), "filled", s.filled.mean(), flush=True)


if __name__ == "__main__":
    main()
