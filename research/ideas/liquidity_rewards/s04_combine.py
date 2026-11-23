"""Step 4: rewards (live) + trading P&L (historical) = net, per game, for the only soccer 1X2 markets that carry rewards.

Reward side, from live snapshots data/reward_snapshots.parquet (Yes book only, rate>0 markets):
  our two-sided Q = N * ((v - s)/v)^2 with s = distance (cents) of our quote from the size-adjusted mid (j ticks behind
  the touch), share = Q / (Q + Qcomp), Qcomp = (Q1+Q2)/2 central (sensitivity x0.25, x0.5, x1, x2 is reported).
  $/market/match-day = rate * (pre-match quoted minutes / minutes the market has qualifying makers that day) * share
  where the reward day = 00:00 UTC .. kickoff+115 min, we quote from TSTART to TEND before kickoff.
Trading side, from out/sim_games*.parquet (strict fills, mode 's'; 'a' = at-price optimistic shown for reference).
Rule choice on TRAIN (kickoff < 2026-07-01) EPL games; report train / validation / test (Oct 1-Nov 18) with game-bootstrap
95% CIs. Outputs out/combined_*.csv, out/summary.json, charts.
"""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(0)


def split(s):
    return np.where(s < "2026-07-01", "train", np.where(s < "2026-10-01", "val", np.where(s < "2026-11-19", "test", "live")))


def reward_share_table():
    o = pd.read_parquet(os.path.join(HERE, "data", "reward_snapshots.parquet"))
    o = o[(o.rate > 1) & o.Q1.notna() & (o.hours_to_kick > 0)].copy()
    rows = []
    for j in [0, 1]:
        for N in [1000, 5000, 20000]:
            bid = np.floor(np.round((o.mid_adj - 0.005 - 0.01 * j) * 100, 6) + 1e-6) / 100
            s = ((o.mid_adj - bid) * 100).round(4)
            S = np.where(s < o.max_spread, ((o.max_spread - s) / o.max_spread) ** 2, 0.0)
            q = N * S * (N >= o.min_size)
            for mult in [0.25, 0.5, 1.0, 2.0]:
                qc = mult * (o.Q1 + o.Q2) / 2
                share = q / (q + qc)
                rows.append(dict(j=j, N=N, comp_mult=mult, n_snap_rows=len(o), share_mean=share.mean(), share_med=np.median(share),
                                 rate_mean=o.rate.mean(), S_mean=S.mean(), hours_to_kick_med=o.hours_to_kick.median()))
    return pd.DataFrame(rows), o


def boot(x, B=2000):
    x = np.asarray(x, float)
    if len(x) < 5: return (np.nan, np.nan)
    m = rng.choice(x, (B, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    sh, snaps = reward_share_table()
    sh.to_csv(os.path.join(OUT, "reward_share.csv"), index=False)
    print("live snapshot rows (rewarded EPL markets):", sh.n_snap_rows.iloc[0], "snapshots:", snaps.snap_utc.nunique())
    print(sh.round(4).to_string())
    rate = snaps.rate.mean()
    d = pd.concat([pd.read_parquet(os.path.join(OUT, f)) for f in
                   ["sim_games.parquet", "sim_games_bigN.parquet", "sim_games_queue.parquet"]], ignore_index=True)
    d["UNW"] = d["UNW"].fillna(0).astype(int)
    keys = ["cad", "j", "mode", "N", "IMAX", "TSTART", "TEND", "UNW"]
    d = d.drop_duplicates(keys + ["unique_id"])
    n_configs = d.groupby(keys).ngroups
    d["grp"] = np.where(d.series.fillna("").str.startswith("premier-league"), "EPL", np.where(d.tier == "top", "top", "other"))
    d_all = d[d.grp != "other"].copy()
    d = d[d.grp == "EPL"].copy()
    d["split"] = split(d.start.astype(str))
    # reward per game (3 markets): kick hour determines the reward-day length
    kick_min = pd.to_datetime(d.start).dt.hour * 60 + pd.to_datetime(d.start).dt.minute
    day_minutes = kick_min + 115
    quoted = np.minimum(d.TSTART, kick_min) - d.TEND      # only match-day minutes earn (rewards start 00:00 UTC)
    quoted = quoted.clip(lower=0)
    shc = sh[sh.comp_mult == 1.0].set_index(["j", "N"]).share_mean
    sh_lo = sh[sh.comp_mult == 2.0].set_index(["j", "N"]).share_mean
    sh_hi = sh[sh.comp_mult == 0.25].set_index(["j", "N"]).share_mean
    key = list(zip(d.j, d.N))
    for nm, tab in [("rew", shc), ("rew_lo", sh_lo), ("rew_hi", sh_hi)]:
        s = np.array([tab.get(k, np.nan) for k in key])
        d[nm] = 3 * rate * quoted / day_minutes * s
    d["net"] = d.pnl + d.rew; d["net_hi"] = d.pnl + d.rew_hi; d["net_mtm_hi"] = d.mtm + d.rew_hi
    agg = d.groupby(keys + ["split"]).agg(games=("pnl", "size"), trade_pnl=("pnl", "mean"), mtm=("mtm", "mean"),
                                          exit_cost=("exit_cost", "mean"), rew=("rew", "mean"), rew_hi=("rew_hi", "mean"),
                                          net=("net", "mean"), net_hi=("net_hi", "mean"), net_mtm_hi=("net_mtm_hi", "mean"),
                                          notional=("notional", "mean"), fills=("nfill", "mean")).reset_index()
    agg.to_csv(os.path.join(OUT, "combined_by_config.csv"), index=False)
    SELMODE = "q20000"   # selection uses the central queue assumption (EPL touch depth ~10-40k shares live)
    tr = agg[(agg.split == "train") & (agg["mode"] == SELMODE)].sort_values("net", ascending=False)
    pd.set_option("display.width", 250)
    print("configs tried:", n_configs, " EPL games per split:", d.drop_duplicates("unique_id").split.value_counts().to_dict())
    print("TOP train configs by net (queue 20k ahead, central competition):")
    print(tr.head(12).round(2).to_string())
    best = tr.iloc[0][keys].to_dict()
    res = []
    for fm, sp in [(m, sp) for m in [SELMODE, "s", "q5000", "q50000", "a"] for sp in ["train", "val", "test"]]:
        bm = dict(best, mode=fm)
        sel = d[np.logical_and.reduce([d[k] == v for k, v in bm.items()])]
        x = sel[sel.split == sp]
        if not len(x): continue
        cap = 3 * best["N"]  # ~$1 collateral per share per market (bid + 1-ask)
        r = dict(fill_model=fm, split=sp, games=len(x), net_per_game=x.net.mean(), ci=boot(x.net), trade_pnl=x.pnl.mean(),
                 ci_trade=boot(x.pnl), reward=x.rew.mean(), reward_optimistic=x.rew_hi.mean(),
                 net_optimistic=x.net_hi.mean(), ci_net_opt=boot(x.net_hi), worst_game=x.net.min(),
                 ret_per_capital=x.net.mean() / cap, fills_per_game=x.nfill.mean(), notional_per_game=x.notional.mean(),
                 games_per_month=len(x) / max(1, (pd.to_datetime(x.start).max() - pd.to_datetime(x.start).min()).days / 30.4))
        res.append(r)
    res = pd.DataFrame(res)
    px = d_all[np.logical_and.reduce([d_all[k] == v for k, v in best.items()])].copy()
    px["split"] = split(px.start.astype(str))
    prox = px.groupby(["grp", "split"]).agg(games=("pnl", "size"), trade_pnl=("pnl", "mean"), mtm=("mtm", "mean")).reset_index()
    prox["ci"] = [boot(px[(px.grp == g) & (px.split == s_)].pnl) for g, s_ in zip(prox.grp, prox.split)]
    print("same config, trading P&L only, all top-tier leagues (bigger sample):"); print(prox.round(2).to_string())
    prox.to_csv(os.path.join(OUT, "proxy_top_tier.csv"), index=False)
    print("BEST train config:", best); print(res.round(3).to_string())
    # zero-distance-from-touch baseline at the reward-relevant size, for the report
    base = agg[(agg.TSTART == 900) & (agg.TEND == 15) & (agg.IMAX == 1) & (agg.cad == 15) & (agg.UNW == 120) & (agg.split == "train")]
    print(base.sort_values(["j", "N", "mode"])[["j", "N", "mode", "games", "trade_pnl", "mtm", "exit_cost", "rew", "rew_hi", "net", "net_hi", "notional", "fills"]].round(2).to_string())
    json.dump(dict(best=best, n_configs=int(n_configs), rate_mean=float(rate), results=res.to_dict("records")),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    # chart: per game, trading P&L vs reward income by size at the touch (train, strict)
    b = agg[(agg.split == "train") & (agg.j == 0) & (agg.TSTART == 900) & (agg.TEND == 15) & (agg.IMAX == 1) & (agg.cad == 15) & (agg.UNW == 120)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, c, lb in [("s", "#c0392b", "strict-through fills"), ("q20000", "#e67e22", "queue 20k ahead"),
                        ("q5000", "#f5b041", "queue 5k ahead"), ("a", "#aaaaaa", "at-price (no queue, unrealistic)")]:
        bb = b[b["mode"] == mode].sort_values("N")
        ax.plot(bb.N, bb.trade_pnl, "o-", color=c, label=f"trading P&L, {lb}")
    bb = b[b["mode"] == "s"].sort_values("N")
    ax.plot(bb.N, bb.rew, "s-", color="#2471a3", label="reward income (central)")
    ax.plot(bb.N, bb.rew_hi, "s--", color="#5dade2", label="reward income (competition x0.25)")
    ax.axhline(0, color="k", lw=.5); ax.set_xscale("log"); ax.set_xlabel("shares quoted per side per market (N)")
    ax.set_ylabel("$ per EPL game (3 markets)"); ax.set_title("EPL, quote at touch, 15h match-day window, train")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_reward_vs_trading.png"), dpi=120)


if __name__ == "__main__":
    main()
