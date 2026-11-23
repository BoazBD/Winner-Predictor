"""s07: rule search on TRAIN only, check on VALIDATION, look at TEST once (research questions 1, 2, 4, 5).

Signal families (each a filter on s05 bets + copy or fade) x execution choices; one trade per
(game, outcome, our direction) = the first qualifying bet. Execution for selection uses the production case:
detection delay uniform 0-60 min (hourly job). Every combination's train result is saved (out/s07_all_train.csv);
the count of combinations tried is reported. Selection score = mean return per $ - 2 * standard error (games
resampled analytically via per-game sums), with at least 60 train trades.
Finalists (best of each family) get validation + test, all delays, CIs, fills/month, $/month at $100/trade, worst day,
capacity. Output: out/s07_finalists.csv, out/s07_delays.csv, out/fig_s07_cum.png
"""
import os, itertools, json
import numpy as np, pandas as pd
from common import D, OUT, period, boot_ci

MIN_TRAIN = 60


def families(b):
    """name -> list of (params, mask, side)."""
    F = {}
    sk = b.s_n >= 20
    F["copy_move_skill"] = [(dict(th=th, usd=u), sk & (b.s_move >= th) & (b.usd >= u), 1)
                            for th in [0.003, 0.005, 0.01, 0.02] for u in [200, 1000, 5000]]
    F["copy_move_tstat"] = [(dict(t=th, usd=u), sk & (b.s_tmove >= th) & (b.usd >= u), 1)
                            for th in [1.5, 2.0, 3.0] for u in [200, 1000, 5000]]
    F["fade_move_unskilled"] = [(dict(th=th, usd=u), sk & (b.s_move <= -th) & (b.usd >= u), -1)
                                for th in [0.003, 0.005, 0.01] for u in [200, 1000, 5000]]
    F["copy_result_skill"] = [(dict(th=th, usd=u), sk & (b.s_res >= th) & (b.usd >= u), 1)
                              for th in [0.02, 0.05, 0.1] for u in [200, 1000, 5000]]
    F["fade_result_unskilled"] = [(dict(th=th, usd=u), sk & (b.s_res <= -th) & (b.usd >= u), -1)
                                  for th in [0.05, 0.1] for u in [200, 1000, 5000]]
    F["copy_clv_skill"] = [(dict(th=th, usd=u), sk & (b.s_clv >= th) & (b.usd >= u), 1)
                           for th in [0.005, 0.01, 0.02] for u in [200, 1000, 5000]]
    for side, nm in [(1, "copy"), (-1, "fade")]:
        F[f"{nm}_fresh_big"] = [(dict(usd=u, share=s), b.fresh & (b.usd >= u) & (b.share >= s), side)
                                for u in [500, 2000, 10000] for s in [0, 0.05, 0.2]]
        F[f"{nm}_cluster"] = [(dict(n=n, usd=u), (b.cluster_n >= n) & (b.usd >= u), side)
                              for n in [2, 3, 5] for u in [200, 1000]]
        F[f"{nm}_sweep"] = [(dict(lv=lv, usd=u), (b.max_levels >= lv) & (b.usd >= u), side)
                            for lv in [3, 5, 8] for u in [1000, 5000]]
        F[f"{nm}_lineup_window"] = [(dict(usd=u), (b.mins_to_kick.between(50, 80)) & (b.usd >= u), side)
                                    for u in [1000, 5000, 20000]]
        F[f"{nm}_big_share"] = [(dict(share=s, usd=u), (b.share >= s) & (b.usd >= u), side)
                                for s in [0.1, 0.3, 0.5] for u in [1000, 5000]]
        F[f"{nm}_concentrated"] = [(dict(conc=c, usd=u), (b.prior_games >= 5) & (b.series_conc >= c) & (b.usd >= u), side)
                                   for c in [0.8, 0.95] for u in [500, 2000]]
        F[f"{nm}_all_big"] = [(dict(usd=u), b.usd >= u, side) for u in [1000, 5000, 20000, 50000]]
    if "ml" in b:
        F["copy_ml"] = [(dict(q=q), b.ml >= q, 1) for q in [0.005, 0.01, 0.02, 0.03]]
        F["fade_ml"] = [(dict(q=q), b.ml <= -q, -1) for q in [0.005, 0.01, 0.02]]
    return F


EXEC = [(e, d, band) for e in ["taker", "mk0", "mk1"] for d in ["p_d1", "p_d2", "p_d3", "p_d5", "p_close"]
        for band in [(0.05, 0.95), (0.15, 0.85)]]


def evaluate(tr):
    """tr: trades with ret, unique_id -> mean, se (game-clustered), n."""
    n = len(tr)
    if n == 0:
        return np.nan, np.nan, 0
    g = tr.groupby("unique_id").ret.agg(["sum", "size"])
    m = g["sum"].sum() / g["size"].sum()
    # cluster-robust SE of the ratio estimator
    r = g["sum"] - m * g["size"]
    se = np.sqrt((r ** 2).sum()) / g["size"].sum()
    return m, se, n


def pick_trades(b, mask, side, sim, delay, entry, dcol, band):
    """first qualifying bet per (game,k,our dir) -> its sim row (filled ones get ret)."""
    x = b.loc[mask, ["bet_id", "unique_id", "k", "dir", "ts", "kick_ts", "period"]].copy()
    x["od"] = x.dir * side
    x = x.sort_values("ts").drop_duplicates(["unique_id", "k", "od"])
    s = sim[(delay, entry, side)]
    y = x.join(s, on="bet_id", how="inner")
    y = y[(y.filled == 1) & (y.pe >= band[0]) & (y.pe <= band[1])].copy()
    y["ret"] = (y[dcol].astype(float) - y.cost.astype(float)) / y.cost.astype(float)
    return y, len(x)


def main():
    b = pd.read_parquet(os.path.join(D, "bets.parquet"))
    b["bet_id"] = b.index.astype(np.int64)
    if os.path.exists(os.path.join(D, "bets_ml.parquet")):
        b = b.merge(pd.read_parquet(os.path.join(D, "bets_ml.parquet")), on="bet_id", how="left")
    sim = pd.read_parquet(os.path.join(D, "sim.parquet"))
    b = b[b.bet_id.isin(sim.bet_id.unique())]
    sim["entry"] = sim.entry.astype(str)
    sim = {kk: g.drop(columns=["delay", "entry", "side"]).set_index("bet_id") for kk, g in sim.groupby(["delay", "entry", "side"])}
    F = families(b)
    rows = []
    ncomb = 0
    for fam, lst in F.items():
        for params, mask, side in lst:
            for entry, dcol, band in EXEC:
                y, nsig = pick_trades(b, mask & (b.period == "train"), side, sim, -1, entry, dcol, band)
                m, se, n = evaluate(y)
                ncomb += 1
                rows.append(dict(family=fam, params=json.dumps(params), side=side, entry=entry, exit=dcol,
                                 band=f"{band[0]}-{band[1]}", signals=nsig, trades=n, mean=m, se=se,
                                 score=(m - 2 * se) if n >= MIN_TRAIN else np.nan))
        print(fam, ncomb, flush=True)
    allr = pd.DataFrame(rows)
    allr.to_csv(os.path.join(OUT, "s07_all_train.csv"), index=False)
    print("combinations tried on train:", ncomb, "with >= %d trades:" % MIN_TRAIN, int(allr.score.notna().sum()))
    print("share of combos with train mean > 0:", round(float((allr[allr.score.notna()]["mean"] > 0).mean()), 3))
    best = allr[allr.score.notna()].sort_values("score", ascending=False).groupby("family").head(1)
    fin, dl_rows = [], []
    for r in best.itertuples():
        params = json.loads(r.params)
        mask = [m for p, m, s in F[r.family] if p == params][0]
        band = tuple(float(v) for v in r.band.split("-"))
        for delay in [5, 15, 60, -1]:
            for per in ["train", "val", "test"]:
                y, nsig = pick_trades(b, mask & (b.period == per), r.side, sim, delay, r.entry, r.exit, band)
                if not len(y):
                    continue
                m, lo, hi = boot_ci(y.ret, y.unique_id, n=1000)
                months = {"train": 10, "val": 3, "test": 1.6}[per]
                prof = 100 * y.ret
                day = pd.to_datetime(y.kick_ts, unit="s").dt.date
                rec = dict(family=r.family, params=r.params, side=r.side, entry=r.entry, exit=r.exit, band=r.band,
                           delay=delay, period=per, signals=nsig, trades=len(y), fill_rate=len(y) / max(nsig, 1),
                           mean=m, lo=lo, hi=hi, win=float((y.ret > 0).mean()),
                           trades_per_month=len(y) / months, usd_per_month=prof.sum() / months,
                           worst_day=float(prof.groupby(day).sum().min()),
                           cap_median_shares=float(y.cap.median()),
                           cap_ge_100usd=float((y.cap * y.pe >= 100).mean()),
                           limit_sold=float((y[r.exit] != y.p_close).mean()) if r.exit != "p_close" else 0.0)
                dl_rows.append(rec)
                if delay == -1:
                    fin.append(rec)
    fin = pd.DataFrame(fin); dl = pd.DataFrame(dl_rows)
    fin.to_csv(os.path.join(OUT, "s07_finalists.csv"), index=False)
    dl.to_csv(os.path.join(OUT, "s07_delays.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 300):
        print(fin[["family", "params", "entry", "exit", "band", "period", "trades", "mean", "lo", "hi", "usd_per_month",
                   "cap_ge_100usd"]].round(4).to_string())
        print(dl.pivot_table(index=["family", "period"], columns="delay", values="mean").round(4).to_string())


if __name__ == "__main__":
    main()
