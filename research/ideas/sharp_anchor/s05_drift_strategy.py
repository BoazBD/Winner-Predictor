"""s05 - does Polymarket drift toward the sharp anchor after T, and can it be traded?

Gap per outcome: gap = anchor_T - poly_T (both margin-free; poly normalised by its 3-way sum).
(1) Drift regression: d_poly = p_X - p_T (raw midpoints, exit = last grid row >=15 min before kick) on gap, pooled over
    outcomes, slope + game-bootstrap CI, by split. Mirror: anchor drift (closing anchor - early anchor) on -gap.
(2) Strategy grid (chosen on TRAIN only): buy outcome k at T when gap >= theta, at ask = p_T + half-spread
    (0.005 if event volume >= $10k else 0.015, from lead_lag s00 live snapshot), taker fee 0.05*p*(1-p) per share on every
    taker fill (applied to all markets, conservative; fee-free variant also reported). Exit: sell at bid = p_X - hs with fee
    ('flip'), or hold to settlement ('hold', game risk). Return per $ = (proceeds - cost)/cost.
    Anchors: bfe (Betfair Exchange early), avg (market average early), mix=(bfe+avg)/2; theta in 7 values; 2 exits
    -> 42 combos. Pinnacle (ps) only exists Sep25-Jan26 -> reported separately, not tradeable later.
    Entry time variants: T (Tue/Fri 18:00 London, conservative) and T-5h (13:00, where s04 says the odds were most
    likely taken; this is a bound, it risks look-ahead if the odds were collected later than 13:00).
(3) Real-print check: where a real taker buy printed in [T, T+60] (ask_T_trade) and a real taker sell in the last 75 min
    (bid_X_trade), recompute flip return with those prices.
Outputs out/drift.csv, out/strategy_grid.csv, out/strategy_best.json, out/fig_drift.png, out/fig_equity.png.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lead_lag", ".pylib"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(1)
FEE = 0.05
THETAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]


def split(k):
    k = pd.to_datetime(k)
    return np.where(k < "2026-07-01", "train", np.where(k < "2026-10-01", "val", np.where(k < "2026-11-19", "test", "live")))


def boot_game(df, col, n=2000):
    """mean of col with CI resampling games (ratio of sums for returns: sum pnl / sum cost)."""
    g = df.groupby("unique_id")[[col, "cost"]].sum() if col == "pnl" else df.groupby("unique_id")[[col]].agg(["sum", "count"])
    if col == "pnl":
        a, b = g.pnl.to_numpy(), g.cost.to_numpy()
    else:
        a, b = g[(col, "sum")].to_numpy(), g[(col, "count")].to_numpy().astype(float)
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(a), (n, len(a)))
    m = a[idx].sum(1) / b[idx].sum(1)
    return float(a.sum() / b.sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def slope_ci(df, x, y, n=1000):
    ids = df.unique_id.unique()
    pos = df.groupby("unique_id").indices
    X, Y = df[x].to_numpy(), df[y].to_numpy()
    b = float(np.polyfit(X, Y, 1)[0])
    bs = []
    for _ in range(n):
        s = rng.choice(ids, len(ids))
        ii = np.concatenate([pos[u] for u in s])
        bs.append(np.polyfit(X[ii], Y[ii], 1)[0])
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    P = pd.read_parquet(os.path.join(HERE, "data", "panel.parquet"))
    prof = pd.read_parquet(os.path.join(HERE, "data", "profile.parquet"))
    for c in ["p_T", "p_X", "p_C", "w_T", "psum_T", "ask_T_trade", "bid_X_trade"]:
        P[c] = P[c].astype(np.float64).round(4)
    P["split"] = split(P.kick)
    P = P[(P.split != "live") & P.p_T.notna() & P.p_X.notna() & (P.h_T_to_kick >= 1) & (P.mtk_X <= 120)].copy()
    P["pn_T"] = P.p_T / P.psum_T
    P["mix"] = (P.bfe + P.avg) / 2
    P["hs"] = np.where(P.ev_volume >= 1e4, 0.005, 0.015)
    P["d_poly"] = (P.p_X - P.p_T).round(4)
    P["d_poly_close"] = (P.p_C - P.p_T).round(4)
    # T-5h entry variant from hourly profile (offset -5 -> mins 18:00-5h window [13:00,13:15))
    pr = prof[prof.off_h == -5].copy()
    pm = pr.melt(id_vars=["unique_id"], value_vars=["p0", "p1", "p2"], var_name="kk", value_name="p_T5")
    pm["k"] = pm.kk.str[-1].astype(int)
    ps = pr.set_index("unique_id")[["p0", "p1", "p2"]].astype(np.float64).sum(axis=1).rename("psum_T5")
    P = P.merge(pm[["unique_id", "k", "p_T5"]], on=["unique_id", "k"], how="left").merge(ps.reset_index(), on="unique_id", how="left")
    P["p_T5"] = P.p_T5.astype(np.float64).round(4)
    P["pn_T5"] = P.p_T5 / P.psum_T5

    # ---------------- (1) drift regressions
    rows = []
    for an in ["bfe", "avg", "mix", "ps"]:
        for sp in ["train", "val", "test", "all"]:
            d = P[P[an].notna()] if sp == "all" else P[P[an].notna() & (P.split == sp)]
            if d.unique_id.nunique() < 100:
                continue
            d = d.assign(gap=d[an] - d.pn_T)
            b, lo, hi = slope_ci(d, "gap", "d_poly")
            r = dict(anchor=an, split=sp, n_games=d.unique_id.nunique(), slope_poly_toward_anchor=b, lo=lo, hi=hi,
                     mean_abs_gap=float(d.gap.abs().mean()))
            ac = {"bfe": "bfec", "avg": "avgc", "ps": "psc", "mix": None}[an]
            if ac:
                dd = d[d[ac].notna()].assign(d_anchor=lambda z: z[ac] - z[an], ngap=lambda z: z.pn_T - z[an])
                b2, lo2, hi2 = slope_ci(dd, "ngap", "d_anchor", 300)
                r.update(slope_anchor_toward_poly=b2, lo_a=lo2, hi_a=hi2)
            # T-5h variant
            d5 = d[d.pn_T5.notna()].assign(gap5=lambda z: z[an] - z.pn_T5, d5=lambda z: z.p_X - z.p_T5)
            if len(d5) > 300:
                b3, lo3, hi3 = slope_ci(d5, "gap5", "d5", 300)
                r.update(slope_poly_T5=b3, lo_T5=lo3, hi_T5=hi3)
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "drift.csv"), index=False)

    # binned drift chart (bfe, all splits)
    d = P[P.bfe.notna()].assign(gap=lambda z: z.bfe - z.pn_T)
    bins = [-1, -.06, -.04, -.025, -.015, -.005, .005, .015, .025, .04, .06, 1]
    d["bin"] = pd.cut(d.gap, bins)
    bb = d.groupby("bin", observed=True).agg(gap=("gap", "mean"), drift=("d_poly", "mean"), n=("gap", "size"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bb.gap * 100, bb.drift * 100, "o-", label="Polymarket drift T -> 15 min pre-kick")
    ax.plot([-8, 8], [-8, 8], ":", color="grey", label="full convergence")
    ax.axhline(0, color="k", lw=.5)
    for c in (2.0, -2.0):
        ax.axhline(c, color="r", lw=.6, ls="--")
    ax.set_xlabel("gap at T: Betfair-Exchange early minus Polymarket (pts)"); ax.set_ylabel("Polymarket move (pts)")
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.legend(fontsize=8); ax.set_title("Polymarket drift vs sharp-anchor gap (red: ~2c round-trip cost)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_drift.png"), dpi=120)
    print(bb.to_string())

    # ---------------- (2) strategy grid
    def trades(d, an, th, exit_, entry="T", fee_on=True, real=False):
        pn, p = ("pn_T", "p_T") if entry == "T" else ("pn_T5", "p_T5")
        t = d[d[an].notna() & d[pn].notna()]
        t = t[(t[an] - t[pn]) >= th - 5e-5].copy()
        if real:
            t = t[t.ask_T_trade.notna() & t.bid_X_trade.notna()]
            ask, bid = t.ask_T_trade, t.bid_X_trade
        else:
            ask = (t[p] + t.hs).clip(0.01, 0.99)
            bid = (t.p_X - t.hs).clip(0.0, 0.99)
        f = FEE if fee_on else 0.0
        t["cost"] = ask + f * ask * (1 - ask)
        if exit_ == "flip":
            t["proceeds"] = bid - f * bid * (1 - bid)
        else:
            t["proceeds"] = t.won.astype(float)
        t["pnl"] = t.proceeds - t.cost
        t["ret"] = t.pnl / t.cost
        return t

    grid = []
    for an in ["bfe", "avg", "mix"]:
        for th in THETAS:
            for ex in ["flip", "hold"]:
                for sp in ["train", "val", "test"]:
                    t = trades(P[P.split == sp], an, th, ex)
                    m, lo, hi = boot_game(t, "pnl", 500)
                    grid.append(dict(anchor=an, theta=th, exit=ex, split=sp, n=len(t), ret=m, lo=lo, hi=hi,
                                     avg_drift=float((t.p_X - t.p_T).mean()) if len(t) else np.nan))
    Gd = pd.DataFrame(grid); Gd.to_csv(os.path.join(OUT, "strategy_grid.csv"), index=False)
    tr = Gd[(Gd.split == "train") & (Gd.n >= 100)]
    print(Gd.pivot_table(index=["anchor", "theta"], columns=["exit", "split"], values="ret").round(4).to_string())
    best_flip = tr[tr.exit == "flip"].sort_values("ret").iloc[-1]
    best_hold = tr[tr.exit == "hold"].sort_values("ret").iloc[-1]
    print("best flip on train:", best_flip.to_dict()); print("best hold on train:", best_hold.to_dict())

    # ---------------- detailed results for the chosen rules
    def summarise(t, label):
        res = {}
        for sp in ["train", "val", "test"]:
            s = t[t.split == sp]
            if not len(s):
                continue
            m, lo, hi = boot_game(s, "pnl", 2000)
            months = max(1.0, (s.kick.max() - s.kick.min()).days / 30.4)
            day = (s.ret * 100).groupby(s.kick.dt.date).sum()
            res[sp] = dict(n=len(s), ret_per_usd=round(m, 4), ci=[round(lo, 4), round(hi, 4)], trades_per_month=round(len(s) / months, 1),
                           usd_per_month_at_100=round(s.ret.sum() * 100 / months, 0), worst_day_usd=round(float(day.min()), 0),
                           hit_rate=round(float((s.pnl > 0).mean()), 3), median_usd_traded_after_T=round(float(s.usd_after_T.median()), 0),
                           mean_drift_pts=round(float((s.p_X - s.p_T).mean() * 100), 2), mean_gap_pts=round(float((s[an_] - s.pn_T).mean() * 100), 2))
        return {label: res}
    summary = {}
    for lab, row in [("flip", best_flip), ("hold", best_hold)]:
        an_ = row.anchor
        t = trades(P, row.anchor, row.theta, row.exit)
        summary.update(summarise(t, f"{lab}: anchor={row.anchor} theta={row.theta} entry=T(18:00) taker, fee on"))
        t2 = trades(P, row.anchor, row.theta, row.exit, fee_on=False)
        summary.update(summarise(t2, f"{lab}: same, NO fee (pre-Apr-2026 markets / maker-like)"))
        t3 = trades(P, row.anchor, row.theta, row.exit, entry="T5")
        summary.update(summarise(t3, f"{lab}: same, entry T-5h (13:00 London; look-ahead risk)"))
        if lab == "flip":
            t4 = trades(P, row.anchor, row.theta, row.exit, real=True)
            summary.update(summarise(t4, f"{lab}: same, real printed ask/bid only"))
            # zero-cost upper bound: drift at mid
            t5 = trades(P.assign(hs=0.0), row.anchor, row.theta, row.exit, fee_on=False)
            summary.update(summarise(t5, f"{lab}: same, ZERO spread and fee (upper bound)"))
    # Pinnacle era (train only)
    an_ = "ps"
    for ex in ["flip", "hold"]:
        for th in [0.02, 0.03, 0.05]:
            t = trades(P, "ps", th, ex)
            summary.update(summarise(t, f"PINNACLE early, theta={th}, exit={ex} (Sep25-Jan26 only)"))
    json.dump(summary, open(os.path.join(OUT, "strategy_best.json"), "w"), indent=1, default=str)
    for k, v in summary.items():
        print(k); [print("   ", sp, x) for sp, x in v.items()]

    # equity chart for best flip & hold
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, row in [("flip", best_flip), ("hold", best_hold)]:
        t = trades(P, row.anchor, row.theta, row.exit).sort_values("kick")
        ax.plot(t.kick, (t.ret * 100).cumsum(), label=f"{lab} {row.anchor} th={row.theta}")
    for x in ["2026-07-01", "2026-10-01"]:
        ax.axvline(pd.Timestamp(x), color="grey", ls=":")
    ax.axhline(0, color="k", lw=.5); ax.set_ylabel("cumulative $ at $100/trade"); ax.legend(fontsize=8)
    ax.set_title("Sharp-anchor rules, taker at T, after half-spread + fee", fontsize=9)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_equity.png"), dpi=120)
    print("rule combinations tried:", len(Gd) // 3, "(+ fee/entry/real-print/Pinnacle sensitivity variants on the chosen rules)")


if __name__ == "__main__":
    main()
