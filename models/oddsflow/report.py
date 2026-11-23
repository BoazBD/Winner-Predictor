"""Metrics for a backtest run:  python report.py dev|final [thr ...]"""
import sys
import numpy as np, pandas as pd

W = "/Users/boazbardavid/Desktop/Winner/models/oddsflow/work"
pd.set_option("display.width", 250)


def boot_ci(profit, groups, n=4000, seed=0, q=(5, 95)):
    g = pd.DataFrame({"p": np.asarray(profit, float), "g": np.asarray(groups)}).groupby("g").p.agg(["sum", "size"]).values
    if len(g) < 2:
        return (np.nan, np.nan)
    idx = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    return np.percentile(g[idx, 0].sum(1) / g[idx, 1].sum(1), q)


def line(b, name, weeks):
    if len(b) == 0:
        return f"{name:26s} n=0"
    pr = b.won * b.own_o - 1
    lo, hi = boot_ci(pr, b.unique_id)
    return (f"{name:26s} n={len(b):6d} ROI={100*pr.mean():+6.2f}% CI90=[{100*lo:+5.1f},{100*hi:+5.1f}] "
            f"avg_odds={b.own_o.mean():.2f} bets/wk={len(b)/weeks:5.1f} win%={100*b.won.mean():.1f}")


def first_bets(F, thr):
    return F[F.ev > thr].sort_values("t", kind="stable").drop_duplicates(["unique_id", "outcome"])


def clv_line(b):
    c = b[b.c_close_gap_h < 1]
    r = c.own_o / c.c_o - 1
    cev = c.p_close * c.own_o - 1
    return (f"   closing line: beat {100*(r > 1e-9).mean():.0f}% / equal {100*(np.abs(r) <= 1e-9).mean():.0f}% / worse "
            f"{100*(r < -1e-9).mean():.0f}%, mean price advantage {100*r.mean():+.2f}%, "
            f"calibrated EV at closing probability {100*cev.mean():+.2f}%, predicted EV {100*b.ev.mean():+.2f}%")


def slack_rule(mode):
    """Reference: 'rounding slack' rule (favourite, odds<=1.45, 90% tier, rr>=1%), first trigger."""
    seg = pd.read_parquet(f"{W}/segments.parquet"); mk = pd.read_parquet(f"{W}/markets.parquet")
    O = seg[["r1", "r2", "r3"]].values; q = 1 / O; ovr = q.sum(1)
    R = np.array([1 / 0.9, 1 / 0.85, 1 / 0.8]); M = R[np.abs(ovr[:, None] - R).argmin(1)]
    j = O.argmin(1); qo = q[np.arange(len(O)), j]; o = O[np.arange(len(O)), j]
    rr = o * (M - (ovr - qo)) - 1
    s = seg[(np.abs(M - 1 / 0.9) < 1e-6) & (o <= 1.45 + 1e-9) & (rr >= 0.01)].assign(outcome=j[(np.abs(M - 1 / 0.9) < 1e-6) & (o <= 1.45 + 1e-9) & (rr >= 0.01)])
    s = s.assign(own_o=s[["r1", "r2", "r3"]].values[np.arange(len(s)), s.outcome.values])
    s = s.sort_values("t_start").drop_duplicates(["unique_id", "outcome"]).merge(mk[["unique_id", "y", "kick"]].rename(columns={"kick": "mkick"}), on="unique_id")
    s["won"] = (s.y == s.outcome).astype(int)
    return s


def main(mode, thrs):
    P = pd.read_parquet(f"{W}/bt_{mode}_points.parquet"); F = pd.read_parquet(f"{W}/bt_{mode}_full.parquet")
    lo_k, hi_k = P.mkick.min(), P.mkick.max()
    weeks = (hi_k - lo_k).days / 7
    print(f"=== {mode.upper()}  kickoff {lo_k.date()} .. {hi_k.date()}  ({weeks:.0f} weeks, {P.unique_id.nunique()} matches) ===")
    # ---- log-loss vs margin-free market probabilities
    P["p_mkt"] = 1 / (1 + np.exp(-P.own_m)); P["p_mkt"] /= P.groupby("pt").p_mkt.transform("sum")
    P["w"] = 1.0 / P.groupby(["unique_id", "outcome"]).pt.transform("size")
    wn = P[P.won == 1]
    ll = lambda d, c: float(np.average(-np.log(d[c]), weights=d.w))
    last = P.loc[P.groupby(["unique_id", "outcome"]).t.idxmax()]; lw = last[last.won == 1]
    print("LOG-LOSS (3-way, mean -log p of the result)")
    rows = {"all decision points (per-match weight)": wn, "last snapshot only": lw}
    for t in (0, 1, 2):
        rows[f"  tier {t} ({['90%','85%','80%'][t]} payout), all points"] = wn[wn.tier == t]
    for k, d in rows.items():
        a, m = ll(d, "p_model"), ll(d, "p_mkt")
        print(f"  {k:46s} model {a:.4f}  market {m:.4f}  diff {a-m:+.4f}  (n={d.unique_id.nunique()})")
    # ---- references
    print("REFERENCES")
    lastp = last.assign(mk=last.mkick)
    print("  " + line(last, "flat: every outcome at close", weeks))
    for t in (0, 1, 2):
        print("  " + line(last[last.tier == t], f"  flat, tier {t}", weeks))
    print("  " + line(last[(last["rank"] == 0)], "  flat, favourites", weeks))
    s = slack_rule(mode); s = s[(s.mkick >= lo_k) & (s.mkick <= hi_k)]
    print("  " + line(s, "rounding-slack rule", weeks))
    print("  value model (winner_model/): dev walk-forward +0.5% / 555 bets; its test Mar-Nov 2026 -0.6% / 317 bets (quoted)")
    # ---- policies
    for thr in thrs:
        b = first_bets(F, thr)
        print(f"POLICY  EV > {thr:+.3f}")
        print("  " + line(b, "all bets", weeks)); 
        if len(b) == 0: continue
        print(clv_line(b))
        print(f"   tiers: " + ", ".join(f"tier{t} {int((b.tier==t).sum())}" for t in (0, 1, 2)) + "; outcomes: " + ", ".join(f"{n} {int((b.outcome==k).sum())}" for k, n in enumerate(["home", "draw", "away"])) +
              f"; median hours to kickoff {b.ttk_h.median():.1f}")
        b = b.assign(period=b.mkick.dt.to_period("Q").astype(str), band=pd.cut(b.own_o, [1, 1.1, 1.2, 1.3, 1.45, 1.7, 2.5, 100]),
                     evb=pd.cut(b.ev, [-1, -0.01, 0, 0.005, 0.01, 0.02, 1]))
        for col in ("period", "band", "evb"):
            print(f"   by {col}:")
            for k, g in b.groupby(col, observed=True):
                c = g[g.c_close_gap_h < 1]
                print("     " + line(g, str(k), weeks) + f" closeEV={100*(c.p_close*c.own_o-1).mean():+.2f}%")
    return


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2:]] or [-0.02, -0.01, 0.0, 0.01])
