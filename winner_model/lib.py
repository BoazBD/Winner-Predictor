"""Shared loaders / helpers. All experiments must respect the split:
   DEV   : kickoff <  2026-03-01   (free to explore, walk-forward inside)
   FINAL : kickoff >= 2026-03-01   (touch ONLY for pre-registered final evaluation)
"""
import re
import numpy as np
import pandas as pd

D = "/Users/boazbardavid/Desktop/Winner/winner_model/data"
FINAL_START = pd.Timestamp("2026-03-01")


def load_markets():
    m = pd.read_parquet(f"{D}/markets.parquet")
    ev = m["event"]
    mkt = np.select(
        [ev.str.contains("1X2"),
         (m.sport == "Soccer"),
         ev.str.contains("כולל הארכות"),
         (m.sport == "Tennis"),
         ev.str.contains("ללא הארכות")],
        ["1x2", "hc3", "spread2", "tennis", "spread3"], default="other")
    m["mkt"] = mkt
    # handicap line of option1 (home side), e.g. "וולבס (+1)" -> +1 ; for hc3 / spreads
    def line(s):
        if s is None:
            return np.nan
        g = re.search(r"\(([+-]?\d+(?:\.\d+)?)\)", s)
        return float(g.group(1)) if g else np.nan
    m["line1"] = m["option1"].map(line)
    m.loc[m.mkt.isin(["1x2", "tennis"]), "line1"] = np.nan
    w = [m[c].fillna(False).astype(bool).values for c in ("won1", "wonx", "won2")]
    m["y"] = np.select(w, [0, 1, 2], default=-1)
    m["final"] = m.kickoff >= FINAL_START
    return m


def load_segments(cols=None):
    s = pd.read_parquet(f"{D}/segments.parquet", columns=cols)
    return s


def load_runs():
    return pd.read_parquet(f"{D}/runs.parquet").sort_values("run_idx").reset_index(drop=True)


def roi_table(df, by, profit="profit"):
    g = df.groupby(by, observed=True)[profit].agg(["size", "mean", "std"])
    g["se"] = g["std"] / np.sqrt(g["size"])
    g["roi%"] = 100 * g["mean"]
    g["se%"] = 100 * g["se"]
    return g[["size", "roi%", "se%"]].round(2)


def boot_ci(profit, groups=None, n=2000, seed=0, q=(5, 95)):
    """Bootstrap CI of mean profit, clustered by `groups` (e.g. match id) if given."""
    rng = np.random.default_rng(seed)
    p = np.asarray(profit, float)
    if groups is None:
        idx = rng.integers(0, len(p), (n, len(p)))
        means = p[idx].mean(1)
    else:
        g = pd.Series(p).groupby(np.asarray(groups)).agg(["sum", "size"]).values
        idx = rng.integers(0, len(g), (n, len(g)))
        means = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(means, q)


def to_long(d):
    """Market-level frame (r1,r2,r3,won1,wonx,won2,mkt) -> one row per outcome with odds/won/side.
    side: 0 = home/first, 1 = middle (X), 2 = away/second. 2-way markets have sides 0 and 2."""
    two = d.mkt.isin(["spread2", "tennis"]).values
    out = []
    for side, r, w in [(0, "r1", "won1"), (1, "r2", "wonx"), (2, "r3", "won2")]:
        x = d.copy()
        x["odds"] = x[r].values
        x["won"] = x[w].values.astype(float)
        if side == 1:
            x = x[~two]
        if side == 2:
            x.loc[two, "odds"] = x.loc[two, "r2"]
        x["side"] = side
        out.append(x)
    x = pd.concat(out)
    x = x[x.odds.notna()]
    x["profit"] = x.won * x.odds - 1
    return x
