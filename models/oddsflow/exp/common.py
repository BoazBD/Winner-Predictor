"""Shared helpers for experiments. load_long() returns DEV rows only (kick < 2026-03-01) unless final=True."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/boazbardavid/Desktop/Winner/models/oddsflow")
from features import *
W = "/Users/boazbardavid/Desktop/Winner/models/oddsflow/work"
SPLIT = pd.Timestamp("2026-03-01")

def load_long(cols=None, final=False):
    flt = [("kick", ">=" if final else "<", SPLIT)]
    L = pd.read_parquet(f"{W}/long.parquet", columns=cols, filters=flt)
    return L

def boot_ci(profit, groups, n=2000, seed=0, q=(5, 95)):
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"p": np.asarray(profit, float), "g": np.asarray(groups)}).groupby("g").p.agg(["sum", "size"]).values
    if len(g) == 0: return (np.nan, np.nan)
    idx = rng.integers(0, len(g), (n, len(g)))
    return np.percentile(g[idx, 0].sum(1) / g[idx, 1].sum(1), q)

def roi_line(b, name=""):
    """b: bets frame with own_o, won, unique_id"""
    if len(b) == 0: return f"{name:28s} n=0"
    pr = b.won * b.own_o - 1
    lo, hi = boot_ci(pr, b.unique_id)
    wk = (b.kick.max() - b.kick.min()).days / 7 + 1e-9
    return f"{name:28s} n={len(b):6d} roi={100*pr.mean():+6.2f}% [{100*lo:+.1f},{100*hi:+.1f}] odds={b.own_o.mean():.2f} /wk={len(b)/wk:.1f}"
