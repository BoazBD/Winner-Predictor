"""Shared helpers for the wallet study."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
VAL0 = pd.Timestamp("2026-07-01").timestamp()
TEST0 = pd.Timestamp("2026-10-01").timestamp()
TEST1 = pd.Timestamp("2026-11-19").timestamp()


def period(kick_ts):
    k = np.asarray(kick_ts)
    return np.where(k < VAL0, "train", np.where(k < TEST0, "val", np.where(k < TEST1, "test", "after")))


def fee(p):
    """Polymarket sports taker fee per share."""
    p = np.asarray(p, dtype=float)
    return 0.05 * p * (1 - p)


def boot_ci(values, groups, n=1000, seed=0, weights=None):
    """Mean of values with 95% CI by resampling groups (games). Returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    gi, gu = pd.factorize(np.asarray(groups))
    sv = np.bincount(gi, weights=v * w, minlength=len(gu))
    sw = np.bincount(gi, weights=w, minlength=len(gu))
    rng = np.random.default_rng(seed)
    G = len(gu)
    est = []
    for _ in range(n):
        idx = rng.integers(0, G, G)
        c = np.bincount(idx, minlength=G)
        est.append((c * sv).sum() / max((c * sw).sum(), 1e-12))
    return sv.sum() / sw.sum(), np.percentile(est, 2.5), np.percentile(est, 97.5)


def month_starts(t0="2025-10-01", t1="2026-11-01"):
    return pd.date_range(t0, t1, freq="MS")
