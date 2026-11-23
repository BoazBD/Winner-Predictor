"""Shared helpers: load signals + simulated executions with game metadata, split labels, bootstrap."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.join(HERE, "..", "lead_lag")
TRAIN_END = pd.Timestamp("2026-07-01")
VAL_END = pd.Timestamp("2026-10-01")
EXITS = ("pnl_kick", "pnl_hold", "pnl_t180", "pnl_t720")


def split_of(kick):
    return np.where(kick < TRAIN_END, "train", np.where(kick < VAL_END, "val", "test"))


def load_meta():
    g = pd.read_parquet(os.path.join(LL, "data", "games_meta.parquet"))
    g = g[g.ok == True].copy()
    g["kick"] = pd.to_datetime(np.minimum(g.kick_utc.values, g.start.values))
    g["split"] = split_of(g.kick)
    # volume filter without look-ahead: typical final volume of the same Polymarket series in the TRAIN period
    tr = g[g.split == "train"]
    sv = tr.groupby("series").poly_volume.median()
    g["series_vol"] = g.series.map(sv)
    g["series_vol"] = g.series_vol.fillna(tr.poly_volume.median())
    return g[["unique_id", "event_id", "kick", "split", "tier", "league", "series", "poly_volume", "series_vol",
              "created", "swapped"]].rename(columns={"unique_id": "uid"})


def load_all():
    S = pd.read_parquet(os.path.join(HERE, "data", "signals.parquet"))
    M = pd.read_parquet(os.path.join(HERE, "data", "sims.parquet"))
    g = load_meta()
    S = S.merge(g, on="uid", how="inner")
    S["t_utc"] = pd.to_datetime(S.t_min * 60, unit="s")
    S["htk"] = S.mtk / 60.0
    S["hour_il"] = (S.t_utc.dt.tz_localize("UTC").dt.tz_convert("Asia/Jerusalem").dt.hour)
    S["day"] = S.t_utc.dt.floor("D")
    S["month"] = S.kick.dt.to_period("M").astype(str)
    return S, M


def boot_ci(df, col, by="uid", n=1000, seed=0, denom=None):
    """95% CI of the mean of col (or sum(col)/sum(denom)) resampling games."""
    if len(df) == 0:
        return (np.nan, np.nan)
    gsum = df.groupby(by)[col].sum().to_numpy()
    gcnt = (df.groupby(by)[denom].sum() if denom else df.groupby(by).size()).to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gsum), size=(n, len(gsum)))
    est = gsum[idx].sum(1) / gcnt[idx].sum(1)
    return tuple(np.percentile(est, [2.5, 97.5]))
