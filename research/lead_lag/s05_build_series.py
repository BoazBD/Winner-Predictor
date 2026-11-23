"""Step 5: per matched game, put both venues on one UTC clock and save compact arrays (data/series/<uid>.npz):
  t0      : first minute of the Polymarket grid (unix minutes)
  P       : Polymarket price per outcome (home, draw, away in WINNER's orientation), 1-minute grid, ffilled,
            from max(kickoff-72h, first price point) to kickoff-1min; NaN before the first point
  wt      : Winner snapshot times (unix minutes, float, seconds kept) before kickoff, from kickoff-78h
  Q       : Winner margin-free probabilities at each snapshot (1/odds normalised to sum 1)
  kick    : kickoff (unix minutes) = earlier of Winner's (+1 min) and Polymarket's listed start
Also writes data/games_meta.parquet with per-game metadata and quality checks.
"""
import os, json, glob
from multiprocessing import Pool
import numpy as np, pandas as pd, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); C = os.path.join(HERE, "cache", "prices")
OUT = os.path.join(D, "series"); os.makedirs(OUT, exist_ok=True)
H_POLY = 72 * 60
H_WIN = 78 * 60

TOP = {"פרמייר ליג", "ספרדית ראשונה", "גרמנית ראשונה", "איטלקית ראשונה", "צרפתית ראשונה", "ליגת האלופות",
       "הליגה האירופית", "קונפרנס ליג"}
ISR = {"ליגת העל", "ליגת Winner", "גביע המדינה Winner", "גביע הטוטו"}


def load_tok(tok):
    f = os.path.join(C, f"{tok[-40:]}.json")
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))["h"]
    except (ValueError, KeyError):      # file still being written
        return None


def build(args):
    r, snaps = args
    kick = int(min(pd.Timestamp(r["kick_utc"]), pd.Timestamp(r["start"])).timestamp() // 60)
    toks = [r["tok0"], r["tok1"], r["tok2"]]
    if r["swapped"]:
        toks = [toks[2], toks[1], toks[0]]
    hs = [load_tok(t) for t in toks]
    if any(h is None for h in hs) or min(len(h) for h in hs) == 0:
        return dict(unique_id=r["unique_id"], ok=False, why="no_poly")
    first = max(kick - H_POLY, min(int(h[0]["t"] // 60) for h in hs))
    n = kick - first
    if n < 60:
        return dict(unique_id=r["unique_id"], ok=False, why="short_poly")
    P = np.full((n, 3), np.nan, dtype=np.float32)
    for k, h in enumerate(hs):
        t = np.array([x["t"] // 60 for x in h], dtype=np.int64); p = np.array([x["p"] for x in h], dtype=np.float32)
        idx = np.searchsorted(t, np.arange(first, kick), side="right") - 1   # last point at or before each minute
        ok = idx >= 0
        P[ok, k] = p[idx[ok]]
    s = snaps[(snaps.m < kick) & (snaps.m >= kick - H_WIN)]
    if len(s) < 4:
        return dict(unique_id=r["unique_id"], ok=False, why="no_winner")
    inv = 1.0 / s[["r1", "r2", "r3"]].to_numpy()
    Q = (inv / inv.sum(1, keepdims=True)).astype(np.float32)
    wt = s.m.to_numpy()
    np.savez_compressed(os.path.join(OUT, f"{r['unique_id']}.npz"), t0=first, P=P, wt=wt, Q=Q, kick=kick)
    # quality: level agreement at kickoff-5min and fraction of minutes with a new poly point
    pk = P[-5]; qk = Q[-1]
    nupd = np.mean(np.any(np.abs(np.diff(P[-24 * 60:], axis=0)) > 0, axis=1)) if n > 2 else np.nan
    return dict(unique_id=r["unique_id"], ok=True, why="", n_poly_min=n, n_win=len(s),
                gap_kick=float(np.nanmax(np.abs(pk - qk))), poly_sum=float(np.nansum(pk)),
                poly_update_rate=float(nupd), q_fav=float(qk.max()))


if __name__ == "__main__":
    m = pd.read_parquet(os.path.join(D, "matches.parquet"))
    con = duckdb.connect()
    con.register("m", m[["unique_id"]])
    sn = con.sql(f"""select s.* from '{D}/winner_snap.parquet' s join m using(unique_id)""").df()
    sn["m"] = (pd.to_datetime(sn.run_utc) - pd.Timestamp("1970-01-01")) / pd.Timedelta(minutes=1)
    sn = sn.sort_values(["unique_id", "m"])
    groups = dict(tuple(sn.groupby("unique_id")))
    jobs = [(r, groups[r["unique_id"]]) for r in m.to_dict("records") if r["unique_id"] in groups]
    with Pool(6) as pool:
        res = pool.map(build, jobs, chunksize=50)
    meta = pd.DataFrame(res).merge(m, on="unique_id", how="right")
    meta["tier"] = np.where(meta.league.isin(TOP), "top", np.where(meta.league.isin(ISR), "israel", "other"))
    meta["poly_volume"] = meta[["vol0", "vol1", "vol2"]].sum(axis=1)
    meta.to_parquet(os.path.join(D, "games_meta.parquet"))
    print(meta.ok.value_counts(), meta.why.value_counts())
    print(meta[meta.ok == True][["gap_kick", "poly_sum", "poly_update_rate", "n_poly_min", "n_win"]].describe())
