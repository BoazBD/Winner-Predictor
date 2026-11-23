"""Build the long decision-point table used by all experiments (work/long.parquet)."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/boazbardavid/Desktop/Winner/models/oddsflow")
from features import *

W = "/Users/boazbardavid/Desktop/Winner/models/oddsflow/work"
t0 = time.time()
seg = pd.read_parquet(f"{W}/segments.parquet")
runs = pd.read_parquet(f"{W}/runs.parquet").sort_values("run_idx").run_ts.values.astype("datetime64[us]")
mk = pd.read_parquet(f"{W}/markets.parquet")
st = seg_stats(seg)
pts = make_points(seg, runs)
print("points", len(pts), pts.kind.value_counts().to_dict(), round(time.time() - t0))
f = point_features(st, pts)
f["kind"] = pts.kind.values
f["run_idx"] = pts.run_idx.values
# closing state per market = last point of the market whose last snapshot is within 60 min of kickoff
last = st.groupby("unique_id").tail(1)
close = last[["unique_id", "l0", "l1", "l2", "o0", "o1", "o2", "sl0", "sl1", "sl2"]].copy()
close["close_gap_h"] = (last.kick_h - (last.t_end.values.astype("datetime64[s]").astype(np.int64) / 3600.0)).values
close.columns = ["unique_id"] + [f"c_{c}" for c in close.columns[1:]]
f = f.merge(close, on="unique_id", how="left").merge(mk[["unique_id", "y", "big", "league"]], on="unique_id", how="left")
L = to_long(f)
for c in ("unique_id", "t", "kick", "kind", "run_idx", "y", "big", "league", "c_close_gap_h"):
    L[c] = np.tile(f[c].values, 3)
for c in ("l", "o", "sl"):
    L[f"c_{c}"] = np.concatenate([f[f"c_{c}{j}"].values for j in range(3)])
L["won"] = (L.y == L.outcome).astype(np.int8)
for c in L.columns:
    if L[c].dtype == np.float64:
        L[c] = L[c].astype(np.float32)
L["league"] = L.league.astype("category")
L.to_parquet(f"{W}/long.parquet")
print(L.shape, round(time.time() - t0), "s")
print(L[FEATS].describe().T[["mean", "min", "max"]].round(3).to_string())
