"""segments -> long decision-point table (features + labels + closing state)."""
import numpy as np
import pandas as pd
from features import seg_stats, make_points, make_points_full, point_features, to_long


def closing_state(st):
    last = st.groupby("unique_id").tail(1)
    c = pd.DataFrame({"unique_id": last.unique_id.values})
    for j in range(3):
        c[f"c_l{j}"] = last[f"l{j}"].values
        c[f"c_o{j}"] = last[f"o{j}"].values
    c["c_close_gap_h"] = last.kick_h.values - last.t_end.values.astype("datetime64[s]").astype(np.int64) / 3600.0
    return c


def long_table(st, pts, markets=None, close=None):
    """st = seg_stats(segments); pts from make_points*. markets: unique_id, y (optional labels)."""
    f = point_features(st, pts)
    f["kind"] = pts.kind.values
    if close is None:
        close = closing_state(st)
    f = f.merge(close, on="unique_id", how="left")
    if markets is not None:
        f = f.merge(markets[["unique_id", "y"]], on="unique_id", how="left")
    else:
        f["y"] = -1
    L = to_long(f)
    for c in ("unique_id", "t", "kick", "kind", "y", "c_close_gap_h"):
        L[c] = np.tile(f[c].values, 3)
    for c in ("l", "o"):
        L[f"c_{c}"] = np.concatenate([f[f"c_{c}{j}"].values for j in range(3)])
    L["won"] = (L.y == L.outcome).astype(np.int8)
    return L
