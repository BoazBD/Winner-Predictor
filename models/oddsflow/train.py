"""Retrain the oddsflow model from the raw parquet, from scratch.

    python train.py                               # all data  -> ./model/
    python train.py --cutoff 2026-03-01 --out model_dev     # only matches kicking off before the cutoff

Steps: raw snapshots -> price segments (build_data.py, duckdb) -> decision-point features (features.py)
       -> calibration + drift model (model_lib.py) -> model/model.json + model/trees.npz
"""
import argparse, json, os, sys, time, platform
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_data
from features import seg_stats, make_points
from table import long_table
import model_lib as ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "..", "..", "all_processed_winner_081126.parquet"))
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    ap.add_argument("--out", default=os.path.join(HERE, "model"))
    ap.add_argument("--cutoff", default=None, help="train only on matches with kickoff before this date")
    ap.add_argument("--skip-build", action="store_true", help="reuse work/segments.parquet")
    a = ap.parse_args()
    t0 = time.time()
    if not a.skip_build:
        build_data.build(os.path.abspath(a.src), a.work, big=None)
    seg = pd.read_parquet(f"{a.work}/segments.parquet")
    mk = pd.read_parquet(f"{a.work}/markets.parquet")
    runs = pd.read_parquet(f"{a.work}/runs.parquet").sort_values("run_idx").run_ts.values.astype("datetime64[us]")
    mk = mk[mk.y >= 0]
    if a.cutoff:
        mk = mk[mk.kick < pd.Timestamp(a.cutoff)]
    seg = seg[seg.unique_id.isin(mk.unique_id)].reset_index(drop=True)
    L = long_table(seg_stats(seg), make_points(seg, runs), mk)
    for c in L.columns:
        if L[c].dtype == np.float64:
            L[c] = L[c].astype(np.float32)
    print(f"matches {len(mk)}  decision rows {len(L)}  ({time.time()-t0:.0f}s)", flush=True)
    art = ml.fit(L, verbose=True)
    import lightgbm, sklearn, duckdb
    art.update(ev_threshold=ml.EV_THRESHOLD, cutoff=a.cutoff, n_matches=int(len(mk)),
               kick_min=str(mk.kick.min()), kick_max=str(mk.kick.max()),
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                             lightgbm=lightgbm.__version__, sklearn=sklearn.__version__, duckdb=duckdb.__version__))
    ml.save(art, a.out)
    print(f"saved to {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
