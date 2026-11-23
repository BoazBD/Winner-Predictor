"""Walk-forward backtest at full 15-minute resolution.

  python backtest.py dev     folds inside kickoff < 2026-03-01 (refit before each fold)
  python backtest.py final   fit ONCE on kickoff < 2026-03-01, score kickoff 2026-03-01 .. end   (run once)

Scoring: every market is scored at its compact decision points (price changes, reappearances, time checkpoints);
every segment that has a compact point with EV > CAND_EV is then re-scored at EVERY snapshot, so the bet is
placed at the first real snapshot where the rule fires. Writes work/bt_<mode>_points.parquet (compact points,
for log-loss) and work/bt_<mode>_full.parquet (all snapshots of candidate segments, for bets)."""
import sys, time
import numpy as np, pandas as pd
from features import seg_stats, make_points, make_points_full
from table import long_table, closing_state
import model_lib as ml

W = "/Users/boazbardavid/Desktop/Winner/models/oddsflow/work"
SPLIT = pd.Timestamp("2026-03-01")
DEV_FOLDS = ["2024-10-01", "2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", "2026-01-01", "2026-03-01"]
CAND_EV = -0.06
KEEP = ["pt", "outcome", "unique_id", "mkick", "t", "kick", "kind", "tier", "rank", "own_o", "own_m", "own_l", "own_sl",
        "ttk_h", "won", "c_o", "c_l", "c_close_gap_h", "seg_row"]


def main(mode):
    t0 = time.time()
    seg = pd.read_parquet(f"{W}/segments.parquet")
    mk = pd.read_parquet(f"{W}/markets.parquet")
    runs = pd.read_parquet(f"{W}/runs.parquet").sort_values("run_idx").run_ts.values.astype("datetime64[us]")
    if mode == "dev":                                   # the later period is never loaded in dev mode
        mk = mk[mk.kick < SPLIT]
        seg = seg[seg.unique_id.isin(mk.unique_id)].reset_index(drop=True)
        folds = list(zip(DEV_FOLDS[:-1], DEV_FOLDS[1:]))
    else:
        folds = [("2026-03-01", "2100-01-01")]
    st = seg_stats(seg)
    close = closing_state(st)
    pts = make_points(seg, runs)
    L = long_table(st, pts, mk, close)
    L["seg_row"] = np.tile(pts.seg_row.values, 3)
    L["mkick"] = L.unique_id.map(mk.set_index("unique_id").kick).values
    for c in L.columns:
        if L[c].dtype == np.float64:
            L[c] = L[c].astype(np.float32)
    print("table", L.shape, round(time.time() - t0), flush=True)
    P, F = [], []
    for a, b in folds:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        art = ml.fit(L[L.mkick < a])
        te = L[(L.mkick >= a) & (L.mkick < b)]
        x = te[KEEP].join(ml.predict_long(art, te))
        x["p_close"] = ml.close_prob(art, x.c_l.fillna(0).values, x.tier.values, x.outcome.values)
        x["fold"] = str(a.date())
        P.append(x)
        cand = np.unique(x.seg_row[x.ev > CAND_EV].values)
        if len(cand):
            fp = make_points_full(seg, runs, cand)
            fl = long_table(st, fp, mk, close)
            fl["seg_row"] = np.tile(fp.seg_row.values, 3)
            fl["mkick"] = fl.unique_id.map(mk.set_index("unique_id").kick).values
            y = fl[KEEP].join(ml.predict_long(art, fl))
            y["p_close"] = ml.close_prob(art, y.c_l.fillna(0).values, y.tier.values, y.outcome.values)
            y["fold"] = str(a.date())
            F.append(y)
        print(a.date(), "train", int((L.mkick < a).sum()), "test", len(te), "cand segs", len(cand),
              "full rows", len(F[-1]) if len(cand) else 0, round(time.time() - t0), flush=True)
    pd.concat(P).to_parquet(f"{W}/bt_{mode}_points.parquet")
    pd.concat(F).to_parquet(f"{W}/bt_{mode}_full.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
