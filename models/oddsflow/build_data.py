"""Compress raw 15-minute snapshots of soccer 1X2 markets into price segments (odds_history.soccer_1x2_history,
the cleaning step shared with the live API).

Outputs (in OUT):
  runs.parquet      run_idx, run_ts            every scraper run (global, all sports)
  segments.parquet  unique_id, seg_no, t_start, t_end, i_start, i_end, n_snaps, r1, r2, r3, kick, gap_runs
  markets.parquet   unique_id, league, kick (last seen), n_kicks, first_seen, last_seen, y (0/1/2, -1 unknown), big
"""
import os
import sys
import duckdb
import numpy as np
import pandas as pd

ROOT = "/Users/boazbardavid/Desktop/Winner"
sys.path.insert(0, os.path.join(ROOT, "models"))
from odds_history import soccer_1x2_history

LABELS = ("bet1_won", "tie_won", "bet2_won")


def build(src=f"{ROOT}/all_processed_winner_081126.parquet", out=f"{ROOT}/models/oddsflow/work",
          big=f"{ROOT}/processed_winner_081126.parquet"):
    os.makedirs(out, exist_ok=True)
    has_res = all(c in duckdb.sql(f"select * from '{src}' limit 0").columns for c in LABELS)
    seg, runs = soccer_1x2_history(src, keep=("league",) + (LABELS if has_res else ()))
    pd.DataFrame({"run_ts": runs, "run_idx": np.arange(len(runs))}).to_parquet(f"{out}/runs.parquet", index=False)
    seg.drop(columns=["result_id", "league", *LABELS], errors="ignore").to_parquet(f"{out}/segments.parquet", index=False)
    y = ("case when bool_or(bet1_won) then 0 when bool_or(tie_won) then 1 when bool_or(bet2_won) then 2 else -1 end"
         if has_res else "-1")
    big_sql = f"(select distinct league from '{big}')" if big and os.path.exists(big) else "('')"
    mk = duckdb.sql(f"""
        select unique_id, any_value(league) as league, arg_max(kick, t_start) as kick, count(distinct kick) as n_kicks,
               min(t_start) as first_seen, max(t_end) as last_seen,
               {y} as y,
               (any_value(league) in {big_sql}) as big
        from seg group by unique_id order by unique_id""").df()
    mk.to_parquet(f"{out}/markets.parquet", index=False)
    print(f"segments={len(seg)} markets={len(mk)} labelled={(mk.y >= 0).sum()} big={mk.big.sum()}")


if __name__ == "__main__":
    build(*sys.argv[1:])
