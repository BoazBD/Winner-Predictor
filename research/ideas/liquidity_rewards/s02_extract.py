"""Step 2: build the historical fill table for a passive two-sided quoter (pre-match only).

For every real pre-match trade in the last 24h before kickoff (edge/data/master.parquet, k in Winner orientation, verified
against grid: mean |price - grid mid| 0.8c same-orientation vs 23-29c swapped) we attach the most recent 15-min grid
midpoint at or before the trade (ASOF join on mins_to_kick >= trade's minutes-to-kick).
Trade expressed in Yes-k terms:  y = q if taker bought Yes else 1-q;  side = +1 taker bought Yes (lifts asks -> can fill
our ASK), -1 taker sold Yes (hits bids -> can fill our BID).
Output data/fills_base.parquet (float64, prices rounded to 4dp) + data/grid24.parquet (grid rows <= 24h before kick).
"""
import os, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
con = duckdb.connect(); con.execute("PRAGMA threads=2"); con.execute("PRAGMA memory_limit='2GB'")
con.execute(f"""create table g as select g.unique_id, g.mins_to_kick::int mtk, round(g.p0::double,4) p0, round(g.p1::double,4) p1,
  round(g.p2::double,4) p2 from '{R}/lead_lag/data/grid.parquet' g where mins_to_kick <= 1500""")
con.execute(f"copy g to '{HERE}/data/grid24.parquet' (format parquet)")
con.execute(f"""create table t as select unique_id, k::int k, case when yes then 1 else -1 end side,
  round(case when yes then q else 1-q end, 4) y, size, ts, -rel_min mtk_t from '{R}/edge/data/master.parquet'
  where rel_min < 0 and rel_min >= -1440""")
con.execute(f"""copy (select t.*, g.mtk slot_mtk, case t.k when 0 then g.p0 when 1 then g.p1 else g.p2 end mid
  from t asof join g on t.unique_id = g.unique_id and g.mtk >= t.mtk_t) to '{HERE}/data/fills_base.parquet' (format parquet)""")
print(con.sql(f"select count(*), count(mid), count(distinct unique_id) from '{HERE}/data/fills_base.parquet'").df())
