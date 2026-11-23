"""Compress raw 15-min snapshots into run-length 'segments' with the shared cleaning step (odds_history.dedup_history).

Outputs: runs.parquet, segments.parquet, markets.parquet in OUT.
CLI: build_segments.py [SRC] [OUT] [BIG]
"""
import sys, os, duckdb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))   # in the lambda: same folder
from odds_history import dedup_history

def build(SRC, OUT, BIG=None, quiet=False):
    """BIG: path to big-league parquet, or list of league names, or None (-> no big leagues)."""
    _p = (lambda *a: None) if quiet else print
    if isinstance(BIG, (list, tuple, set)):
        big_sql = "(" + ", ".join("'" + str(x).replace("'", "''") + "'" for x in BIG) + ")" if BIG else "('')"
    elif BIG:
        big_sql = f"(select distinct league from '{BIG}')"
    else:
        big_sql = "('')"
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect()
    con.sql(f"set memory_limit='{os.environ.get('DUCKDB_MEM', '9GB')}'")
    con.sql(f"set temp_directory='{OUT}/_tmp'")
    con.sql("set preserve_insertion_order=false")

    con.sql(f"""
    create table runs as
    select run_time::timestamp as run_ts, row_number() over (order by run_time::timestamp) - 1 as run_idx
    from (select distinct run_time from '{SRC}')
    """)
    con.sql(f"copy runs to '{OUT}/runs.parquet' (format parquet)")
    _p("runs", con.sql("select count(*) from runs").fetchall())

    con.sql(f"""
    create table raw as
    select unique_id, r.run_idx, r.run_ts,
           (event_date::date + cast(strptime(time, '%H:%M') as time)) as kick,
           try_cast(ratio1 as float) as r1, try_cast(ratio2 as float) as r2, try_cast(ratio3 as float) as r3
    from '{SRC}' s join runs r on r.run_ts = s.run_time::timestamp
    """)
    _p("raw", con.sql("select count(*) from raw").fetchall())

    dedup_history(con, "(select * from raw where run_ts < kick)").create("seg")
    con.sql(f"""copy (select * exclude (kick, gap_runs), kick as kickoff_seg, gap_runs > 0 as gap_before
                      from seg order by unique_id, seg_no) to '{OUT}/segments.parquet' (format parquet)""")
    _p("segments", con.sql("select count(*) from seg").fetchall())

    con.sql(f"""
    copy (
      select unique_id, any_value(result_id) as result_id, any_value(type) as sport, any_value(league) as league,
             any_value(event) as event, any_value(option1) as option1, any_value(option2) as option2,
             any_value(option3) as option3,
             max(event_date::date + cast(strptime(time, '%H:%M') as time)) as kickoff,
             count(distinct event_date::varchar || time) as n_kickoffs,
             any_value(try_cast(scorea as float)) as scorea, any_value(try_cast(scoreb as float)) as scoreb,
             any_value(final_scorea) as final_scorea, any_value(final_scoreb) as final_scoreb,
             bool_or(bet1_won) as won1, bool_or(tie_won) as wonx, bool_or(bet2_won) as won2,
             count(*) as n_rows, min(run_time::timestamp) as first_seen, max(run_time::timestamp) as last_seen,
             (any_value(league) in {big_sql}) as big_league
      from '{SRC}' group by unique_id
    ) to '{OUT}/markets.parquet' (format parquet)
    """)
    _p("markets", con.sql(f"select count(*), sum(big_league::int) from '{OUT}/markets.parquet'").fetchall())

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "all_processed_winner_081126.parquet",
          sys.argv[2] if len(sys.argv) > 2 else "winner_model/data",
          sys.argv[3] if len(sys.argv) > 3 else "processed_winner_081126.parquet")
