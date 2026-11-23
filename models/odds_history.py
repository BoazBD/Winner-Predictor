"""De-duplicated odds history: the one cleaning step shared by every model, in training and in live scoring.

Winner is scraped about every 15 minutes and a market's prices usually stay the same across many scrapes, so the
raw data repeats the same row again and again. dedup_history() keeps one row per *segment*: a maximal run of
consecutive scraper runs in which the market is present with the same ratio1/2/3 and the same kickoff. A new
segment starts when a price or the kickoff changes, or when the market reappears after missing one or more scraper
runs (a withdrawal). Timing is kept, so no information is lost:
    t_start / t_end, i_start / i_end   first and last scraper run of the segment (time and index in the run list)
    n_snaps                            scraper runs in the segment
    gap_runs                           scraper runs the market was missing just before the segment (0 = none)
The current state of a market that is in the newest scraper run is its last segment (i_end = the newest run).
"""
import duckdb
import numpy as np


def dedup_history(con, rows, keep=()):
    """rows: table / view / parenthesised query in `con` with unique_id, run_idx (index of the scraper run in the
    list of ALL runs, so that a missing run is visible), run_ts, kick, r1, r2, r3, plus the `keep` columns
    (constant per market, e.g. result_id). Several rows of one market in one run: the lowest r1 is kept.
    Returns a duckdb relation ordered by unique_id, seg_no (seg_no starts at 1)."""
    kept = "".join(f", any_value({c}) as {c}" for c in keep)
    return con.sql(f"""
        with d as (
            select * from (select *, row_number() over (partition by unique_id, run_idx order by r1) as _dup from {rows})
            where _dup = 1
        ), f as (
            select *, lag(run_idx) over w as p_idx, lag(r1) over w as p1, lag(r2) over w as p2, lag(r3) over w as p3,
                   lag(kick) over w as pk
            from d window w as (partition by unique_id order by run_idx)
        ), g as (
            select *, sum(case when p_idx is null or run_idx - p_idx > 1 or r1 is distinct from p1
                                    or r2 is distinct from p2 or r3 is distinct from p3 or kick is distinct from pk
                               then 1 else 0 end) over (partition by unique_id order by run_idx) as seg_no,
                   coalesce(run_idx - p_idx - 1, 0) as gap
            from f
        )
        select unique_id, seg_no, min(run_ts) as t_start, max(run_ts) as t_end, min(run_idx) as i_start,
               max(run_idx) as i_end, count(*) as n_snaps, any_value(r1) as r1, any_value(r2) as r2,
               any_value(r3) as r3, any_value(kick) as kick, max(gap) as gap_runs{kept}
        from g group by unique_id, seg_no order by unique_id, seg_no""")


def soccer_1x2_history(src, keep=("result_id",)):
    """De-duplicated history of the soccer 1X2 markets in raw api_odds rows.
    src: DataFrame (strings OK) or parquet path. ALL rows of src (any sport / market) define the scraper runs.
    Prices are rounded to 2 decimals; rows at or after kickoff and prices <= 1 are dropped.
    Returns (segments DataFrame, runs = sorted datetime64[us] array of every scraper run)."""
    con = duckdb.connect()
    con.sql("set enable_progress_bar=false")
    if isinstance(src, str):
        con.sql(f"create view raw as select * from read_parquet('{src}')")
    else:
        con.register("raw", src)
    con.sql("""create table runs as select t as run_ts, row_number() over (order by t) - 1 as run_idx
               from (select distinct try_cast(run_time as timestamp) as t from raw where t is not null)""")
    cols = "".join(f", {c}" for c in keep)
    con.sql(f"""create view rows as select * from (
        select unique_id, try_cast(run_time as timestamp) as run_ts,
               try_strptime(left(event_date::varchar, 10) || ' ' || time, '%Y-%m-%d %H:%M') as kick,
               round(try_cast(ratio1 as double), 2) as r1, round(try_cast(ratio2 as double), 2) as r2,
               round(try_cast(ratio3 as double), 2) as r3{cols}
        from raw where type = 'Soccer' and contains(event, '1X2')) join runs using (run_ts)
        where run_ts < kick and r1 > 1 and r2 > 1 and r3 > 1""")
    seg = dedup_history(con, "rows", keep).df()
    for c in ("t_start", "t_end", "kick"):
        seg[c] = seg[c].astype("datetime64[us]")
    for c in ("seg_no", "i_start", "i_end", "n_snaps", "gap_runs"):
        seg[c] = seg[c].astype(np.int64)
    runs = con.sql("select run_ts from runs order by run_idx").df().run_ts.values.astype("datetime64[us]")
    return seg, runs
