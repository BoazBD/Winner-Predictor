"""Step 2: Winner soccer 1X2 snapshots (historical parquet + Athena for the days after it) and a games table.
Outputs data/winner_snap.parquet (unique_id, run_utc, r1, r2, r3) and data/winner_games.parquet.
Read-only: Athena results only go to the configured query-output bucket.
"""
import os, duckdb, boto3, awswrangler as wr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(HERE, "data")
START = "2024-09-01"
HIST = os.path.join(ROOT, "all_processed_winner_081126.parquet")
ATH = os.path.join(D, "winner_athena_recent.parquet")
RAW = os.path.join(D, "winner_raw_hist.parquet")

if not os.path.exists(ATH):
    s = boto3.Session(region_name="il-central-1")
    df = wr.athena.read_sql_query(
        "select event_date, time, league, option1, ratio1, option3, ratio3, ratio2, run_time, unique_id, date_parsed "
        "from api_odds where date_parsed >= '2026-11-05' and type = 'Soccer' and event like '%1X2%'",
        database="winner-db", s3_output="s3://winner-athena-output/winner_model/", ctas_approach=False,
        boto3_session=s)
    df["event_date"] = df["event_date"].astype(str)
    df.to_parquet(ATH)

con = duckdb.connect()
con.sql("SET memory_limit='6GB'"); con.sql("SET threads=4"); con.sql("SET preserve_insertion_order=false")
con.sql(f"SET temp_directory='{D}/duck_tmp'")
if not os.path.exists(RAW):
    con.sql(f"""copy (select unique_id, league, option1 home, option3 away, cast(event_date as date) ev_date, time,
        run_time, ratio1, ratio2, ratio3 from read_parquet('{HIST}')
        where type='Soccer' and event like '%1X2%' and run_time >= '{START}') to '{RAW}' (format parquet)""")
con.sql(f"""
create or replace view raw as
select unique_id, league, home, away, ev_date, time, try_cast(run_time as timestamp) run_time,
       try_cast(ratio1 as double) r1, try_cast(ratio2 as double) r2, try_cast(ratio3 as double) r3
from read_parquet('{RAW}')
union all
select unique_id, league, option1, option3, cast(event_date as date), time, try_cast(run_time as timestamp),
       try_cast(ratio1 as double), try_cast(ratio2 as double), try_cast(ratio3 as double)
from read_parquet('{ATH}') where run_time > '2026-11-07 23:00:31'
""")
# kickoff local = event_date + time + 1 minute (Winner lists kickoff 1 minute early); Israel local -> UTC
con.sql("""
create or replace table games as
select unique_id, arg_max(league, run_time) league, arg_max(home, run_time) home, arg_max(away, run_time) away,
       timezone('UTC', timezone('Asia/Jerusalem',
           arg_max(ev_date + cast(time as time) + interval 1 minute, run_time))) kick_utc,
       arg_max(ev_date + cast(time as time) + interval 1 minute, run_time) kick_local,
       count(*) n_rows
from raw where run_time is not null and time is not null group by unique_id
""")
con.sql(f"copy games to '{D}/winner_games.parquet' (format parquet)")
con.sql(f"""copy (
  select r.unique_id, timezone('UTC', timezone('Asia/Jerusalem', r.run_time)) run_utc,
         min(r.r1) r1, min(r.r2) r2, min(r.r3) r3
  from raw r join games g using (unique_id)
  where r.r1 > 1 and r.r2 > 1 and r.r3 > 1 and r.run_time is not null
    and r.ev_date + cast(r.time as time) + interval 1 minute = g.kick_local
  group by 1, 2) to '{D}/winner_snap.parquet' (format parquet)""")
print(con.sql("select count(*), min(kick_utc), max(kick_utc) from games"))
print(con.sql(f"select count(*), count(distinct unique_id) from '{D}/winner_snap.parquet'"))
