import sys, time, resource, duckdb, numpy as np, pandas as pd
sys.path.insert(0, "/Users/boazbardavid/Desktop/Winner/models/oddsflow")
from predict_api import Model
import model_lib as ml
from features import seg_stats, make_points_full
from table import long_table
T = "2026-10-24 18:05:00"
c = duckdb.connect()
# live-like input: full history (3 weeks back) of every market present in the newest run, all sports, no result columns
df = c.sql(f"""with src as (select event_date::varchar event_date, time, league, event, option1, ratio1, option2, ratio2, option3, ratio3,
       run_time, result_id, unique_id, date_parsed, type from '../../all_processed_winner_081126.parquet'
       where run_time <= '{T}' and run_time >= '2026-10-03'),
   last as (select max(run_time) m from src)
   select src.* from src where unique_id in (select unique_id from src, last where run_time = last.m)""").df()
print("input rows", len(df), "markets", df.unique_id.nunique(), "soccer 1X2", df[df.event.str.contains("1X2") & (df.type == "Soccer")].unique_id.nunique())
t0 = time.time(); m = Model(); t1 = time.time()
out = m.score(df, latest=True); t2 = time.time()
print(f"load {t1-t0:.2f}s  score(latest) {t2-t1:.2f}s  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6 if sys.platform!='darwin' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.2f} GB")
print(out.dtypes.to_string()); print(out.head(6).to_string())
s = out.groupby(["unique_id"]).p_model.sum(); print("p sums", s.min(), s.max(), "markets", len(s), "flags", int(out.flag.sum()), "flag&new", int((out.flag & ~out.flag_before).sum()))
print(out[out.flag].to_string())
# consistency with the training pipeline (segments built by duckdb from the full file)
W = "work"; seg = pd.read_parquet(f"{W}/segments.parquet"); runs = pd.read_parquet(f"{W}/runs.parquet").sort_values("run_idx").run_ts.values.astype("datetime64[us]")
ids = out.unique_id.unique()
sg = seg[seg.unique_id.isin(ids) & (seg.t_start <= pd.Timestamp(T))].reset_index(drop=True)
first = sg.groupby("unique_id").t_start.min(); ok = first[first > pd.Timestamp("2026-10-03 06:00")].index     # whole history inside the window
sg = sg[sg.unique_id.isin(ok)].reset_index(drop=True)
Tn = np.datetime64(pd.Timestamp(df.run_time.max()))
sg["t_end"] = np.minimum(sg.t_end.values, Tn); sg["i_end"] = np.minimum(sg.i_end.values, np.searchsorted(runs, Tn))
st = seg_stats(sg); pts = make_points_full(sg, runs); L = long_table(st, pts); L = L.join(ml.predict_long(m.art, L))
ref = L.sort_values("t").groupby(["unique_id", "outcome"]).tail(1)[["unique_id", "outcome", "t", "own_o", "p_model", "ev"]]
cmp = out.merge(ref, on=["unique_id", "outcome"], suffixes=("", "_ref"))
print("compared", len(cmp), "rows; time match", (cmp.run_time == cmp.t).mean(), "max |p diff|", np.abs(cmp.p_model - cmp.p_model_ref).max(), "max |odds diff|", np.abs(cmp.odds - cmp.own_o).max())
t3 = time.time(); allo = m.score(df[df.unique_id.isin(ids[:40])], latest=False); print("score(all snapshots, 40 markets)", len(allo), f"{time.time()-t3:.1f}s", "flag_before any", bool(allo.flag_before.any()))
# string-typed input
o2 = m.score(df.astype(str), latest=True); print("string input identical:", np.allclose(o2.p_model, out.p_model))
