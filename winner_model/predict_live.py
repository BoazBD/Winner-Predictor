"""Score the latest snapshot.
Usage: .venv/bin/python predict_live.py <raw_snapshots.parquet> [--thr -0.03] [--model data/model_full.pkl]
The parquet must have the same columns as processed_winner_*.parquet and should contain the FULL snapshot
history of the currently open markets (results columns may be null). Output: outcomes to bet NOW, i.e. markets
present in the most recent scraper run whose EV exceeds the threshold. Bet each (market, outcome) at most once.
"""
import sys, os, subprocess, tempfile, pickle, argparse
import numpy as np, pandas as pd, lightgbm as lgb
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
ap=argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('--thr',type=float,default=-0.03)
ap.add_argument('--model',default=os.path.join(HERE,'data/model_full.pkl')); ap.add_argument('--out',default=None)
a=ap.parse_args()
tmp=tempfile.mkdtemp(prefix='winner_live_')
subprocess.run([sys.executable,os.path.join(HERE,'build_segments.py'),os.path.abspath(a.src),tmp,os.path.join(HERE,'..','processed_winner_081126.parquet')],check=True,stdout=subprocess.DEVNULL)
import lib; lib.D=tmp
import build_cand2, model
c=build_cand2.build(out_path=None,add_now=True).reset_index(drop=True)
runs=lib.load_runs(); last_i=int(runs.run_idx.max())
c=model.prep(c)
M=pickle.load(open(a.model,'rb')); bs=[lgb.Booster(model_str=s) for s in M['drift']]
now=c[(c.i==last_i)&(c.hrs_to_ko>0)].copy()
now['P_now']=model.pred_level(M['level'],now).values
now['ev_now']=now.P_now*now.odds-1; now['drift_pred']=model.pred_drift(bs,now); now['ev']=now.ev_now+now.drift_pred
now['p_model']=(1+now.ev)/now.odds
mk=lib.load_markets().set_index('unique_id')
now['outcome']=[mk.at[u,['option1','option2','option3'][s if mk.at[u,'mkt']!='tennis' else (0 if s==0 else 1)]] for u,s in zip(now.unique_id,now.side)]
now['flag']=now.ev>a.thr
cols=['kickoff','league','mkt','outcome','odds','p_model','ev_now','drift_pred','ev','hrs_to_ko','flag','unique_id','side']
res=now.sort_values('ev',ascending=False)[cols]
if a.out: res.to_csv(a.out,index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(f'snapshot {runs.run_ts.max()}  open outcomes scored: {len(res)}  flagged (ev>{a.thr}): {int(res.flag.sum())}')
print(res[res.flag].round(4).to_string(index=False) if res.flag.any() else res.head(10).round(4).to_string(index=False))
