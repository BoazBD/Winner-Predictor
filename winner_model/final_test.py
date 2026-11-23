import pandas as pd, numpy as np, pickle
from model import *
from wf import summarize
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
c=pd.read_parquet('data/cand2.parquet').reset_index(drop=True); c=prep(c)
mc=pd.read_parquet('data/markets_close.parquet'); close_ok=set(mc.index[mc.mins_close<60])
te,(mods,bs)=run_split(c,close_ok,'2026-03-01','2027-01-01')
te.to_parquet('data/final_preds.parquet')
pickle.dump(dict(level=mods,drift=[b.model_to_string() for b in bs]),open('data/model_dev.pkl','wb'))
print('FINAL rows',len(te),te.kickoff.min(),te.kickoff.max())
rows=[]
for col,thr in [('ev',-0.03),('ev',-0.02),('ev_now',-0.04),('ev',-0.04),('ev',-0.01),('ev',0.0),('ev_now',-0.03),('ev_now',-0.02)]:
    b=first_bets(te,col,thr); r=summarize(b,f'{col}>{thr}'); r['clv']=round(100*b.clv.mean(),2) if len(b) else None; rows.append(r)
print(pd.DataFrame(rows).to_string())
b=first_bets(te,'ev',-0.03); b.to_parquet('data/final_bets_primary.parquet')
print(roi_table(b,['grp']).T.to_string()); print(roi_table(b,[pd.cut(b.odds,[1,1.1,1.2,1.3,1.5,2,100])]).T.to_string()); print(roi_table(b,[b.kickoff.dt.to_period('M')]).T.to_string())
# references
cl=closing_rows(te,close_ok); print('flat at close, all outcomes:',summarize(cl,'flat'))
print(roi_table(cl,['grp']).T.to_string())
r=te[(te.mkt=='1x2')&te.is_fav&(te.q_raw>=0.70)&(te.ov<=1.16)&(te.n_chg>=10)]
r=r.sort_values('t').groupby(['unique_id','side']).head(1); print('user rule:',summarize(r,'rule'))
