import pandas as pd, numpy as np
from lib import *
from wf import *
c=load_dev()
# closing rows
last=c.groupby(['unique_id','side']).seg_no.transform('max')
c['is_close']=(c.seg_no==last)&((c.kickoff-c.t_end)<pd.Timedelta('60min'))
cl=c[c.is_close].copy(); cl['w']=1.0
CF=['lq','lq2','fav','rgain']
tr=cl[cl.kickoff<'2025-01-01']; te=cl[cl.kickoff>='2025-01-01'].copy()
te['z_c']=fit_struct(tr,te,CF); te['P_c']=1/(1+np.exp(-te.z_c))
te['ev_c']=te.P_c*te.odds-1
x=te.dropna(subset=['z_c']).copy(); x['b']=pd.cut(x.ev_c,[-9,-.3,-.2,-.15,-.1,-.07,-.05,-.04,-.03,-.02,-.01,0,9])
print('closing calibrator: predicted vs realised EV at close (OOS 2025-01..2026-02)')
print(x.groupby('b',observed=True).agg(n=('profit','size'),pred=('ev_c','mean'),real=('profit','mean'),se=('profit','sem')).mul([1,100,100,100]).round(2).to_string())
print(x.groupby(['grp',pd.cut(x.odds,[1,1.2,1.45,2,3,100])],observed=True).agg(n=('profit','size'),pred=('ev_c','mean'),real=('profit','mean'),se=('profit','sem')).mul([1,100,100,100]).round(2).to_string())
# oracle: CLV-EV of every earlier state
d=c[c.kickoff>='2025-01-01'].merge(te[['unique_id','side','P_c','odds']].rename(columns={'odds':'odds_close'}),on=['unique_id','side'])
d['clv_ev']=d.P_c*d.odds-1
d.to_parquet('data/dev_clv.parquet')
best=d.groupby(['grp','unique_id','side']).clv_ev.max().reset_index()
print('oracle best CLV-EV per market-outcome: share >0, >2%')
print(best.groupby('grp').clv_ev.agg(n='size',gt0=lambda v:(v>0).mean(),gt2=lambda v:(v>.02).mean(),gt5=lambda v:(v>.05).mean()).round(4))
pos=d[d.clv_ev>0]
print('states with clv_ev>0: realised ROI (all states, weighted per market-side)')
f=pos.sort_values('t_start').groupby(['unique_id','side']).head(1)
print(roi_table(f,['grp'])); print('mean clv_ev of those',f.groupby('grp').clv_ev.mean().round(4).to_dict())
f2=d[d.clv_ev>0.03].sort_values('t_start').groupby(['unique_id','side']).head(1)
print(roi_table(f2,['grp'])); print(roi_table(f2,[pd.cut(f2.odds,[1,1.2,1.45,2,3,100])]))
# does clv_ev rank realised profit? (all states)
d['b']=pd.cut(d.clv_ev,[-9,-.2,-.1,-.05,-.02,0,.02,.05,.1,9])
print(d.groupby('b',observed=True).apply(lambda g: pd.Series(dict(n=len(g),mk=g.unique_id.nunique(),clv=100*np.average(g.clv_ev,weights=g.w),real=100*np.average(g.profit,weights=g.w)))).round(2).to_string())
