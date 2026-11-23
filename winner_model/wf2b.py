import pandas as pd, numpy as np, warnings
from lib import *
from wf import fit_struct, bets, summarize
from wf2 import *
warnings.filterwarnings('ignore')
c=pd.read_parquet('data/cand2.parquet'); c=c[~c.final].reset_index(drop=True); c=prep(c)
v1=pd.read_parquet('data/cand.parquet'); v1=v1[~v1.final]; cl=closes(v1); del v1
c=add_resid(c,'2025-01-01')
a='2025-07-01'
z=fit_struct(cl[cl.kickoff<a],cl,CF)
pc=cl.loc[z.index,['unique_id','side']].assign(P_c=1/(1+np.exp(-z.values))).dropna()
d=c.merge(pc,on=['unique_id','side'],how='inner'); d['clv']=d.P_c*d.odds-1
# current-state calibrated EV (same calibrator applied to current state) -> residual target = expected drift value
d['z_now']=fit_struct(cl[cl.kickoff<a],d,CF); d['ev_now']=d.odds/(1+np.exp(-d.z_now))-1
d['drift']=d.clv-d.ev_now
print(d[["clv","z_now","ev_now","drift"]].isna().mean(), d.kickoff.dtype, len(d)); d=d.dropna(subset=["drift"]); print(len(d), (d.kickoff<a).sum())
tr=d[d.kickoff<a]; te=d[d.kickoff>=a].copy()
trs=tr.sample(min(len(tr),2000000),random_state=0)
te['pd'],imp=fit_reg(trs,te,GF2,'drift',seeds=2,params=dict(objective='regression',min_data_in_leaf=500),rounds=500)
print(imp.head(15).round(3).to_dict())
te['pred']=te.ev_now+te.pd
te.to_parquet('data/wf2b_preds.parquet')
def show(col,bins):
    te['b']=pd.cut(te[col],bins)
    print(te.groupby('b',observed=True).apply(lambda g: pd.Series(dict(n=len(g),mk=g.unique_id.nunique(),pred=100*np.average(g[col],weights=g.w),drift=100*np.average(g.drift,weights=g.w),clv=100*np.average(g.clv,weights=g.w),real=100*np.average(g.profit,weights=g.w)))).round(2).to_string())
print('by predicted drift'); show('pd',[-9,-.05,-.02,-.01,-.005,0,.005,.01,.02,.03,.05,9])
print('by predicted total EV'); show('pred',[-9,-.2,-.1,-.07,-.05,-.03,-.02,-.01,0,.01,.02,.05,9])
rows=[]
for thr in [-0.05,-0.03,-0.02,-0.01,0,0.01,0.02]:
    b=bets(te,te.pred,thr); r=summarize(b,f'pred>{thr}'); r['clv']=round(100*b.clv.mean(),2) if len(b) else None; rows.append(r)
print(pd.DataFrame(rows).to_string())
for thr in [0.01,0.02,0.03]:
    b=bets(te,te.pd,thr); r=summarize(b,f'drift>{thr}'); r['clv']=round(100*b.clv.mean(),2); r['drift']=round(100*b.drift.mean(),2); print(r)
    print(roi_table(b,['grp']).T.to_string())
