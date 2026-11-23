import pandas as pd, numpy as np
from lib import *
from wf import bets, summarize
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
p=pd.read_parquet('data/wf_preds.parquet')
sig=lambda z:1/(1+np.exp(-z))
p['P_s']=sig(p.z_s); p['P_g']=sig(p.z_g)
p['ev_s']=p.P_s*p.odds-1; p['ev_g']=p.P_g*p.odds-1; p['ev_m']=p.q_raw*p.odds-1
def ll(P,y,w): P=np.clip(P,1e-6,1-1e-6); return -(w*(y*np.log(P)+(1-y)*np.log(1-P))).sum()/w.sum()
print('binary logloss (weighted per market-outcome): market q_raw / struct / gbm')
for g,x in p.groupby('grp'):
    x=x.dropna(subset=['z_s'])
    if len(x): print(g, len(x), round(ll(x.q_raw,x.won,x.w),5), round(ll(x.P_s,x.won,x.w),5), round(ll(x.P_g,x.won,x.w),5))
rows=[]
for ev in ['ev_s','ev_g']:
    for thr in [-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.05]:
        b=bets(p,p[ev],thr); r=summarize(b,f'{ev}>{thr}'); rows.append(r)
print(pd.DataFrame(rows).to_string())
# EV-decile calibration over all candidate rows (weighted)
for ev in ['ev_s','ev_g']:
    x=p.dropna(subset=[ev]).copy(); x['b']=pd.cut(x[ev],[-9,-.3,-.2,-.15,-.1,-.07,-.05,-.03,-.02,-.01,0,.01,.02,.05,9])
    t=x.groupby('b',observed=True).apply(lambda d: pd.Series(dict(n=len(d),mk=d.unique_id.nunique(),pred=100*np.average(d[ev],weights=d.w),real=100*np.average(d.profit,weights=d.w))))
    print(ev); print(t.round(2).to_string())
for ev in ['ev_s','ev_g']:
    b=bets(p,p[ev],-0.01)
    print(ev,'>-0.01 by grp'); print(roi_table(b,['grp']).to_string())
    print(roi_table(b,[pd.cut(b.odds,[1,1.1,1.2,1.3,1.5,2,100])]).to_string())
    print(roi_table(b,[b.kickoff.dt.to_period('Q')]).to_string())
