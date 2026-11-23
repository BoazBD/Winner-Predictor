import pandas as pd, numpy as np, sys
from model import *
from wf import summarize
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
p=pd.read_parquet(sys.argv[1] if len(sys.argv)>1 else 'data/dev_final_preds.parquet')
def cal(col,bins=[-9,-.2,-.1,-.07,-.05,-.04,-.03,-.02,-.01,0,.01,.02,.05,9]):
    p['b']=pd.cut(p[col],bins)
    print(p.groupby('b',observed=True).apply(lambda g: pd.Series(dict(n=len(g),mk=g.unique_id.nunique(),pred=100*np.average(g[col],weights=g.w),clv=100*np.nanmean(g.clv),real=100*np.average(g.profit,weights=g.w)))).round(2).to_string())
print('== level only (ev_now)'); cal('ev_now'); print('== level+drift (ev)'); cal('ev')
rows=[]
for col in ['ev_now','ev']:
    for thr in [-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01]:
        b=first_bets(p,col,thr); r=summarize(b,f'{col}>{thr}'); r['clv']=round(100*b.clv.mean(),2) if len(b) else None; r['matches']=b.result_id.nunique(); rows.append(r)
print(pd.DataFrame(rows).to_string())
for col,thr in [('ev',-0.03),('ev',-0.02),('ev_now',-0.04)]:
    b=first_bets(p,col,thr); print(col,thr)
    print(roi_table(b,['grp']).T.to_string())
    print(roi_table(b,[pd.cut(b.odds,[1,1.1,1.2,1.3,1.5,2,100])]).T.to_string())
    print(roi_table(b,[b.kickoff.dt.to_period('Q')]).T.to_string())
    print('hrs_to_ko quantiles',b.hrs_to_ko.quantile([.1,.5,.9]).round(1).to_dict(),'mean rgain',round(b.rgain.mean(),4))
