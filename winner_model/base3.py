import pandas as pd, numpy as np
from lib import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
m=load_markets().set_index('unique_id'); s=load_segments()
m=m[(m.mkt=='1x2')&(~m.final)]
s=s[s.unique_id.isin(m.index)].copy()
s['kick']=s.unique_id.map(m.kickoff)
s=s[s.kickoff_seg==s.kick]  # ignore segments under an older (rescheduled) kickoff
def state_at(h):
    t=s.kick-pd.Timedelta(hours=h if not isinstance(h,str) else 0)
    if h=='open': x=s.sort_values('t_start').groupby('unique_id').head(1)
    elif h=='close':
        x=s.sort_values('t_start').groupby('unique_id').tail(1); x=x[(x.kick-x.t_end)<pd.Timedelta('60min')]
    else: x=s[(s.t_start<=t)&(s.t_end>=t)]
    return x.set_index('unique_id')[['r1','r2','r3']]
TIER=np.array([1/0.9,1/0.85,1/0.8])
res=[]
for h in ['open',72,24,6,2,'close']:
    x=state_at(h).join(m[['mkt','won1','wonx','won2','league','big_league','kickoff']])
    x['ov']=1/x.r1+1/x.r2+1/x.r3
    x['tier']=np.abs(x.ov.values[:,None]-TIER[None]).argmin(1)
    # favourite-only
    r=x[['r1','r2','r3']].values; f=np.nanargmin(r,1)
    x['odds']=r[np.arange(len(x)),f]; x['won']=x[['won1','wonx','won2']].values[np.arange(len(x)),f].astype(float)
    x['profit']=x.won*x.odds-1
    x['dov']=x.ov-TIER[x.tier]
    x['h']=str(h); res.append(x)
x=pd.concat(res)
x['ob']=pd.cut(x.odds,[1,1.1,1.2,1.3,1.45,1.6,1.8,2.1,3])
x['rb']=pd.cut(x.dov,[-1,-0.012,-0.004,0.004,0.012,1])
print('fav ROI by horizon x odds bucket, per tier')
for t in [0,1,2]:
    g=x[x.tier==t]; rt=roi_table(g,['h','ob'])
    print('tier',t); print(rt['roi%'].unstack().to_string()); print(rt['size'].unstack().to_string())
print('fav ROI by rounding residual (dov) x odds bucket, all horizons pooled per tier (close only)')
for t in [0,1,2]:
    g=x[(x.tier==t)&(x.h=='close')]; rt=roi_table(g,['rb','ob'])
    print('tier',t); print(rt['roi%'].unstack().to_string()); print(rt['size'].unstack().to_string())
g=x[(x.h=='close')&(x.odds<=1.45)]; rt=roi_table(g,['rb','ob']); print('all tiers close'); print(rt.unstack().to_string())
