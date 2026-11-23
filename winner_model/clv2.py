import pandas as pd, numpy as np
from lib import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=pd.read_parquet('data/dev_clv.parquet')
d['good']=(d.clv_ev>0.03)
d['base_ev']=np.nan
print('share of states (weighted) with clv>3% by features')
def tab(by): 
    t=d.groupby(by,observed=True).apply(lambda g: pd.Series(dict(n=len(g),good=100*np.average(g.good,weights=g.w),clv=100*np.average(g.clv_ev,weights=g.w),real=100*np.average(g.profit,weights=g.w))))
    print(t.round(2).to_string())
tab('grp')
tab(pd.cut(d.hrs_to_ko,[0,1,3,6,12,24,48,96,1000]))
tab(pd.cut(d.n_chg,[-1,0,1,2,5,10,20,1000]))
tab(pd.cut(d.odds,[1,1.2,1.5,2,3,5,100]))
tab('gap_before')
tab(pd.cut(d.tick_prev.fillna(0),[-100,-3,-1.5,-.5,.5,1.5,3,100]))
tab(pd.cut(d.rel_open,[-1,-.2,-.1,-.03,.03,.1,.2,10]))
# duration of good states: how long do they stay available?
g=d[d.good].copy(); g['dur_h']=(g.t_end-g.t_start).dt.total_seconds()/3600
print(g.dur_h.describe(percentiles=[.1,.25,.5,.75,.9]).round(2))
print('time from good state to kickoff', g.hrs_to_ko.describe(percentiles=[.1,.25,.5,.75,.9]).round(1).to_dict())
# is the good state the opening state?
print('share of good states that are opening segs', (g.seg_no==1).mean(), ' vs all', (d.seg_no==1).mean())
