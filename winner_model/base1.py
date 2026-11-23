import pandas as pd, numpy as np
from lib import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
m=pd.read_parquet('data/markets_close.parquet') if False else None
m=load_markets(); s=load_segments()
last=s.sort_values(['unique_id','seg_no']).groupby('unique_id').tail(1).set_index('unique_id')
first=s[s.seg_no==1].set_index('unique_id')
m=m.set_index('unique_id').join(last[['t_end','r1','r2','r3']]).join(first[['t_start','r1','r2','r3']].add_suffix('_o'))
m['mins_close']=(m.kickoff-m.t_end).dt.total_seconds()/60
m=m.join(s.groupby('unique_id').agg(nseg=('seg_no','size'), gaps=('gap_before','sum')))
m.to_parquet('data/markets_close.parquet')
print(m.groupby('mkt').mins_close.describe(percentiles=[.1,.5,.9]).round(1).to_string())
d=m[(~m.final)&(m.mins_close<60)&(m.y>=0)]
x=to_long(d)
x['ov']=1/x.r1+1/x.r2+(1/x.r3).fillna(0)
print(x.groupby(['sport','mkt']).ov.median())
x['ob']=pd.cut(x.odds,[1,1.1,1.2,1.3,1.45,1.6,1.8,2.1,2.6,3.5,5,100])
print(roi_table(x,['sport','mkt','side']))
for (sp,mk),g in x.groupby(['sport','mkt']):
    if len(g)>3000: print(sp,mk); print(roi_table(g,['ob']).T.to_string())
g=x[(x.mkt=='1x2')]; print('1x2 big vs not'); print(roi_table(g,['big_league','ob']).unstack(0).to_string())
