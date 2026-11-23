import pandas as pd, numpy as np
from lib import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
m=pd.read_parquet('data/markets_close.parquet')
d=m[(~m.final)&(m.mins_close<60)&(m.mkt=='1x2')].copy()
d['ov']=1/d.r1+1/d.r2+1/d.r3
d['ovb']=pd.cut(d.ov,[0,1.06,1.09,1.12,1.15,1.18,1.21,1.24,1.27,2])
print(d.groupby(['ovb']).size())
print(d.groupby([d.kickoff.dt.to_period('Q'),'ovb'],observed=True).size().unstack().fillna(0).astype(int).to_string())
x=to_long(d); x['ob']=pd.cut(x.odds,[1,1.2,1.45,1.8,2.6,5,100])
print(roi_table(x,['ovb','ob'])['roi%'].unstack().to_string())
print(roi_table(x,['ovb','ob'])['size'].unstack().to_string())
print(roi_table(x,['ovb','side']).unstack().to_string())
lg=d.groupby('league').agg(n=('ov','size'),ov=('ov','median'),ovmin=('ov',lambda v:v.quantile(.05))).sort_values('ov')
print(lg[lg.n>=100].head(40).to_string())
