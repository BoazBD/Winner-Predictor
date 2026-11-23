import pandas as pd, numpy as np
from lib import *
m=load_markets(); m=m[(m.sport=='Soccer')&(~m.final)]
s=load_segments(['unique_id','seg_no','i_start','i_end','t_start','gap_before'])
s=s.merge(m[['unique_id','result_id','mkt','line1']],on='unique_id')
# change events = seg starts with seg_no>1 and no gap
ch=s[(s.seg_no>1)]
a=ch[ch.mkt=='1x2'][['result_id','i_start']].drop_duplicates()
h=ch[ch.mkt=='hc3'][['result_id','i_start']].drop_duplicates()
# presence intervals of hc3 per match: need hc3 present at the time
hp=s[s.mkt=='hc3'].groupby('result_id').agg(lo=('i_start','min'),hi=('i_end','max'))
ap=s[s.mkt=='1x2'].groupby('result_id').agg(lo=('i_start','min'),hi=('i_end','max'))
a=a.join(hp,on='result_id',how='inner'); a=a[(a.i_start>a.lo)&(a.i_start<=a.hi)]
hs=set(zip(h.result_id,h.i_start))
for k in [0,1,2,-1,-2]:
    a[f'k{k}']=[(r,i+k) in hs for r,i in zip(a.result_id,a.i_start)]
print('1x2 change -> hc3 change at lag k (runs):', a[[c for c in a if c.startswith('k')]].mean().round(3).to_dict(), len(a))
a['any']=a[['k0','k1','k-1']].any(axis=1); print('within +-1:',a['any'].mean())
h=h.join(ap,on='result_id',how='inner'); h=h[(h.i_start>h.lo)&(h.i_start<=h.hi)]
as_=set(zip(a.result_id,a.i_start))
for k in [0,1,2,-1,-2]:
    h[f'k{k}']=[(r,i+k) in as_ for r,i in zip(h.result_id,h.i_start)]
print('hc3 change -> 1x2 change at lag k:', h[[c for c in h if c.startswith('k')]].mean().round(3).to_dict(), len(h))
print(m.groupby('result_id').mkt.agg(lambda v:(v=='hc3').sum()).value_counts().head(6))
print(m[m.mkt=='hc3'].line1.value_counts().head(10))
