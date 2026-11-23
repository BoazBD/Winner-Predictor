"""Candidate table: one row per (market price-state, outcome), decision time = segment start.
All features use only information available at t (segment start). Markets: soccer 1x2, hc3, tennis."""
import pandas as pd, numpy as np
from lib import *
m=load_markets().set_index('unique_id')
m=m[m.mkt.isin(['1x2','hc3','tennis'])]
s=load_segments()
s=s[s.unique_id.isin(m.index)].sort_values(['unique_id','seg_no']).reset_index(drop=True)
two=(s.unique_id.map(m.mkt)=='tennis').values
s['mkt']=s.unique_id.map(m.mkt).values
# 2-way: move r2 -> r3 so that columns are (first, middle, second)
s.loc[two,'r3']=s.loc[two,'r2']; s.loc[two,'r2']=np.nan
R=s[['r1','r2','r3']].values.astype(float)
inv=1/R; ov=np.nansum(inv,1)
TIERS={'1x2':np.array([1/0.9,1/0.85,1/0.8]),'hc3':np.array([1/0.9,1/0.85,1/0.8]),'tennis':np.array([1/0.9,1/0.825])}
tier_ov=np.zeros(len(s)); tier=np.zeros(len(s),int)
for k,T in TIERS.items():
    i=(s.mkt==k).values; a=np.abs(ov[i,None]-T[None]).argmin(1); tier[i]=a; tier_ov[i]=T[a]
dov=ov-tier_ov
sens=0.05/R**2                       # prob change per tick
v=sens**2; e_hat=dov[:,None]*v/np.nansum(v,1)[:,None]   # E[rounding error_i | sum]
p_lat=inv-e_hat                      # latent margin-inclusive prob
q=p_lat/tier_ov[:,None]              # proportional de-margin
g=s.groupby('unique_id')
first=g[['r1','r2','r3']].transform('first').values.astype(float)
prev=g[['r1','r2','r3']].shift(1).values.astype(float)
t_open=g.t_start.transform('first')
prev_start=g.t_start.shift(1)
s['hrs_to_ko']=(s.kickoff_seg-s.t_start).dt.total_seconds()/3600
s['hrs_open']=(s.t_start-t_open).dt.total_seconds()/3600
s['hrs_prev']=(s.t_start-prev_start).dt.total_seconds()/3600
s['n_chg']=s.seg_no-1
s['ov']=ov; s['dov']=dov; s['tier']=tier
# running min/max of each price so far (incl current)
cm=g[['r1','r2','r3']].cummax().values.astype(float); cn=g[['r1','r2','r3']].cummin().values.astype(float)
fav=np.nanargmin(R,1)
won=m.loc[s.unique_id,['won1','wonx','won2']].values.astype(float)
base=s[['unique_id','seg_no','t_start','t_end','i_start','mkt','tier','ov','dov','hrs_to_ko','hrs_open','hrs_prev','n_chg','gap_before']].copy()
base['fav_odds']=R[np.arange(len(s)),fav]
out=[]
for k in range(3):
    x=base.copy()
    x['side']=k; x['odds']=R[:,k]; x['is_fav']=(fav==k)
    x['e_hat']=e_hat[:,k]; x['p_lat']=p_lat[:,k]; x['q']=q[:,k]; x['q_raw']=inv[:,k]/ov
    x['odds_open']=first[:,k]; x['odds_prev']=prev[:,k]
    x['odds_max']=cm[:,k]; x['odds_min']=cn[:,k]
    x['won']=won[:,k]
    out.append(x[~np.isnan(R[:,k])])
c=pd.concat(out,ignore_index=True)
c['profit']=c.won*c.odds-1
mm=m[['result_id','league','kickoff','big_league','final','line1','sport']]
c=c.join(mm,on='unique_id')
c['rgain']=-c.odds*c.e_hat             # rounding gain as fraction of stake
c['lq']=np.log(c.q.clip(1e-4,1-1e-4)/(1-c.q.clip(1e-4,1-1e-4)))
c['tick_prev']=((c.odds-c.odds_prev)/0.05).round()
c['rel_open']=c.odds/c.odds_open-1
c['pos_rng']=((c.odds-c.odds_min)/(c.odds_max-c.odds_min)).fillna(0.5)
for col in c.select_dtypes('float64').columns: c[col]=c[col].astype('float32')
c.to_parquet('data/cand.parquet')
print(c.shape); print(c.groupby(['mkt','tier']).unique_id.nunique()); print(c.head().T)
