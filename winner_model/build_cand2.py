"""cand2: candidates at every time ANY market of the match changes (match timeline), with sibling state.
Row = (market, time t, outcome); market must be present in the scraper run at t."""
import pandas as pd, numpy as np
from lib import *
import lib

def build(out_path=None, add_now=False):
    m=load_markets().set_index('unique_id')
    m=m[m.mkt.isin(['1x2','hc3','tennis'])]
    s=load_segments()
    s=s[s.unique_id.isin(m.index)].sort_values(['unique_id','seg_no']).reset_index(drop=True)
    s['mkt']=s.unique_id.map(m.mkt).values; s['result_id']=s.unique_id.map(m.result_id).values
    two=(s.mkt=='tennis').values
    s.loc[two,'r3']=s.loc[two,'r2']; s.loc[two,'r2']=np.nan
    R=s[['r1','r2','r3']].values.astype(float); inv=1/R; ov=np.nansum(inv,1)
    TIERS={'1x2':np.array([1/0.9,1/0.85,1/0.8]),'hc3':np.array([1/0.9,1/0.85,1/0.8]),'tennis':np.array([1/0.9,1/0.825])}
    tier_ov=np.zeros(len(s)); tier=np.zeros(len(s),int)
    for k,T in TIERS.items():
        i=(s.mkt==k).values; a=np.abs(ov[i,None]-T[None]).argmin(1); tier[i]=a; tier_ov[i]=T[a]
    dov=ov-tier_ov; v=(0.05/R**2)**2; e_hat=dov[:,None]*v/np.nansum(v,1)[:,None]
    q=(inv-e_hat)/tier_ov[:,None]
    for k in range(3): s[f'q{k}']=q[:,k]; s[f'e{k}']=e_hat[:,k]
    s['ov']=ov; s['dov']=dov; s['tier']=tier
    g=s.groupby('unique_id')
    for k,r in enumerate(['r1','r2','r3']):
        s[f'open{k}']=g[r].transform('first'); s[f'prev{k}']=g[r].shift(1)
        s[f'max{k}']=g[r].cummax(); s[f'min{k}']=g[r].cummin()
    s['t_open']=g.t_start.transform('first'); s['n_chg']=s.seg_no-1
    s['seg_start']=s.t_start; s['seg_i']=s.i_start
    runs=load_runs()
    # match timeline
    tl=s[['result_id','i_start']].drop_duplicates()
    if add_now:
        last_i=int(runs.run_idx.max()); live=s.loc[s.i_end==last_i,['result_id']].drop_duplicates().assign(i_start=last_i)
        tl=pd.concat([tl,live]).drop_duplicates()
    mk=s[['result_id','unique_id']].drop_duplicates()
    tl=tl.merge(mk,on='result_id').rename(columns={'i_start':'i'}).sort_values('i')
    st=pd.merge_asof(tl,s.sort_values('i_start'),left_on='i',right_on='i_start',by='unique_id',suffixes=('','_s'))
    st=st[st.i_start.notna()&(st.i<=st.i_end)].copy()
    st['t']=runs.run_ts.values[st.i.values]
    print('market-time states',len(st))
    # sibling: 1x2 <-> primary hc3 (smallest |line|, then earliest first_seen)
    hc=m[m.mkt=='hc3'].copy(); hc['al']=hc.line1.abs()
    prim=hc.sort_values(['al','first_seen']).groupby('result_id').head(1)
    prim_id=set(prim.index)
    x1=m[m.mkt=='1x2'].reset_index().set_index('result_id').unique_id
    sibcols=['r1','r2','r3','q0','q1','q2','dov','tier','seg_start','n_chg']
    S=st[['result_id','unique_id','i']+sibcols].copy(); S['line1']=S.unique_id.map(m.line1)
    Sh=S[S.unique_id.isin(prim_id)].drop(columns='unique_id').add_prefix('sib_').rename(columns={'sib_result_id':'result_id','sib_i':'i'})
    Sx=S[S.unique_id.isin(set(x1.values))].drop(columns='unique_id').add_prefix('sib_').rename(columns={'sib_result_id':'result_id','sib_i':'i'})
    is1=(st.mkt=='1x2').values; ish=st.unique_id.isin(prim_id).values
    a=st[is1].merge(Sh,on=['result_id','i'],how='left')
    b=st[ish].merge(Sx,on=['result_id','i'],how='left')
    c_=st[~is1&~ish]
    st=pd.concat([a,b,c_],ignore_index=True)
    st['hrs_to_ko']=(st.kickoff_seg-st.t).dt.total_seconds()/3600
    st['hrs_open']=(st.t-st.t_open).dt.total_seconds()/3600
    st['own_age']=(st.t-st.seg_start).dt.total_seconds()/3600
    st['sib_age']=(st.t-st.sib_seg_start).dt.total_seconds()/3600
    st['line1']=st.unique_id.map(m.line1).values
    won=m.loc[st.unique_id,['won1','wonx','won2']].values.astype(float)
    R=st[['r1','r2','r3']].values.astype(float); fav=np.nanargmin(R,1)
    lg=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
    base=st[['unique_id','result_id','mkt','t','i','seg_no','tier','ov','dov','hrs_to_ko','hrs_open','own_age','sib_age','n_chg','sib_n_chg','sib_tier','sib_dov','gap_before']].copy()
    base['fav_odds']=R[np.arange(len(st)),fav]
    out=[]
    for k in range(3):
        o=2-k if k!=1 else 2   # opposite side index
        x=base.copy(); x['side']=k; x['odds']=R[:,k]; x['is_fav']=(fav==k)
        x['e_hat']=st[f'e{k}'].values; x['q']=st[f'q{k}'].values; x['q_raw']=(1/R[:,k])/st.ov.values
        for nm in ['open','prev','max','min']: x[f'odds_{nm}']=st[f'{nm}{k}'].values
        own,opp=(k,2-k) if k!=1 else (0,2)
        x['own_lq']=lg(st[f'q{own}'].values); x['mid_lq']=lg(st['q1'].values); x['opp_lq']=lg(st[f'q{opp}'].values)
        x['sib_own_lq']=lg(st[f'sib_q{own}'].values); x['sib_mid_lq']=lg(st['sib_q1'].values); x['sib_opp_lq']=lg(st[f'sib_q{opp}'].values)
        sgn=1 if own==0 else -1
        x['line_own']=sgn*st.line1.values; x['sib_line_own']=sgn*st.sib_line1.values
        x['sib_odds_own']=st[f'sib_r{own+1}'].values
        x['won']=won[:,k]
        out.append(x[~np.isnan(R[:,k])])
    c=pd.concat(out,ignore_index=True)
    c['profit']=c.won*c.odds-1
    c=c.join(m[['league','kickoff','big_league','final','sport']],on='unique_id')
    c['rgain']=-c.odds*c.e_hat
    c['lq']=lg(c.q.values)
    c['tick_prev']=((c.odds-c.odds_prev)/0.05).round()
    c['rel_open']=c.odds/c.odds_open-1
    c['pos_rng']=((c.odds-c.odds_min)/(c.odds_max-c.odds_min)).fillna(0.5)
    c['t_start']=c.t
    for col in c.select_dtypes('float64').columns: c[col]=c[col].astype('float32')
    if out_path: c.to_parquet(out_path)
    print(c.shape, c.sib_own_lq.notna().mean()); print(c.groupby('mkt').size())
    return c

if __name__=='__main__':
    build(f'{lib.D}/cand2.parquet')
