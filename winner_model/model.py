"""Final pipeline.
  level  : pooled logistic calibration of odds-ratio de-margined, rounding-corrected probabilities (fit on closing prices vs results)
  drift  : LightGBM regression of (calibrated closing value - calibrated current value) on state features
  EV     = level EV at current price + predicted drift
"""
import pandas as pd, numpy as np, warnings, pickle
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lib import *
warnings.filterwarnings('ignore')
TIER_OV={'1x2':np.array([1/0.9,1/0.85,1/0.8]),'hc3':np.array([1/0.9,1/0.85,1/0.8]),'tennis':np.array([1/0.9,1/0.825,1/0.825])}
sig=lambda z:1/(1+np.exp(-z)); lg=lambda p: np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))

def add_or(c):
    """odds-ratio de-margined prob of own side from latent margin-inclusive probs"""
    tov=np.select([c.mkt=='tennis'],[np.array([1/0.9,1/0.825,1/0.825])[c.tier.values]],default=np.array([1/0.9,1/0.85,1/0.8])[c.tier.values])
    side=c.side.values
    P=np.stack([sig(c.own_lq.values)*tov, sig(c.mid_lq.values)*tov, sig(c.opp_lq.values)*tov],1)
    P=np.clip(P,1e-4,0.995)
    lo=np.full(len(c),1.0); hi=np.full(len(c),20.0)
    for _ in range(50):
        mid=(lo+hi)/2; v=np.nansum(P/(mid[:,None]+P-mid[:,None]*P),1); lo=np.where(v>1,mid,lo); hi=np.where(v>1,hi,mid)
    cc=(lo+hi)/2; F=P/(cc[:,None]+P-cc[:,None]*P)
    own=np.where(side==1,F[:,1],F[:,0])
    c['p_or']=own; c['lor']=lg(own); c['lor2']=c.lor**2; c['lor3']=c.lor**3
    for t in [0,1]:
        c[f't{t}']=(c.tier==t).astype(float); c[f't{t}_lor']=c[f't{t}']*c.lor
    c['draw']=(side==1).astype(float); c['favf']=c.is_fav.astype(float)
    return c
LF=['lor','lor2','lor3','t0','t1','t0_lor','t1_lor','draw','favf']

def prep(c):
    c['tier']=c.tier.astype(int)
    c['w']=1.0/c.groupby(['unique_id','side']).odds.transform('size')
    c['grp']=c.mkt+'_'+c.tier.astype(str)
    c['fav']=c.is_fav.astype(float)
    c['mktc']=c.mkt.map({'1x2':0,'hc3':1,'tennis':2}); c['big']=c.big_league.astype(int); c['gapb']=c.gap_before.astype(float)
    return add_or(c)

def fit_level(cl):
    mods={}
    for k,g in cl.groupby('mkt'):
        mu=g[LF].mean(); sd=g[LF].std().replace(0,1)
        mods[k]=(mu,sd,LogisticRegression(C=10,max_iter=500).fit((g[LF]-mu)/sd,g.won))
    return mods
def pred_level(mods,x):
    z=pd.Series(np.nan,index=x.index)
    for k,g in x.groupby('mkt'):
        mu,sd,lr=mods[k]; z.loc[g.index]=lr.decision_function((g[LF]-mu)/sd)
    return sig(z)

DF=['lor','rgain','e_hat','dov','tier','odds','fav','fav_odds','hrs_to_ko','hrs_open','own_age','sib_age','n_chg','sib_n_chg','tick_prev','rel_open','pos_rng','side','mktc','big',
    'own_lq','mid_lq','opp_lq','sib_own_lq','sib_mid_lq','sib_opp_lq','line_own','sib_line_own','sib_tier','sib_dov','sib_odds_own','gapb']
def fit_drift(tr, seeds=3, rounds=500, nmax=2500000):
    trs=tr.sample(min(len(tr),nmax),random_state=0)
    p=dict(objective='regression',learning_rate=0.03,num_leaves=31,min_data_in_leaf=500,feature_fraction=0.7,bagging_fraction=0.5,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=8)
    return [lgb.train({**p,'seed':s},lgb.Dataset(trs[DF],trs.drift,weight=trs.w),num_boost_round=rounds) for s in range(seeds)]
def pred_drift(bs,x): return np.mean([b.predict(x[DF]) for b in bs],0)

def closing_rows(c, close_ok):
    last=c.groupby(['unique_id','side']).i.transform('max')
    return c[(c.i==last)&c.unique_id.isin(close_ok)]

def run_split(c, close_ok, a, b):
    """train on kickoff<a, predict a<=kickoff<b. returns test frame with ev_now, drift_pred, ev, clv"""
    d=c[c.kickoff<b]
    cl=closing_rows(d,close_ok)
    mods=fit_level(cl[cl.kickoff<a])
    pc=cl[['unique_id','side']].assign(P_c=pred_level(mods,cl).values)
    d=d.merge(pc,on=['unique_id','side'],how='left')
    d['P_now']=pred_level(mods,d).values
    d['ev_now']=d.P_now*d.odds-1; d['clv']=d.P_c*d.odds-1; d['drift']=d.clv-d.ev_now
    tr=d[(d.kickoff<a)&d.drift.notna()]
    bs=fit_drift(tr)
    te=d[d.kickoff>=a].copy()
    te['drift_pred']=pred_drift(bs,te); te['ev']=te.ev_now+te.drift_pred
    return te,(mods,bs)

def first_bets(te, col, thr):
    x=te[te[col]>thr]
    return x.sort_values('t').groupby(['unique_id','side']).head(1)
