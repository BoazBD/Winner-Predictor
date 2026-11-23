"""Walk-forward harness on DEV (kickoff < 2026-03-01). FINAL is never loaded here."""
import pandas as pd, numpy as np, sys, warnings
from lib import *
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
warnings.filterwarnings('ignore')
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)

def load_dev():
    c=pd.read_parquet('data/cand.parquet')
    c=c[~c.final].reset_index(drop=True)
    c['w']=1.0/c.groupby(['unique_id','side']).odds.transform('size')
    c['grp']=c.mkt+'_'+c.tier.astype(str)
    c['lhk']=np.log1p(c.hrs_to_ko.clip(0))
    c['up']=(c.tick_prev>0).astype(float); c['dn']=(c.tick_prev<0).astype(float)
    c['lq2']=c.lq**2; c['fav']=c.is_fav.astype(float)
    c['lodds']=np.log(c.odds)
    return c

FOLDS=[('2025-01-01','2025-04-01'),('2025-04-01','2025-07-01'),('2025-07-01','2025-10-01'),('2025-10-01','2026-01-01'),('2026-01-01','2026-03-01')]
SF=['lq','lq2','fav','rgain','lhk','up','dn','rel_open']

def fit_struct(tr, te, feats=SF):
    """per-group logistic; returns logit predictions for te"""
    out=pd.Series(np.nan,index=te.index)
    for g,trg in tr.groupby('grp'):
        teg=te[te.grp==g]
        if len(teg)==0 or trg.unique_id.nunique()<300: continue
        mu=trg[feats].mean(); sd=trg[feats].std().replace(0,1)
        lr=LogisticRegression(C=1.0,max_iter=300).fit(((trg[feats]-mu)/sd).fillna(0),trg.won,sample_weight=trg.w)
        out.loc[teg.index]=lr.decision_function(((teg[feats]-mu)/sd).fillna(0))
    return out

GF=['lq','rgain','e_hat','dov','tier','odds','fav','fav_odds','hrs_to_ko','hrs_open','hrs_prev','n_chg','tick_prev','rel_open','pos_rng','side','mktc','big','line1']
def fit_gbm(tr, te, base_tr, base_te, feats=GF, seeds=3, params=None):
    p=dict(objective='binary',learning_rate=0.03,num_leaves=15,min_data_in_leaf=2000,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=8)
    if params: p.update(params)
    pred=np.zeros(len(te))
    ok=base_tr.notna().values
    for sd in range(seeds):
        p['seed']=sd
        ds=lgb.Dataset(tr.loc[ok,feats],tr.won[ok],weight=tr.w[ok],init_score=base_tr[ok])
        b=lgb.train(p,ds,num_boost_round=p.get('n_round',150))
        pred+=b.predict(te[feats],raw_score=True)/seeds
    return base_te+pred

def bets(te, ev, thr, col='ev', max_odds=None, per_match=False):
    x=te[ev>thr]
    if max_odds: x=x[x.odds<=max_odds]
    x=x.sort_values('t_start').groupby(['unique_id','side']).head(1)
    return x

def summarize(b, name=''):
    if len(b)==0: return dict(name=name,n=0)
    lo,hi=boot_ci(b.profit.values,b.result_id.values,n=1000)
    return dict(name=name,n=len(b),roi=round(100*b.profit.mean(),2),lo=round(100*lo,2),hi=round(100*hi,2),odds=round(b.odds.mean(),3),win=round(b.won.mean(),3))

if __name__=='__main__':
    c=load_dev()
    c['mktc']=c.mkt.astype('category').cat.codes; c['big']=c.big_league.astype(int)
    allp=[]
    for a,b_ in FOLDS:
        tr=c[c.kickoff<a]; te=c[(c.kickoff>=a)&(c.kickoff<b_)].copy()
        te['z_s']=fit_struct(tr,te)
        ztr=fit_struct(tr,tr)   # in-sample base for gbm init (low-df model, fine)
        te['z_g']=fit_gbm(tr,te,ztr,te.z_s)
        allp.append(te); print(a,len(tr),len(te),flush=True)
    p=pd.concat(allp)
    p.to_parquet('data/wf_preds.parquet')
