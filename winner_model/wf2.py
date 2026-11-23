"""Walk-forward CLV-distillation model on DEV. Target = calibrated closing prob * current odds - 1."""
import pandas as pd, numpy as np, warnings, sys
from lib import *
from wf import fit_struct, FOLDS, bets, summarize
import lightgbm as lgb
warnings.filterwarnings('ignore')
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
CF=['lq','lq2','fav','rgain']
GF2=['lq','rgain','e_hat','dov','tier','odds','fav','fav_odds','hrs_to_ko','hrs_open','own_age','sib_age','n_chg','sib_n_chg','tick_prev','rel_open','pos_rng','side','mktc','big',
     'own_lq','mid_lq','opp_lq','sib_own_lq','sib_mid_lq','sib_opp_lq','line_own','sib_line_own','sib_tier','sib_dov','res_own','res_opp','res_mid','sib_odds_own','gapb']

def prep(c):
    c['tier']=c.tier.astype(int)
    c['w']=1.0/c.groupby(['unique_id','side']).odds.transform('size')
    c['grp']=c.mkt+'_'+c.tier.astype(str)
    c['lq2']=c.lq**2; c['fav']=c.is_fav.astype(float)
    c['mktc']=c.mkt.map({'1x2':0,'hc3':1,'tennis':2}); c['big']=c.big_league.astype(int); c['gapb']=c.gap_before.astype(float)
    return c

def closes(v1):
    last=v1.groupby(['unique_id','side']).seg_no.transform('max')
    cl=v1[(v1.seg_no==last)&((v1.kickoff-v1.t_end)<pd.Timedelta('60min'))].copy()
    cl['w']=1.0; cl['grp']=cl.mkt+'_'+cl.tier.astype(str); cl['lq2']=cl.lq**2; cl['fav']=cl.is_fav.astype(float)
    return cl

def add_resid(c, cutoff):
    RF=['own_lq','mid_lq','opp_lq','sib_line_own','line_own','tier','sib_tier','mktc']
    has=c.sib_own_lq.notna()
    tr=c[has&(c.kickoff<cutoff)]
    tr=tr.sample(min(len(tr),600000),random_state=0)
    for tgt,nm in [('sib_own_lq','res_own'),('sib_opp_lq','res_opp'),('sib_mid_lq','res_mid')]:
        mdl=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=63,min_child_samples=200,verbose=-1,n_jobs=8).fit(tr[RF],tr[tgt])
        c[nm]=np.nan; c.loc[has,nm]=c.loc[has,tgt]-mdl.predict(c.loc[has,RF])
    return c

def fit_reg(tr, te, feats, target, seeds=3, params=None, rounds=400):
    p=dict(objective='huber',alpha=0.1,learning_rate=0.03,num_leaves=31,min_data_in_leaf=1000,feature_fraction=0.7,bagging_fraction=0.5,bagging_freq=1,lambda_l2=10,verbose=-1,num_threads=8)
    if params: p.update(params)
    pred=np.zeros(len(te)); imp=np.zeros(len(feats))
    for sd in range(seeds):
        p['seed']=sd
        b=lgb.train(p,lgb.Dataset(tr[feats],tr[target],weight=tr.w),num_boost_round=rounds)
        pred+=b.predict(te[feats])/seeds; imp+=b.feature_importance('gain')
    return pred, pd.Series(imp/imp.sum(),index=feats).sort_values(ascending=False)

if __name__=='__main__':
    c=pd.read_parquet('data/cand2.parquet'); c=c[~c.final].reset_index(drop=True); c=prep(c)
    v1=pd.read_parquet('data/cand.parquet'); v1=v1[~v1.final]; cl=closes(v1); del v1
    c=add_resid(c,'2025-01-01')
    outp=[]
    for a,b_ in FOLDS:
        ctr=cl[cl.kickoff<a]
        z=fit_struct(ctr,cl[cl.kickoff<b_],CF)
        pc=cl.loc[z.index,['unique_id','side']].assign(P_c=1/(1+np.exp(-z.values))).dropna()
        d=c[c.kickoff<b_].merge(pc,on=['unique_id','side'],how='inner')
        d['clv']=d.P_c*d.odds-1
        tr=d[d.kickoff<a]; te=d[d.kickoff>=a].copy()
        te['pred'],imp=fit_reg(tr,te,GF2,'clv')
        outp.append(te); print(a,len(tr),len(te),flush=True); print(imp.head(12).round(3).to_dict(),flush=True)
    p=pd.concat(outp); p.to_parquet('data/wf2_preds.parquet')
    p['b']=pd.cut(p.pred,[-9,-.2,-.1,-.07,-.05,-.03,-.02,-.01,0,.01,.02,.05,9])
    print(p.groupby('b',observed=True).apply(lambda g: pd.Series(dict(n=len(g),mk=g.unique_id.nunique(),pred=100*np.average(g.pred,weights=g.w),clv=100*np.average(g.clv,weights=g.w),real=100*np.average(g.profit,weights=g.w)))).round(2).to_string())
    rows=[]
    for thr in [-0.05,-0.03,-0.02,-0.01,0,0.01,0.02,0.03]:
        b=bets(p,p.pred,thr); r=summarize(b,f'pred>{thr}'); r['clv']=round(100*b.clv.mean(),2) if len(b) else None; rows.append(r)
    print(pd.DataFrame(rows).to_string())
