from common import *
from sklearn.linear_model import LogisticRegression

class _R: pass
def logit_fit(y, X):
    X = X.assign(const=1.0); A = X.values.astype(float); yv = np.asarray(y, float); b = np.zeros(A.shape[1])
    for _ in range(30):
        p = 1/(1+np.exp(-A@b)); H = (A*(p*(1-p))[:,None]).T@A; b = b + np.linalg.solve(H, A.T@(yv-p))
    r=_R(); r.params=pd.Series(b,X.columns); r.tvalues=r.params/np.sqrt(np.diag(np.linalg.inv(H))); return r
L = pd.read_parquet(f"{W}/oof_joined.parquet")
L["drift"] = (L.c_l-L.own_l).where(L.c_close_gap_h<1)
d = L[L.drift.notna()]
print("drift: corr", np.corrcoef(d.drift, d.drift_hat)[0,1], "R2", 1-((d.drift-d.drift_hat)**2).mean()/d.drift.var(), "sd", d.drift.std())
d["q"]=pd.qcut(d.drift_hat, [0,.001,.01,.05,.25,.75,.95,.99,.999,1])
print(d.groupby("q",observed=True).agg(n=("drift","size"),hat=("drift_hat","mean"),real=("drift","mean"),med=("drift","median")).round(4))
for t in (0,1,2):
    x=d[(d.tier==t)&(d["rank"]==0)]; print("tier",t,"fav corr",round(np.corrcoef(x.drift,x.drift_hat)[0,1],3), "sd hat",round(x.drift_hat.std(),4),"sd drift",round(x.drift.std(),4))
# (b) closing efficiency wrt path: full dev long table closing rows
A = load_long(["unique_id","outcome","kick","t","tier","rank","own_l","own_rr","own_dopen","own_lastd","own_d24h","own_d3h","won","ttk_h","n_chg","own_o","age_h"])
C = A.loc[A.groupby(["unique_id","outcome"]).t.idxmax()]; C=C[C.ttk_h<1]
for t in (0,1,2,"all"):
    c = C if t=="all" else C[C.tier==t]
    X = pd.DataFrame({"l":c.own_l,"l2":c.own_l**2,"l3":c.own_l**3,"draw":(c.outcome==1)*1.0,"away":(c.outcome==2)*1.0,"rr":c.own_rr,"dopen":c.own_dopen,"d3h":c.own_d3h,"lastd":c.own_lastd})
    if t=="all":
        for k in (0,1): X[f"t{k}"]=(c.tier==k)*1.0
    r = logit_fit(c.won, X)
    print("tier",t,"n",len(c)); print(pd.DataFrame({"coef":r.params,"z":r.tvalues}).round(3).T.to_string())
