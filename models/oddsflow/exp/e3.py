from common import *
exec(open("/Users/boazbardavid/Desktop/Winner/models/oddsflow/exp/e2.py").read().split("class _R")[1].split("L = pd.read")[0].join(["class _R",""]))
A = load_long(["unique_id","outcome","kick","t","tier","rank","own_l","own_m","own_sl","own_rr","won","ttk_h","own_o"])
C = A.loc[A.groupby(["unique_id","outcome"]).t.idxmax()]; C=C[C.ttk_h<1]
for t in (0,1,2):
    c = C[C.tier==t]
    for base in ("own_m","own_l"):
        X = pd.DataFrame({"l":c[base],"l2":c[base]**2,"l3":c[base]**3,"draw":(c.outcome==1)*1.0,"away":(c.outcome==2)*1.0,"sl":c.own_sl,"sl_fav":c.own_sl*(c["rank"]==0)})
        r = logit_fit(c.won, X); p=1/(1+np.exp(-(X.assign(const=1.0).values@r.params.values)))
        print("tier",t,base,"ll",round(-np.mean(np.log(np.where(c.won==1,p,1-p))),5)); print(pd.DataFrame({"coef":r.params,"z":r.tvalues}).round(2).T.to_string())
F=C[(C["rank"]==0)&(C.own_o<=1.5)].copy(); F["pr"]=F.won*F.own_o-1
print(F.pivot_table(index="tier",columns=pd.cut(F.own_sl,[-1,-0.02,-0.01,0,0.01,0.02,1]),values="pr",aggfunc=["mean","size"]).round(3).to_string())
