from common import *
L = load_long(["unique_id","kick","kind","outcome","rank","tier","own_o","own_rr","own_l","won","c_o","c_l","c_close_gap_h","ttk_h","dov","t","big","own_dopen"])
print(len(L), L.unique_id.nunique())
L["pr"] = L.won*L.own_o-1
# closing rows: last point per market
C = L.sort_values("t").groupby(["unique_id","outcome"]).tail(1)
C = C[C.ttk_h<1]
print("closing rows", len(C)//3)
print("flat at close by tier"); print(C.groupby("tier").pr.agg(["size","mean"]).round(4))
C["ob"]=pd.cut(C.own_o,[1,1.1,1.2,1.3,1.45,1.6,1.8,2,2.5,3,4,6,10,100])
print(C.pivot_table(index="ob",columns="tier",values="pr",aggfunc=["mean","size"],observed=True).round(3))
# favourites: by odds grid and slack sign
F=C[(C["rank"]==0)&(C.own_o<=1.5)].copy()
F["sl"]=pd.cut(F.own_rr,[-1,-0.01,0.0,0.01,1])
print(F.pivot_table(index=["tier","sl"],columns=pd.cut(F.own_o,[1,1.07,1.12,1.17,1.22,1.32,1.5]),values="pr",aggfunc=["mean","size"],observed=True).round(3).to_string())
# opening rows
Op = L[L.kind==0].sort_values("t").groupby(["unique_id","outcome"]).head(1)
Op["pr"]=Op.won*Op.own_o-1
print("flat at open by tier"); print(Op.groupby("tier").pr.agg(["size","mean"]).round(4))
# does the path matter given close? outcome ~ close logit + (close-open)
import statsmodels.api as sm
