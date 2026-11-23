from common import *
cols = ["pt","outcome","unique_id","kick","t","kind","tier","rank","own_o","own_l","own_m","own_sl","won","ttk_h","c_o","c_l","c_close_gap_h","big","y","fav_o"]
L = load_long(cols).merge(pd.read_parquet(f"{W}/oof.parquet"), on=["pt","outcome"])
sig = lambda z: 1/(1+np.exp(-z))
L["p_mkt"] = sig(L.own_m)
MODELS = ["p_mkt","p_lvl","p_g_all","p_2s"]
for m in MODELS:
    L[m] = L[m]/L.groupby("pt")[m].transform("sum")
L["w"] = 1.0/L.groupby(["unique_id","outcome"]).pt.transform("size")
print("rows", len(L), "matches", L.unique_id.nunique())
# log-loss (multiclass: -log p of the winning outcome), weighted per market
won = L[L.won==1]
def ll(d, m): return float(np.average(-np.log(d[m]), weights=d.w))
print("LOG-LOSS all points (per-market weight)")
print(pd.DataFrame({g: {m: ll(d,m) for m in MODELS} for g,d in [("all",won)]+[(f"tier{t}",won[won.tier==t]) for t in (0,1,2)]+[("ttk>24h",won[won.ttk_h>24]),("ttk<2h",won[won.ttk_h<2])]}).round(4).T)
print(pd.DataFrame({f: {m: ll(d,m)-ll(d,"p_mkt") for m in MODELS[1:]} for f,d in won.groupby("fold")}).round(4).T)
# betting sims: first qualifying point per (market,outcome)
def bets(d, ev, thr):
    x = d[ev > thr]
    return x.sort_values("t").drop_duplicates(["unique_id","outcome"])
for m in MODELS[1:]:
    L["ev_"+m] = L[m]*L.own_o-1
    print("==", m, "EV quantiles", L["ev_"+m].quantile([.5,.9,.99,.999]).round(3).tolist())
    for thr in (-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02):
        b = bets(L, L["ev_"+m], thr)
        if len(b) < 30: continue
        clv = (b.own_o/b.c_o-1)[b.c_close_gap_h<1]
        cev = (b.p_close*b.own_o-1)[b.c_close_gap_h<1]
        print(roi_line(b, f"ev>{thr:+.2f}"), f"closeEV={100*cev.mean():+.2f}% clv={100*clv.mean():+.2f}% beat={100*(clv>0).mean():.0f}% worse={100*(clv<0).mean():.0f}% predEV={100*b['ev_'+m].mean():+.1f}%")
L.to_parquet(f"{W}/oof_joined.parquet")
