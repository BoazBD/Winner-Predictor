"""Compare drift-model variants on 3 walk-forward folds. Metric: R2 and first-trigger policy results."""
from common import *
import lightgbm as lgb, time, warnings, sys
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
exec(open(f"{W}/../exp/wf.py").read().split("def calib_X")[1].split("out = []")[0].join(["def calib_X",""]))
FOLDS = ["2025-01-01","2025-07-01","2026-01-01","2026-03-01"]; TRAIN_TO = {"2025-01-01":"2025-07-01","2025-07-01":"2026-01-01","2026-01-01":"2026-03-01"}
L = load_long().sort_values(["pt","outcome"]).reset_index(drop=True)
L["w"] = 1.0/L.groupby(["unique_id","outcome"]).pt.transform("size")
L["drift"] = (L.c_l - L.own_l).where(L.c_close_gap_h < 1.0)
B = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=800, feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=20, verbose=-1, num_threads=8, max_bin=127, objective="huber", alpha=0.3)
VARS = {
 "v0_base": dict(),
 "v1_l2": dict(p=dict(objective="regression")),
 "v2_big": dict(p=dict(num_leaves=63, min_data_in_leaf=300), rounds=1000),
 "v3_small": dict(p=dict(num_leaves=15, min_data_in_leaf=2000), rounds=300),
 "v4_favonly": dict(sub=lambda d: d.own_o <= 2.0),
 "v5_roww": dict(roww=True),
 "v6_nopath": dict(feats=LEVEL_FEATS+["ttk_h","open_h","kick_hod","kick_dow","snap_hod"]),
 "v8_wtier": dict(p=dict(objective="regression"), tw={0:5.0,1:3.0,2:1.0}),
 "v9_t01": dict(p=dict(objective="regression"), sub=lambda d: d.tier <= 1),
 "v10_bag3": dict(p=dict(objective="regression", min_data_in_leaf=500), rounds=600, seeds=3),
 "v11_t01_bag": dict(p=dict(objective="regression", min_data_in_leaf=300), sub=lambda d: d.tier <= 1, seeds=3),
 "v7_notime": dict(feats=[f for f in FEATS if f not in ("kick_hod","kick_dow","snap_hod")]),
}
only = sys.argv[1:] 
res = {}
for a, b in zip(FOLDS[:-1], FOLDS[1:]):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    tr = L[(L.kick < a) & L.drift.notna()]; te = L[(L.kick >= a) & (L.kick < b)].copy()
    cl = L[L.kick < a]; cl = cl.loc[cl.groupby(["unique_id","outcome"]).t.idxmax()]
    lr = LogisticRegression(C=10, max_iter=1000).fit(calib_X(cl.own_l.values, cl), cl.won)
    te["p_close"] = lr.predict_proba(calib_X(te.c_l.fillna(0).values, te))[:,1]
    for name, v in VARS.items():
        if only and name not in only: continue
        t0 = time.time()
        t = tr[v["sub"](tr)] if "sub" in v else tr
        t = t.sample(min(len(t), 2_500_000), random_state=0)
        feats = v.get("feats", FEATS)
        wt = t.w * (t.tier.map(v["tw"]) if "tw" in v else 1.0)
        ms = [lgb.train({**B, **v.get("p", {}), "seed": sd}, lgb.Dataset(t[feats], t.drift, weight=None if v.get("roww") else wt), v.get("rounds", 400)) for sd in range(v.get("seeds", 1))]
        x = te[["pt","outcome","unique_id","kick","t","tier","rank","own_o","won","drift","c_o","c_close_gap_h","p_close","ttk_h"]].copy()
        x["drift_hat"] = np.mean([m.predict(te[feats]) for m in ms], 0)
        p = lr.predict_proba(calib_X((te.own_l + x.drift_hat).values, te))[:,1]
        x["p"] = p/pd.Series(p).groupby(te.pt.values).transform("sum").values
        x["ev"] = x.p*x.own_o-1
        res.setdefault(name, []).append(x); print(a.date(), name, round(time.time()-t0), flush=True)
for name, xs in res.items():
    x = pd.concat(xs); d = x[x.drift.notna()]; f = d[d.own_o <= 1.6]
    print(f"== {name}: R2 all {1-((d.drift-d.drift_hat)**2).mean()/d.drift.var():.4f}  R2 fav {1-((f.drift-f.drift_hat)**2).mean()/f.drift.var():.4f}")
    for thr in (-0.02, -0.01, 0.0, 0.01):
        bt = x[x.ev > thr].sort_values("t").drop_duplicates(["unique_id","outcome"])
        cev = (bt.p_close*bt.own_o-1)[bt.c_close_gap_h<1]
        print("  ", roi_line(bt, f"ev>{thr:+.2f}"), f"closeEV={100*cev.mean():+.2f}% predEV={100*bt.ev.mean():+.2f}% t0={100*(bt.tier==0).mean():.0f}%")
    x.to_parquet(f"{W}/dv_{name}.parquet")
