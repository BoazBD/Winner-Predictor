"""Walk-forward (by kickoff) out-of-fold predictions on DEV for several model families.
Saves work/oof.parquet with p_* columns per long row (rows with kick >= first fold start)."""
from common import *
import lightgbm as lgb, time, warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
FOLDS = ["2024-10-01","2025-01-01","2025-04-01","2025-07-01","2025-10-01","2026-01-01","2026-03-01"]
L = load_long()
L = L.sort_values(["pt","outcome"]).reset_index(drop=True)
L["w"] = 1.0/L.groupby(["unique_id","outcome"]).pt.transform("size")
L["drift"] = (L.c_l - L.own_l).where(L.c_close_gap_h < 1.0)
sig = lambda z: 1/(1+np.exp(-z))
P = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=800, feature_fraction=0.7, bagging_fraction=0.5,
         bagging_freq=1, lambda_l2=20, verbose=-1, num_threads=8, max_bin=127)
def fit_bin(tr, feats, rounds=300, seed=0, nmax=2_500_000):
    t = tr.sample(min(len(tr), nmax), random_state=seed)
    ds = lgb.Dataset(t[feats], t.won, weight=t.w, init_score=t.own_l, categorical_feature=["outcome"])
    return lgb.train({**P, "objective":"binary", "seed":seed}, ds, rounds)
def fit_reg(tr, feats, rounds=400, seed=0, nmax=2_500_000):
    t = tr[tr.drift.notna()]; t = t.sample(min(len(t), nmax), random_state=seed)
    ds = lgb.Dataset(t[feats], t.drift, weight=t.w, categorical_feature=["outcome"])
    return lgb.train({**P, "objective":"huber", "alpha":0.3, "seed":seed}, ds, rounds)
def calib_X(l, d):
    X = pd.DataFrame({"l":l, "l2":l**2, "l3":l**3})
    for t in (0,1):
        X[f"t{t}"] = (d.tier.values==t)*1.0; X[f"t{t}l"] = X[f"t{t}"]*l
    X["draw"] = (d.outcome.values==1)*1.0; X["away"] = (d.outcome.values==2)*1.0
    X["drawl"] = X.draw*l
    return X
out = []
for a, b in zip(FOLDS[:-1], FOLDS[1:]):
    t0 = time.time(); a, b = pd.Timestamp(a), pd.Timestamp(b)
    tr = L[L.kick < a]; te = L[(L.kick >= a) & (L.kick < b)].copy()
    # M1 level logistic on closing rows
    cl = tr[(tr.kind==2)]
    cl = tr.loc[tr.groupby(["unique_id","outcome"]).t.idxmax()]
    lr = LogisticRegression(C=10, max_iter=1000).fit(calib_X(cl.own_l.values, cl), cl.won)
    te["p_lvl"] = lr.predict_proba(calib_X(te.own_l.values, te))[:,1]
    m3 = fit_bin(tr, FEATS); te["p_g_all"] = sig(te.own_l + m3.predict(te[FEATS], raw_score=True))
    # M4 two-stage
    m4 = fit_reg(tr, FEATS); te["drift_hat"] = m4.predict(te[FEATS])
    te["p_2s"] = lr.predict_proba(calib_X((te.own_l + te.drift_hat).values, te))[:,1]
    te["p_close"] = lr.predict_proba(calib_X(te.c_l.fillna(0).values, te))[:,1]
    te["fold"] = str(a.date())
    out.append(te[["pt","outcome","fold","p_lvl","p_g_all","drift_hat","p_2s","p_close"]])
    print(a.date(), len(tr), len(te), round(time.time()-t0), flush=True)
    if b == pd.Timestamp(FOLDS[-1]):
        imp = pd.Series(m3.feature_importance("gain"), FEATS).sort_values(ascending=False); print((imp/imp.sum()).round(3).head(25).to_string())
        imp = pd.Series(m4.feature_importance("gain"), FEATS).sort_values(ascending=False); print((imp/imp.sum()).round(3).head(25).to_string())
pd.concat(out).to_parquet(f"{W}/oof.parquet")
