"""s05b: can a model combine the wallet/insider features into a better predictor of the move LEFT after we can act?

LightGBM regression, target = mv_d30 (mid move in the bet's direction from 30 min after the bet to the close, clipped
to +-0.2), on bets >= $200 with grid marks. Features use only information available at the bet time (no final volume).
Train-period predictions are out-of-fold (5 contiguous time blocks); val/test predictions come from a model fit on
all train bets. Output: data/bets_ml.parquet (bet_id, ml), out/s05b_ml.json (correlations / decile table).
"""
import os, json
import numpy as np, pandas as pd, lightgbm as lgb
from common import D, OUT

FEATS = ["lusd", "share", "lvol_before", "mins_to_kick", "tok_px", "dir", "k", "fresh", "age_days", "prior_games",
         "lprior_usd", "series_conc", "max_levels", "n_tx", "impact", "cluster_n", "cluster_fresh", "s_n", "s_move",
         "s_clv", "s_res", "s_tmove", "top"]


def prep(b):
    b = b.copy()
    b["lusd"] = np.log10(b.usd); b["lvol_before"] = np.log10(b.vol_before + 1)
    b["lprior_usd"] = np.log10(b.prior_usd.fillna(0) + 1)
    b["tok_px"] = np.where(b.dir == 1, b.ypx, 1 - b.ypx)
    b["fresh"] = b.fresh.astype(float); b["top"] = (b.tier == "top").astype(float)
    return b


def main():
    b = pd.read_parquet(os.path.join(D, "bets.parquet"))
    b["bet_id"] = b.index.astype(np.int64)
    b = b[(b.usd >= 200) & b.m_d0.notna() & (b.mins_to_kick >= 30)]
    b = prep(b)
    b["y"] = b.mv_d30.clip(-0.2, 0.2)
    tr = b[(b.period == "train") & b.y.notna()].sort_values("ts")
    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=8)
    oof = pd.Series(np.nan, index=tr.index)
    blocks = np.array_split(np.arange(len(tr)), 5)
    for bl in blocks:
        mtr = np.ones(len(tr), bool); mtr[bl] = False
        m = lgb.train(params, lgb.Dataset(tr[FEATS][mtr], tr.y[mtr]), 300)
        oof.iloc[bl] = m.predict(tr[FEATS].iloc[bl])
    m = lgb.train(params, lgb.Dataset(tr[FEATS], tr.y), 300)
    rest = b[b.period != "train"]
    pred = pd.concat([oof, pd.Series(m.predict(rest[FEATS]), index=rest.index)])
    b["ml"] = pred
    b[["bet_id", "ml"]].to_parquet(os.path.join(D, "bets_ml.parquet"))
    rep = {"importance": dict(zip(FEATS, m.feature_importance("gain").round(0).tolist()))}
    for per in ["train", "val", "test"]:
        x = b[(b.period == per) & b.y.notna()]
        x = x.assign(dec=pd.qcut(x.ml, 10, labels=False, duplicates="drop"))
        rep[per] = dict(n=len(x), spearman=float(x.ml.corr(x.y, method="spearman")),
                        decile_mean_move=x.groupby("dec").y.mean().round(4).tolist(),
                        decile_mean_pred=x.groupby("dec").ml.mean().round(4).tolist())
    json.dump(rep, open(os.path.join(OUT, "s05b_ml.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
