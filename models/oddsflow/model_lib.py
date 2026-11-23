"""oddsflow model: closing-line drift (gradient-boosted trees) + favourite-longshot calibration.

    latent logit now  --(+ predicted drift to the close)-->  expected closing logit
                      --(logistic calibration fitted on closing prices vs results)-->  win probability

Inference needs only numpy / pandas (trees are evaluated in numpy; lightgbm is needed for training only).
"""
import json
import os
import numpy as np
import pandas as pd
from features import FEATS, seg_stats, point_features, to_long

EV_THRESHOLD = 0.0          # flag when p_model * odds - 1 > EV_THRESHOLD   (pre-registered, see PREREGISTRATION.md)
CAL_COLS = ["l", "l2", "l3", "t0", "t0l", "t1", "t1l", "draw", "away", "drawl"]
DRIFT_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500,
                    feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=20, verbose=-1,
                    num_threads=8, max_bin=127)
DRIFT_ROUNDS, DRIFT_SEEDS, DRIFT_MAX_ROWS = 600, 3, 2_500_000


def calib_X(l, tier, outcome):
    l = np.asarray(l, float); tier = np.asarray(tier); outcome = np.asarray(outcome)
    t0 = (tier == 0) * 1.0; t1 = (tier == 1) * 1.0; draw = (outcome == 1) * 1.0
    return np.column_stack([l, l ** 2, l ** 3, t0, t0 * l, t1, t1 * l, draw, (outcome == 2) * 1.0, draw * l])


# ---------------------------------------------------------------- numpy tree ensemble
class Trees:
    """Minimal LightGBM regression-ensemble evaluator (numerical splits, no missing values)."""

    def __init__(self, arrs):
        self.a = arrs

    @staticmethod
    def from_boosters(boosters):
        feat, thr, left, right, value, root, base = [], [], [], [], [], [], 0
        for b in boosters:
            for t in b.dump_model()["tree_info"]:
                nodes = []

                def walk(nd):
                    i = len(nodes); nodes.append(None)
                    if "leaf_value" in nd and "split_feature" not in nd:
                        nodes[i] = (-1, 0.0, -1, -1, nd["leaf_value"])
                    else:
                        assert nd["decision_type"] == "<=", nd["decision_type"]
                        l = walk(nd["left_child"]); r = walk(nd["right_child"])
                        nodes[i] = (nd["split_feature"], nd["threshold"], l, r, 0.0)
                    return i
                walk(t["tree_structure"])
                root.append(base)
                for f, th, l, r, v in nodes:
                    feat.append(f); thr.append(th); value.append(v)
                    left.append(l + base if l >= 0 else -1); right.append(r + base if r >= 0 else -1)
                base += len(nodes)
        return Trees(dict(feat=np.array(feat, np.int32), thr=np.array(thr, np.float64), left=np.array(left, np.int32),
                          right=np.array(right, np.int32), value=np.array(value, np.float64),
                          root=np.array(root, np.int32), n_models=np.array(len(boosters))))

    def predict(self, X):
        a = self.a; X = np.ascontiguousarray(X, dtype=np.float64); n = len(X)
        out = np.zeros(n)
        rows = np.arange(n)
        for r in a["root"]:
            node = np.full(n, r, np.int32)
            act = rows if a["feat"][r] >= 0 else rows[:0]
            while len(act):
                nd = node[act]
                go_left = X[act, a["feat"][nd]] <= a["thr"][nd]
                nd = np.where(go_left, a["left"][nd], a["right"][nd])
                node[act] = nd
                act = act[a["feat"][nd] >= 0]
            out += a["value"][node]
        return out / float(a["n_models"])


# ---------------------------------------------------------------- fit / predict
def fit(L, verbose=False):
    """L: long decision-point table (features + won, c_l, c_close_gap_h, unique_id, t). Returns artifact dict."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    cl = L.loc[L.groupby(["unique_id", "outcome"]).t.idxmax()]
    cl = cl[cl.ttk_h < 1.0]
    lr = LogisticRegression(C=10, max_iter=2000).fit(calib_X(cl.own_l, cl.tier, cl.outcome), cl.won)
    tr = L[L.c_close_gap_h < 1.0]
    w = 1.0 / tr.groupby(["unique_id", "outcome"]).t.transform("size")
    tr = tr.assign(_w=w, _d=tr.c_l - tr.own_l)
    tr = tr.sample(min(len(tr), DRIFT_MAX_ROWS), random_state=0)
    bs = [lgb.train({**DRIFT_PARAMS, "seed": s}, lgb.Dataset(tr[FEATS].values.astype(np.float32), tr._d, weight=tr._w),
                    DRIFT_ROUNDS) for s in range(DRIFT_SEEDS)]
    trees = Trees.from_boosters(bs)
    chk = tr[FEATS].values.astype(np.float32)[:20000]
    err = np.abs(trees.predict(chk) - np.mean([b.predict(chk) for b in bs], 0)).max()
    assert err < 1e-9, err
    if verbose:
        print(f"calibration rows {len(cl)}, drift rows {len(tr)}, numpy-vs-lightgbm max diff {err:.2e}")
    return dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), trees=trees, feats=FEATS, boosters=bs,
                n_cal=int(len(cl)), n_drift=int(len(tr)))


def predict_long(art, L):
    """Adds drift_hat, p_model (normalised over the 3 outcomes of a point; rows must be complete triples), ev."""
    X = L[art["feats"]].values.astype(np.float32)          # float32 everywhere so training and live agree
    if art.get("boosters"):                                 # fast path for large backtests (same trees)
        drift = np.mean([b.predict(X) for b in art["boosters"]], 0)
    else:
        drift = art["trees"].predict(X)
    z = calib_X(L.own_l.values + drift, L.tier.values, L.outcome.values) @ np.array(art["coef"]) + art["intercept"]
    p = 1.0 / (1.0 + np.exp(-z))
    tot = pd.Series(p).groupby(L.pt.values).transform("sum").values
    out = L[[]].copy()
    out["drift_hat"] = drift
    out["p_model"] = p / tot
    out["ev"] = out.p_model * L.own_o.values - 1.0
    return out


def close_prob(art, c_l, tier, outcome):
    """Calibrated probability implied by an (actual) closing logit - used for the low-variance closing-EV check."""
    z = calib_X(c_l, tier, outcome) @ np.array(art["coef"]) + art["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def save(art, d):
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "trees.npz"), **art["trees"].a)
    meta = {k: v for k, v in art.items() if k not in ("trees", "boosters")}
    json.dump(meta, open(os.path.join(d, "model.json"), "w"), indent=1)


def load(d):
    art = json.load(open(os.path.join(d, "model.json")))
    z = np.load(os.path.join(d, "trees.npz"))
    art["trees"] = Trees({k: z[k] for k in z.files})
    return art
