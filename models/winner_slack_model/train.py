"""Train the probability model used for p_model / EV output.
usage: python train.py RAW.parquet OUT_DIR [--cutoff 2026-07-01]
The bet flag itself is a fixed rule (fable_core.POLICY); this model only supplies calibrated probabilities.
Trained on the de-duplicated history (fable_core.history_rows): one row per price segment, weighted 1/rows per
(market, outcome).
"""
import sys, json, argparse
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fable_core import history_rows, snapshot_features, design, POLICY


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('out'); ap.add_argument('--cutoff', default=None)
    a = ap.parse_args()
    s = history_rows(a.raw)
    if a.cutoff:
        s = s[s.kick < pd.Timestamp(a.cutoff)]
    L = snapshot_features(s)
    L = L[L.res >= 0]
    y = (L.res == L.outcome).astype(int).values
    X = design(L)
    key = L.unique_id.astype(str) + '_' + L.outcome.astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    models, info = {}, {'cutoff': a.cutoff, 'policy': POLICY}
    for g in ('111', 'other'):
        sel = (L.M < 1.14).values if g == '111' else (L.M >= 1.14).values
        sc = StandardScaler().fit(X[sel])
        m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[sel]), y[sel], sample_weight=w[sel])
        models[g] = (sc, m)
        info[g] = {'n_rows': int(sel.sum()), 'n_markets': int(L.unique_id[sel].nunique()),
                   'coef_std': dict(zip(X.columns, np.round(m.coef_[0], 4).tolist()))}
        print(g, info[g])
    import os; os.makedirs(a.out, exist_ok=True)
    joblib.dump({'models': models, 'cols': list(X.columns), 'cutoff': a.cutoff, 'policy': POLICY}, f'{a.out}/model.joblib')
    json.dump(info, open(f'{a.out}/model_info.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
