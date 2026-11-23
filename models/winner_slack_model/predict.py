"""Score raw Winner odds (same schema as processed_winner_*.parquet) with the final policy.

usage:
  python predict.py RAW.parquet --model model_pre2026H2 --out scores.parquet            # de-duplicated history x outcome
  python predict.py RAW.parquet --model model_pre2026H2 --latest --out live.csv          # latest snapshot per market (live use)
  python predict.py RAW.parquet --model model_pre2026H2 --bets --out bets.parquet        # placed bets (first flag per market/outcome)
  optional: --kick-min / --kick-max to restrict kickoff window.

Output columns: unique_id, result_id, run_time, kick, league, outcome (0 home,1 draw,2 away), odds, p_mkt (power de-vig),
p_model (calibrated LR, normalised per snapshot), ev = p_model*odds-1, M (margin regime), rr (rounding residual), flag.
"""
import argparse
import numpy as np, pandas as pd, joblib
from fable_core import history_rows, snapshot_features, design, flag, first_trigger


def score(raw, model_dir, kick_min=None, kick_max=None, latest=False):
    s = history_rows(raw, current=True)
    if kick_min is not None:
        s = s[s.kick >= pd.Timestamp(kick_min)]
    if kick_max is not None:
        s = s[s.kick < pd.Timestamp(kick_max)]
    if latest:
        s = s.groupby('unique_id', observed=True).tail(1)
    L = snapshot_features(s)
    art = joblib.load(f'{model_dir}/model.joblib')
    X = design(L)[art['cols']]
    p = np.zeros(len(L))
    for g, (sc, m) in art['models'].items():
        sel = (L.M < 1.14).values if g == '111' else (L.M >= 1.14).values
        if sel.any():
            p[sel] = m.predict_proba(sc.transform(X[sel]))[:, 1]
    L['p_model'] = p
    L['p_model'] = L.p_model / L.groupby(['unique_id', 'run_time'], observed=True).p_model.transform('sum')
    L['ev'] = L.p_model * L.odds - 1
    L['flag'] = flag(L, art['policy'])
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--latest', action='store_true'); ap.add_argument('--bets', action='store_true')
    ap.add_argument('--kick-min'); ap.add_argument('--kick-max')
    a = ap.parse_args()
    L = score(a.raw, a.model, a.kick_min, a.kick_max, a.latest)
    if a.bets:
        b = first_trigger(L)
        b['stake'] = 1.0
        L = b
    cols = ['unique_id', 'result_id', 'run_time', 'kick', 'league', 'outcome', 'odds', 'p_mkt', 'p_model', 'ev', 'M', 'rr', 'flag'] + \
           (['stake'] if a.bets else [])
    L = L[cols]
    (L.to_csv if a.out.endswith('.csv') else L.to_parquet)(a.out, index=False)
    print('wrote', len(L), 'rows;', int(L.flag.sum()), 'flagged')


if __name__ == '__main__':
    main()
