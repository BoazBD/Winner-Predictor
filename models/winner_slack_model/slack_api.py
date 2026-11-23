"""In-process API for the Winner 'rounding slack' model (soccer 1X2).

    from slack_api import SlackModel
    m = SlackModel()                    # loads ./model/model.joblib
    out = m.score(df)                   # df: raw rows, processed_winner schema (strings OK)
    live = m.score(df, latest=True)     # only the latest snapshot per market

Scores the de-duplicated history (one row per price segment, see odds_history.py) plus each market's latest
snapshot. The bet flag depends only on the prices, so its first trigger is the same as on every raw snapshot.
Returns one row per (scored snapshot, outcome): unique_id, result_id, run_time, kick, league, home, away,
outcome (0 home / 1 draw / 2 away), pick, odds, p_mkt, p_model, ev, M, rr, flag.
flag=True  -> the policy says BET this outcome now (favourite, odds<=1.45, 90%-payout margin tier, rr>=0.01).
Bet each (unique_id, outcome) only once: the first time flag becomes True.
"""
import os, numpy as np, pandas as pd, joblib
from fable_core import history_rows, snapshot_features, design, flag, POLICY

HERE = os.path.dirname(os.path.abspath(__file__))


class SlackModel:
    def __init__(self, model_dir=os.path.join(HERE, 'model')):
        self.art = joblib.load(os.path.join(model_dir, 'model.joblib'))

    def score(self, df, latest=False):
        s = history_rows(df, current=True)
        if latest:
            s = s.groupby('unique_id').tail(1)
        if s.empty:
            return pd.DataFrame()
        L = snapshot_features(s)
        X = design(L)[self.art['cols']]
        p = np.zeros(len(L))
        for g, (sc, m) in self.art['models'].items():
            sel = (L.M < 1.14).values if g == '111' else (L.M >= 1.14).values
            if sel.any():
                p[sel] = m.predict_proba(sc.transform(X[sel]))[:, 1]
        L['p_model'] = p / pd.Series(p).groupby([L.unique_id, L.run_time]).transform('sum').values
        L['ev'] = L.p_model * L.odds - 1
        L['flag'] = flag(L, self.art.get('policy', POLICY))
        names = s.set_index('unique_id')[['home', 'away']].groupby(level=0).last()
        L = L.join(names, on='unique_id')
        L['pick'] = np.select([L.outcome == 0, L.outcome == 1], [L.home, 'Draw'], L.away)
        return L[['unique_id', 'result_id', 'run_time', 'kick', 'league', 'home', 'away', 'outcome', 'pick', 'odds',
                  'p_mkt', 'p_model', 'ev', 'M', 'rr', 'flag']]
