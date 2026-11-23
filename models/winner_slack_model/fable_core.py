"""Shared, per-snapshot feature code for the final policy (soccer 1X2, Winner).

Everything here uses only the snapshot row itself (prices at run_time) plus the market's earlier rows (for
first-trigger de-duplication), so it is safe to run on live odds.
"""
import os, sys
import duckdb, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # odds_history.py, one level up
from odds_history import soccer_1x2_history

REGIMES = np.array([1 / 0.9, 1 / 0.85, 1 / 0.8])   # Winner payout regimes 90% / 85% / 80%

# ---- final policy parameters (chosen on kick < 2026-01-01 data, see REPORT.md) ----
POLICY = dict(regime=1 / 0.9, odds_max=1.45, rr_min=0.01, favourite_only=True)
MARKET_COLS = ('result_id', 'league', 'option1', 'option3', 'bet1_won', 'tie_won', 'bet2_won')


def history_rows(src, current=False):
    """De-duplicated snapshot rows (shared odds_history step): one row per price segment, at its first scraper run
    (first appearance, price or kickoff change, reappearance after a withdrawal). current=True also adds each
    market's latest state (last segment at its last run): the row a live bet is placed on.
    src: raw rows (DataFrame, strings OK) or parquet path. Label columns are kept if src has them."""
    cols = src.columns if isinstance(src, pd.DataFrame) else duckdb.sql(f"select * from '{src}' limit 0").columns
    seg, _ = soccer_1x2_history(src, keep=tuple(c for c in MARKET_COLS if c in cols))
    rows = [seg.assign(run_time=seg.t_start)]
    if current:
        last = seg.groupby('unique_id').tail(1)
        rows.append(last[last.t_end > last.t_start].assign(run_time=lambda x: x.t_end))
    s = pd.concat(rows).rename(columns={'option1': 'home', 'option3': 'away'})
    return s.drop(columns=['seg_no', 't_start', 't_end', 'i_start', 'i_end', 'n_snaps', 'gap_runs']) \
            .sort_values(['unique_id', 'run_time']).reset_index(drop=True)


def power_devig(P):
    a = np.ones(len(P)); lp = np.log(P)
    for _ in range(30):
        pa = np.exp(a[:, None] * lp); f = pa.sum(1) - 1; df = (pa * lp).sum(1)
        a = a - f / df
    pa = np.exp(a[:, None] * lp)
    return pa / pa.sum(1, keepdims=True)


def snapshot_features(s):
    """s: rows from history_rows. Returns long frame: one row per (snapshot, outcome) with features."""
    O = s[['r1', 'r2', 'r3']].values.astype(float)
    imp = 1 / O
    ovr = imp.sum(1)
    pp = power_devig(imp)
    M = REGIMES[np.abs(ovr[:, None] - REGIMES[None, :]).argmin(1)]
    mins = (s.kick - s.run_time).dt.total_seconds().values / 60
    rank = O.argsort(1).argsort(1)            # 0 = favourite (lowest odds)
    res = np.select([s.bet1_won.fillna(False).astype(bool).values, s.tie_won.fillna(False).astype(bool).values,
                     s.bet2_won.fillna(False).astype(bool).values], [0, 1, 2], -1) if 'bet1_won' in s else np.full(len(s), -1)
    rows = []
    for i in range(3):
        other = ovr - imp[:, i]
        u = 1 / (M - other)                       # price this outcome would have with zero rounding
        rows.append(pd.DataFrame({
            'unique_id': s.unique_id.values, 'result_id': s.result_id.values, 'run_time': s.run_time.values,
            'kick': s.kick.values, 'league': s.league.values, 'outcome': i if i < 2 else 2, 'odds': O[:, i],
            'p_mkt': pp[:, i], 'ovr': ovr, 'M': M, 'rr': O[:, i] / u - 1, 'rank': rank[:, i], 'mins_to_kick': mins,
            'res': res}))
    L = pd.concat(rows, ignore_index=True)
    L['outcome'] = np.where(L.outcome == 1, 1, np.where(L.outcome == 0, 0, 2))
    return L


def design(L):
    lp = np.log(L.p_mkt / (1 - L.p_mkt))
    r176 = ((L.M > 1.14) & (L.M < 1.2)).astype(float)
    fav = (L['rank'] == 0).astype(float)
    return pd.DataFrame({'lp': lp, 'lp2': lp ** 2, 'lp176': lp * r176, 'r176': r176,
                         'rr': L.rr * 10, 'rr_fav': L.rr * 10 * fav, 'fav': fav,
                         'draw': (L.outcome == 1).astype(float), 'away': (L.outcome == 2).astype(float),
                         'lh': np.log1p(L.mins_to_kick.clip(lower=0) / 60)}, index=L.index).astype(float)


def flag(L, policy=POLICY):
    f = (np.abs(L.M - policy['regime']) < 1e-6) & (L.odds <= policy['odds_max'] + 1e-9) & (L.rr >= policy['rr_min'])
    if policy.get('favourite_only', True):
        f &= (L['rank'] == 0)
    return f.values


def first_trigger(L):
    """Bets: earliest flagged snapshot per (market, outcome)."""
    b = L[L.flag].sort_values('run_time').drop_duplicates(['unique_id', 'outcome'])
    return b.sort_values('run_time').reset_index(drop=True)
