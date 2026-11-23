"""Live scoring API for the oddsflow model (Winner soccer 1X2).

    from predict_api import Model
    m = Model()                        # loads ./model/  (numpy, pandas, duckdb; no lightgbm needed)
    out = m.score(df, latest=True)     # df: raw rows of the Athena table `api_odds` (strings are fine)

`df` must hold the full snapshot history of the open markets (all sports / markets are fine; everything that
is not a soccer 1X2 market is ignored, but ALL rows are used to recognise scraper runs, so do not pre-filter).

Output, one row per (market, outcome):
    unique_id, result_id, run_time, kick, outcome (0 home / 1 draw / 2 away), odds, p_model, ev, flag
    + extras: tier (0 = 90% payout, 1 = 85%, 2 = 80%), p_market (margin-free, multiplicative), drift_hat,
      hours_to_kick, flag_before (the rule already fired earlier for this market/outcome -> already bet).
p_model sums to 1 over the three outcomes. flag=True: bet this outcome now (ev > threshold).
Bet each (unique_id, outcome) once: act on flag only when flag_before is False (or keep your own ledger).

latest=True  : only the latest snapshot, and only for markets present in the newest scraper run of df
               (a market that is absent from the newest run cannot be bet).
latest=False : every snapshot of every market (for backtests).
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # odds_history.py: the shared cleaning step, one level up
from odds_history import soccer_1x2_history
from features import seg_stats, make_points, make_points_full, make_points_latest
from table import long_table
import model_lib as ml
OUT = ["unique_id", "result_id", "run_time", "kick", "outcome", "odds", "p_model", "ev", "flag",
       "tier", "p_market", "drift_hat", "hours_to_kick", "flag_before"]


class Model:
    def __init__(self, model_dir=os.path.join(HERE, "model"), threshold=None):
        self.art = ml.load(model_dir)
        self.threshold = self.art.get("ev_threshold", ml.EV_THRESHOLD) if threshold is None else threshold

    def _score_points(self, st, pts):
        L = long_table(st, pts)
        L["seg_row"] = np.tile(pts.seg_row.values, 3)
        return L.join(ml.predict_long(self.art, L))

    def score(self, df, latest=True):
        if len(df) == 0:
            return pd.DataFrame(columns=OUT)
        seg, runs = soccer_1x2_history(df)
        if len(seg) == 0:
            return pd.DataFrame(columns=OUT)
        st = seg_stats(seg)
        rid = seg.groupby("unique_id").result_id.first()
        if latest:
            pts = make_points_latest(seg)
            pts = pts[pts.run_idx.values == len(runs) - 1].reset_index(drop=True)   # present in the newest run
            if len(pts) == 0:
                return pd.DataFrame(columns=OUT)
            L = self._score_points(st, pts)
            # was the rule already satisfied earlier? (checked at every price change / reappearance / time checkpoint)
            hist = make_points(seg, runs, closing=False)
            hist = hist[hist.seg_row.isin(np.flatnonzero(seg.unique_id.isin(L.unique_id.unique()).values))]
            H = self._score_points(st, hist.reset_index(drop=True))
            H = H.merge(L[["unique_id", "outcome", "t"]].rename(columns={"t": "t_now"}), on=["unique_id", "outcome"])
            fired = H[(H.ev > self.threshold) & (H.t < H.t_now)][["unique_id", "outcome"]].drop_duplicates()
            fired["flag_before"] = True
            L = L.merge(fired, on=["unique_id", "outcome"], how="left")
            L["flag_before"] = L.flag_before.fillna(False).astype(bool)
        else:
            L = self._score_points(st, make_points_full(seg, runs))
            fl = L.ev > self.threshold
            order = L.assign(_f=fl).sort_values("t", kind="stable")
            L["flag_before"] = (order.groupby(["unique_id", "outcome"])._f.cumsum() - order._f).reindex(L.index) > 0
        p_m = 1 / (1 + np.exp(-L.own_m.values))
        out = pd.DataFrame({
            "unique_id": L.unique_id.values, "result_id": rid.reindex(L.unique_id.values).values,
            "run_time": L.t.values, "kick": L.kick.values, "outcome": L.outcome.values.astype(int),
            "odds": L.own_o.values, "p_model": L.p_model.values, "ev": L.ev.values,
            "flag": (L.ev.values > self.threshold) & (L.ttk_h.values > 0),
            "tier": L.tier.values.astype(int),
            "p_market": p_m / pd.Series(p_m).groupby(L.pt.values).transform("sum").values,
            "drift_hat": L.drift_hat.values, "hours_to_kick": L.ttk_h.values, "flag_before": L.flag_before.values})
        return out.sort_values(["unique_id", "run_time", "outcome"]).reset_index(drop=True)[OUT]


if __name__ == "__main__":
    import sys
    d = pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet") else pd.read_csv(sys.argv[1], dtype=str)
    o = Model().score(d, latest=True)
    print(o[o.flag].to_string() if o.flag.any() else "no flagged outcomes", f"\n{len(o)//3} markets scored")
