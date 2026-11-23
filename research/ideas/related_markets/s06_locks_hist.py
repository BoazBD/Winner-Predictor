"""Step 6: historical search for TRUE hedged locks between nested markets of the same game (pre-kickoff, real prints).

Ask evidence: a real print where a taker BOUGHT token X at q means X could be bought at q at that second.
(1X2 prints from research/edge/data/master.parquet; side prints from data/panel.parquet.)
A pair (L1, L2) is a lock structure if payoff(L1)+payoff(L2) >= 1 for EVERY scoreline (checked numerically below on 0..10 x 0..10):
  totals L Over + totals L+1 Under | 1X2 home Yes + spread home -1.5 'B' (away +1.5) | spread -1.5 A + spread -2.5 B (same team)
  O1.5 + BTTS No | O0.5 + Draw Yes | team-total O0.5 + 1X2 win No (same team) | team-total O0.5 + BTTS No
  game O/U L Over + team-total L Under | team-total O1.5 + spread -1.5 B (same team)
For every print of one leg we take the latest print of the other leg within W seconds before it (W = 60, 300);
margin = 1 - q1 - q2 - fee(q1) - fee(q2) (fee 0.05 q(1-q) per share where enabled). size = min of the two print sizes.
This is an UPPER BOUND (assumes both asks coexisted and we were first). Output: out/locks_hist.csv, out/locks_hist_summary.csv
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
FEE = lambda q: 0.05 * q * (1 - q)
G = 11; I = np.arange(G); HG = np.repeat(I, G).reshape(G, G); AG = HG.T


def leg_payoff(leg):
    k, s = leg.rsplit(":", 1)
    if k.startswith("ml|"):
        side = k.split("|")[1]
        P = {"home": HG > AG, "draw": HG == AG, "away": HG < AG}[side].astype(float)
        return P if s == "yes" else 1 - P
    smt, line, team = k.split("|"); L = abs(float(line))
    if smt == "totals":
        P = HG + AG > L
    elif smt == "both_teams_to_score":
        P = (HG > 0) & (AG > 0)
    elif smt == "spreads":
        P = (HG - AG > L) if team == "home" else (AG - HG > L)
    elif smt == "soccer_team_totals":
        P = (HG > L) if team == "home" else (AG > L)
    P = P.astype(float)
    return P if s == "A" else 1 - P


def pairs():
    ps = []
    for L in [0.5, 1.5, 2.5, 3.5, 4.5]:
        ps.append((f"totals|{L}|:A", f"totals|{L + 1}|:B"))
    for t in ["home", "away"]:
        ps += [(f"ml|{t}:yes", f"spreads|-1.5|{t}:B"), (f"spreads|-1.5|{t}:A", f"spreads|-2.5|{t}:B"),
               (f"soccer_team_totals|0.5|{t}:A", f"ml|{t}:no"), (f"soccer_team_totals|0.5|{t}:A", "both_teams_to_score|0.0|:B"),
               (f"soccer_team_totals|1.5|{t}:A", f"spreads|-1.5|{t}:B")]
        for L in [0.5, 1.5, 2.5]:
            ps.append((f"totals|{L}|:A", f"soccer_team_totals|{L}|{t}:B"))
    ps += [("totals|1.5|:A", "both_teams_to_score|0.0|:B"), ("totals|0.5|:A", "ml|draw:yes")]
    for a, b in ps:
        assert (leg_payoff(a) + leg_payoff(b)).min() >= 1 - 1e-9, (a, b)
    return ps


def main():
    p = pd.read_parquet(os.path.join(HERE, "data", "panel.parquet"), columns=["unique_id", "mkey", "ts", "pA", "buyA", "size", "fees", "kick_utc"])
    p["leg"] = p.mkey + ":" + np.where(p.buyA, "A", "B")
    p["q"] = np.where(p.buyA, p.pA, 1 - p.pA).round(4)
    ev = [p[["unique_id", "leg", "ts", "q", "size", "fees"]]]
    games = p.unique_id.unique()
    gm = pd.read_parquet(os.path.join(ROOT, "lead_lag", "data", "games_meta.parquet"), columns=["unique_id", "swapped"])
    m = pd.read_parquet(os.path.join(ROOT, "edge", "data", "master.parquet"), columns=["unique_id", "k", "yes", "q", "size", "ts", "rel_min", "fees0"],
                        filters=[("unique_id", "in", list(games))])
    m = m[m.rel_min < 0].merge(gm, on="unique_id")
    side = np.where(m.k == 1, "draw", np.where((m.k == 0) != m.swapped, "home", "away"))
    m["leg"] = "ml|" + pd.Series(side, index=m.index) + ":" + np.where(m.yes, "yes", "no")
    m["q"] = m.q.astype("float64").round(4)
    ev.append(m[["unique_id", "leg", "ts", "q", "size"]].assign(fees=m.fees0.astype(bool)))
    E = pd.concat(ev, ignore_index=True)
    E = E[(E.q > 0.0005) & (E.q < 0.9995)].sort_values("ts")
    E["cost"] = E.q + np.where(E.fees, FEE(E.q), 0)
    kick = p.groupby("unique_id").kick_utc.first()
    rows = []
    for W in [60, 300]:
        for a, b in pairs():
            A = E[E.leg == a]; B = E[E.leg == b]
            if not len(A) or not len(B):
                continue
            for X, Y in ((A, B), (B, A)):
                j = pd.merge_asof(X, Y[["unique_id", "ts", "q", "size", "cost"]].rename(columns={"ts": "ts2", "q": "q2", "size": "size2", "cost": "cost2"}).sort_values("ts2"),
                                  left_on="ts", right_on="ts2", by="unique_id", direction="backward", tolerance=W)
                j = j[j.q2.notna()]
                j["margin"] = 1 - j.cost - j.cost2
                j["shares"] = np.minimum(j["size"], j.size2)
                hit = j[j.margin > 1e-4]
                for r in hit.itertuples():
                    rows.append(dict(W=W, pair=f"{a} + {b}", unique_id=r.unique_id, ts=r.ts, q1=r.q, q2=r.q2, margin=r.margin, shares=r.shares,
                                     profit=r.margin * r.shares, kick=kick.get(r.unique_id)))
            print(W, a, b, len(rows), flush=True)
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(OUT, "locks_hist.csv"), index=False)
    if len(L):
        L["per"] = pd.to_datetime(L.kick).map(lambda k: "train" if k < pd.Timestamp("2026-07-01") else ("valid" if k < pd.Timestamp("2026-10-01") else "test"))
        # one lock per game x pair (best), profit capped at $100 notional
        best = L.sort_values("profit").groupby(["W", "pair", "unique_id"]).tail(1)
        S = best.groupby(["W", "per"]).agg(games_with_lock=("unique_id", "nunique"), locks=("pair", "size"), med_margin=("margin", "median"),
                                           med_shares=("shares", "median"), profit_sum=("profit", "sum")).reset_index()
        S["games_in_sample"] = S.per.map(p.assign(per=pd.to_datetime(p.kick_utc).map(lambda k: "train" if k < pd.Timestamp("2026-07-01") else ("valid" if k < pd.Timestamp("2026-10-01") else "test"))).groupby("per").unique_id.nunique())
        S.to_csv(os.path.join(OUT, "locks_hist_summary.csv"), index=False)
        print(S.to_string())
        print(best.groupby(["W", "pair"]).agg(n=("unique_id", "size"), m=("margin", "median"), sh=("shares", "median"), pr=("profit", "sum")).to_string())


if __name__ == "__main__":
    main()
