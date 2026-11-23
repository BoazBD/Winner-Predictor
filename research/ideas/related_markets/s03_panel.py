"""Step 3: trade-level panel of pre-kickoff side-market prints with fair values from the 1X2 market.

Inputs : data/trades_cache/*.parquet (s02), data/side_markets.parquet (s01), lead_lag grid (15-min 1X2 midpoints, float32)
Markets: full-time goal markets only: totals, both_teams_to_score, spreads (-1.5/-2.5), soccer_team_totals.
Each print is expressed in token-A terms (A = first outcome: Over / Yes / the -line team / team Over):
  pA   = price of token A implied by the print (float64, rounded to 4 dp)
  buyA = the taker bought token A (= taker sold token B). A taker BUY of token X at q proves X's ask was <= q then.
Fair values (probability token A wins):
  m1 = Dixon-Coles (rho=-0.08) fitted to the 1X2 midpoints only (home/away win probs, normalised).
  m2 = Dixon-Coles fitted to 1X2 + the game's own O/U 2.5 reference price (mean of last taker-buy and last taker-sell
       print of O/U 2.5, both within 120 min) -- least squares in logit space over (pH, pA, pOver2.5).
1X2 midpoint / O/U 2.5 reference use only data from minutes STRICTLY before the print minute;
m2 is circular for O/U 2.5 itself, so m2 is set to NaN for that market.
1X2 midpoint used = last 15-min grid point before the print minute (grid only covers the last 72 h).
Output: data/panel.parquet
"""
import os, sys, glob, numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Table, G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FT = ["totals", "both_teams_to_score", "spreads", "soccer_team_totals"]
I = np.arange(G); HG = np.repeat(I, G).reshape(G, G); AG = HG.T


def payoff(smt, line, mslug):
    L = abs(line) if pd.notna(line) else 0
    if smt == "totals":
        return HG + AG > L
    if smt == "both_teams_to_score":
        return (HG > 0) & (AG > 0)
    if smt == "spreads":
        return (HG - AG > L) if "-spread-home-" in mslug else (AG - HG > L)
    if smt == "soccer_team_totals":
        return (HG > L) if "-team-total-home-" in mslug else (AG > L)


def lg(p):
    p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))


def main():
    sm = pd.read_parquet(os.path.join(HERE, "data", "side_markets.parquet"))
    sm = sm[sm.smt.isin(FT) & sm.closed]
    sm = sm[~((sm.smt == "spreads") & (sm.line > 0))]          # rare positive-line spreads: skip
    sm["mkey"] = sm.smt + "|" + sm.line.fillna(0).astype(str) + "|" + np.where(sm.mslug.str.contains("-home-"), "home", np.where(sm.mslug.str.contains("-away-"), "away", ""))
    t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(HERE, "data", "trades_cache", "*.parquet")))], ignore_index=True).drop_duplicates()
    t = t.merge(sm[["cond", "tokA", "unique_id", "smt", "line", "mslug", "mkey", "resA", "fees", "kick_utc"]], on="cond")
    t["price"] = t.price.astype("float64").round(4)
    isA = t.token == t.tokA
    t["pA"] = np.where(isA, t.price, 1 - t.price).round(4)
    t["buyA"] = isA == (t.side == "BUY")
    t["minute"] = t.ts // 60
    t = t.drop(columns=["tokA", "token", "price"])
    print("trades", len(t), "games", t.unique_id.nunique(), flush=True)

    gm = pd.read_parquet(os.path.join(ROOT, "lead_lag", "data", "games_meta.parquet"), columns=["unique_id", "swapped"])
    grid = pd.read_parquet(os.path.join(ROOT, "lead_lag", "data", "grid.parquet"), columns=["unique_id", "t_min", "mins_to_kick", "p0", "p1", "p2"])
    grid = grid[grid.unique_id.isin(t.unique_id.unique())].merge(gm, on="unique_id")
    for c in ["p0", "p1", "p2"]:
        grid[c] = grid[c].astype("float64").round(4)
    grid["ph"] = np.where(grid.swapped, grid.p2, grid.p0); grid["pa"] = np.where(grid.swapped, grid.p0, grid.p2); grid["pd"] = grid.p1
    s = grid.ph + grid.pa + grid.pd
    grid = grid[(s > 0.9) & (s < 1.1)]
    grid["ph"] /= s; grid["pa"] /= s
    grid = grid[["unique_id", "t_min", "mins_to_kick", "ph", "pa"]].sort_values("t_min")
    t = t.sort_values("minute")
    t = pd.merge_asof(t, grid.rename(columns={"t_min": "minute"}), on="minute", by="unique_id", direction="backward", tolerance=15,
                      allow_exact_matches=False)   # strictly earlier minute: no intra-minute look-ahead
    t = t[t.ph.notna()].copy()
    print("with 1X2 grid", len(t), flush=True)

    tab = Table()
    Mf = tab.M.reshape(len(tab.M), -1)
    t["ix1"] = tab.idx(t.ph.to_numpy(), t.pa.to_numpy())
    # O/U 2.5 reference price per game
    o = t[t.mkey == "totals|2.5|"][["unique_id", "minute", "pA", "buyA"]].sort_values("minute")
    ref = []
    for side in (True, False):
        x = o[o.buyA == side][["unique_id", "minute", "pA"]].rename(columns={"pA": f"o25_{int(side)}", "minute": f"m_{int(side)}"})
        x["minute"] = x[f"m_{int(side)}"]
        ref.append(x)
    t = t.sort_values("minute")
    for x in ref:
        t = pd.merge_asof(t, x.sort_values("minute"), on="minute", by="unique_id", direction="backward", tolerance=120,
                          allow_exact_matches=False)
    t["o25"] = (t.o25_1 + t.o25_0) / 2
    p25 = Mf @ (HG + AG > 2.5).ravel().astype(np.float32)
    tree = cKDTree(np.c_[lg(tab.pH), lg(tab.pA), lg(p25)])
    ok = t.o25.notna().to_numpy()
    t["ix2"] = -1
    t.loc[ok, "ix2"] = tree.query(np.c_[lg(t.ph[ok]), lg(t.pa[ok]), lg(t.o25[ok])])[1]
    t["m1"] = np.nan; t["m2"] = np.nan
    for key, d in t.groupby("mkey"):
        r = sm[sm.mkey == key].iloc[0]
        v = Mf @ payoff(r.smt, r.line, r.mslug).ravel().astype(np.float32)
        t.loc[d.index, "m1"] = v[d.ix1.to_numpy()]
        okk = d.ix2.to_numpy() >= 0
        t.loc[d.index[okk], "m2"] = v[d.ix2.to_numpy()[okk]]
    t["m1"] = t.m1.astype("float64"); t["m2"] = t.m2.astype("float64")
    t.loc[t.mkey == "totals|2.5|", "m2"] = np.nan
    t = t.drop(columns=["o25_1", "o25_0", "m_1", "m_0", "ix1", "ix2"])
    t.to_parquet(os.path.join(HERE, "data", "panel.parquet"))
    print(t.groupby("mkey").agg(n=("pA", "size"), games=("unique_id", "nunique"), dev1=("m1", lambda x: np.nan)).head(40).to_string())
    print("m2 coverage", t.m2.notna().mean())


if __name__ == "__main__":
    main()
