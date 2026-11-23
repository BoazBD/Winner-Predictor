"""Step 8: can the chosen relative-value entries be hedged with the liquid markets (1X2 + O/U 2.5)?

For every entry of the chosen rules (out/rv_best_entries.parquet) we take the DC score distribution at entry
(1X2 mids + O/U 2.5 reference -> grid row) and find the min-variance hedge h on [home-win Yes, away-win Yes, Over 2.5]
(weighted least squares of the token-X payoff over 121 scorelines). Hedge legs are bought/sold at the mid plus an
assumed half-spread of 0.5c plus the taker fee 0.05 p(1-p) per share (negative h = buy the complement token).
Reports: model variance reduction, worst-case payoff of the hedged package over all scorelines (per $1 of X),
realized hedged vs unhedged return per $ by period with game-bootstrap CIs.
Usage: python s08_hedge.py [entries parquet in out/]  -> out/hedge_summary_<name>.csv
"""
import os, sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Table, G
from s03_panel import payoff, lg

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FEE = lambda q: 0.05 * q * (1 - q)
I = np.arange(G); HG = np.repeat(I, G).reshape(G, G); AG = HG.T
RNG = np.random.default_rng(1)


def ci(g):
    C, P = g.c.to_numpy(), g.p.to_numpy(); bs = []
    for _ in range(400):
        ix = RNG.integers(0, len(g), len(g)); bs.append(P[ix].sum() / C[ix].sum())
    return np.quantile(bs, [.025, .975])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "rv_best_entries.parquet"
    e = pd.read_parquet(os.path.join(OUT, src))
    if "strategy" not in e:
        e["strategy"] = "M_family"
    e = e[e.strategy.isin(["T", "M", "M_family"])]          # hold-to-settlement rules only (C mostly exits before kickoff)
    p = pd.read_parquet(os.path.join(HERE, "data", "panel_cal.parquet"), columns=["unique_id", "mkey", "ts", "ph", "pa", "o25", "mslug", "smt", "line"])
    p = p.drop_duplicates(["unique_id", "mkey", "ts"])
    e = e.merge(p, on=["unique_id", "mkey", "ts"], how="left")
    e = e[e.o25.notna()].copy()
    sm = pd.read_parquet(os.path.join(HERE, "data", "side_markets.parquet"), columns=["unique_id", "mslug", "resA"])
    o25 = sm[sm.mslug.str.endswith("-total-2pt5")].groupby("unique_id").resA.first().rename("over25")
    gm = pd.read_parquet(os.path.join(ROOT, "lead_lag", "data", "games_meta.parquet"), columns=["unique_id", "outcome_price0", "outcome_price2"])
    gm["homewin"] = gm.outcome_price0.astype(float); gm["awaywin"] = gm.outcome_price2.astype(float)   # poly orientation tok0=home
    e = e.merge(o25, on="unique_id", how="left").merge(gm[["unique_id", "homewin", "awaywin"]], on="unique_id", how="left")
    e = e[e.over25.isin([0, 1]) & e.homewin.isin([0, 1])].copy()
    tab = Table(); Mf = tab.M.reshape(len(tab.M), -1)
    tree = cKDTree(np.c_[lg(tab.pH), lg(tab.pA), lg(Mf @ (HG + AG > 2.5).ravel().astype(np.float32))])
    ix = tree.query(np.c_[lg(e.ph), lg(e.pa), lg(e.o25)])[1]
    B = np.c_[(HG > AG).ravel(), (HG < AG).ravel(), (HG + AG > 2.5).ravel()].astype(float)
    rows = []
    for n, (i, r) in enumerate(zip(ix, e.itertuples())):
        w = Mf[i].astype(float)
        P = payoff(r.smt, r.line, r.mslug).ravel().astype(float)
        X = P if r.xA else 1 - P
        mu = w @ X
        Bc = B - w @ B; Xc = X - mu
        h = np.linalg.lstsq(Bc * np.sqrt(w)[:, None], Xc * np.sqrt(w), rcond=None)[0]   # payoff X ~ const + B h
        resid = Xc - Bc @ h
        prices = np.array([r.ph, r.pa, r.o25])
        hcost = np.abs(h) @ (0.005 + FEE(prices))
        rows.append(dict(i=r.Index, var_red=1 - (w @ resid ** 2) / max(w @ Xc ** 2, 1e-12), h0=h[0], h1=h[1], h2=h[2], hcost=hcost,
                         minpay=(X - B @ h).min(), hdot=prices @ h, hedge_pnl_ps=-(np.array([r.homewin, r.awaywin, r.over25]) - prices) @ h - hcost))
    H = pd.DataFrame(rows).set_index("i")
    e = e.join(H)
    # entry price q of token X = price of the entry print
    e["q"] = e.px          # our entry price (taker: print price; maker: our resting bid)
    e["shares"] = e.cost / e.q
    e["worst"] = e.minpay - e.q + e.hdot - e.hcost - np.where(e.fees & (e.strategy == "T"), FEE(e.q), 0)   # worst-case $ per share of X, hedged package
    e["pnl_h"] = e.pnl + e.shares * e.hedge_pnl_ps
    out = []
    for (s, per), d in e.groupby(["strategy", "per"]):
        g = d.groupby("unique_id").agg(c=("cost", "sum"), p=("pnl", "sum"), ph=("pnl_h", "sum"))
        lo, hi = ci(g.rename(columns={})[["c", "p"]])
        lo2, hi2 = ci(g[["c", "ph"]].rename(columns={"ph": "p"}))
        out.append(dict(strategy=s, per=per, n=len(d), var_reduction_med=d.var_red.median(), hedge_cost_per_share_med=d.hcost.median(),
                        worst_case_per_share_med=d.worst.median(), ret_unhedged=g.p.sum() / g.c.sum(), lo=lo, hi=hi,
                        ret_hedged=g.ph.sum() / g.c.sum(), lo_h=lo2, hi_h=hi2,
                        sd_unhedged_per_trade=(d.pnl / d.cost).std(), sd_hedged_per_trade=(d.pnl_h / d.cost).std()))
    O = pd.DataFrame(out); O.to_csv(os.path.join(OUT, "hedge_summary_" + src.replace(".parquet", ".csv")), index=False); print(O.to_string())


if __name__ == "__main__":
    main()
