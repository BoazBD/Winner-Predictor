"""s06 - two last attempts to rescue the anchor idea.

(A) Maker entry with strict fills: at T post a buy for outcome k at the bid b = p_T - hs (rounded down to the cent) when
    gap = bfe - poly_T >= theta. Filled only if a real taker SELL of Yes (= taker buy of No at q, Yes price 1-q)
    prints STRICTLY below b (1-q < b - 5e-5) within [T, T+6h] (and before kick-30). No fee for makers.
    Exit (i) taker at bid p_X - hs with fee; (ii) maker: post ask p_X + hs... approximated as: sell at ask a = mid+hs at
    kick-75 if a real taker BUY of Yes prints strictly above a in [kick-75, kick-5], else taker at the last bid with fee.
(B) Closing-line check (game risk, NOT a pre-kick trade): buy at kick-~15 min when Betfair closing (bfec) exceeds the
    Polymarket close by >= theta; hold to the result. Tells whether a live Betfair feed at kickoff has value.
theta chosen on train from {0.01,0.02,0.03,0.05}. Output out/maker_close.json.
"""
import os, json
import numpy as np, pandas as pd, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
MASTER = os.path.join(R, "edge", "data", "master.parquet")
rng = np.random.default_rng(2)
FEE = 0.05


def split(k):
    k = pd.to_datetime(k)
    return np.where(k < "2026-07-01", "train", np.where(k < "2026-10-01", "val", np.where(k < "2026-11-19", "test", "live")))


def boot(t, n=2000):
    g = t.groupby("unique_id")[["pnl", "cost"]].sum()
    a, b = g.pnl.to_numpy(), g.cost.to_numpy()
    if not len(a):
        return [np.nan] * 3
    idx = rng.integers(0, len(a), (n, len(a)))
    m = a[idx].sum(1) / b[idx].sum(1)
    return [round(float(a.sum() / b.sum()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    P = pd.read_parquet(os.path.join(HERE, "data", "panel.parquet"))
    for c in ["p_T", "p_X", "p_C", "psum_T", "psum_C"]:
        P[c] = P[c].astype(np.float64).round(4)
    P["split"] = split(P.kick)
    P["hs"] = np.where(P.ev_volume >= 1e4, 0.005, 0.015)
    P = P[P.split != "live"].copy()
    res = {}
    # ---------- (A) maker entry
    A = P[P.bfe.notna() & P.p_T.notna() & P.p_X.notna() & (P.h_T_to_kick >= 1) & (P.mtk_X <= 120)].copy()
    A["gap"] = A.bfe - A.p_T / A.psum_T
    A = A[A.gap >= 0.01 - 5e-5].copy()
    A["bid"] = np.floor((A.p_T - A.hs) * 100 + 1e-6) / 100
    A["ask_X"] = np.ceil((A.p_X + A.hs) * 100 - 1e-6) / 100
    A["coll_min"] = A["T"].values.astype("datetime64[m]").astype(np.int64)
    A["kick_min"] = A.kick.values.astype("datetime64[m]").astype(np.int64)
    con = duckdb.connect(); con.register("A", A[["unique_id", "k", "bid", "ask_X", "coll_min", "kick_min"]])
    fill = con.execute(f"""select A.unique_id, A.k, min(m.ts//60) fill_min from '{MASTER}' m join A on m.unique_id=A.unique_id and m.k=A.k
        where not m.yes and (1-m.q) < A.bid - 0.00005 and m.ts//60 >= A.coll_min and m.ts//60 <= least(A.coll_min+360, A.kick_min-30)
        group by 1,2""").df()
    sell = con.execute(f"""select A.unique_id, A.k, count(*) n_through from '{MASTER}' m join A on m.unique_id=A.unique_id and m.k=A.k
        where m.yes and m.q > A.ask_X + 0.00005 and m.rel_min between -75 and -5 group by 1,2""").df()
    A = A.merge(fill, on=["unique_id", "k"], how="left").merge(sell, on=["unique_id", "k"], how="left")
    A["filled"] = A.fill_min.notna()
    print("maker entry fill rate", A.filled.mean().round(3))
    grid = []
    for th in [0.01, 0.02, 0.03, 0.05]:
        for ex in ["taker_exit", "maker_exit"]:
            t = A[A.filled & (A.gap >= th - 5e-5)].copy()
            t["cost"] = t.bid
            bidX = (t.p_X - t.hs).clip(0, .99)
            taker = bidX - FEE * bidX * (1 - bidX)
            t["proceeds"] = taker if ex == "taker_exit" else np.where(t.n_through.fillna(0) > 0, t.ask_X, taker)
            t["pnl"] = t.proceeds - t.cost
            # adverse-selection diagnostic: unfilled orders' drift
            for sp in ["train", "val", "test"]:
                s = t[t.split == sp]
                grid.append(dict(theta=th, exit=ex, split=sp, n=len(s), ret=boot(s), mean_drift_pts=round(float((s.p_X - s.p_T).mean() * 100), 2)))
    G = pd.DataFrame(grid)
    print(G.to_string())
    res["maker_entry"] = dict(fill_rate=float(A.filled.mean()), unfilled_mean_drift_pts=float((A[~A.filled].p_X - A[~A.filled].p_T).mean() * 100),
                              filled_mean_drift_pts=float((A[A.filled].p_X - A[A.filled].p_T).mean() * 100), grid=grid)
    # ---------- (B) closing-line hold
    C = P[P.bfec.notna() & P.p_C.notna()].copy()
    C["pn_C"] = C.p_C / C.psum_C
    C["gap"] = C.bfec - C.pn_C
    out = []
    for th in [0.01, 0.02, 0.03, 0.05]:
        for fee_on in [True, False]:
            t = C[C.gap >= th - 5e-5].copy()
            ask = (t.p_C + t.hs).clip(0.01, .99)
            t["cost"] = ask + (FEE * ask * (1 - ask) if fee_on else 0)
            t["pnl"] = t.won - t.cost
            for sp in ["train", "val", "test"]:
                s = t[t.split == sp]
                out.append(dict(theta=th, fee=fee_on, split=sp, n=len(s), ret=boot(s)))
    Cg = pd.DataFrame(out); print(Cg.to_string())
    res["close_hold"] = out
    # zero-cost "edge" of the close gap: mean(won - pn_C) for gap>=0.02
    for sp in ["train", "val", "test"]:
        s = C[(C.gap >= 0.02) & (C.split == sp)]
        print(sp, "close gap>=2pt: n", len(s), "mean won - poly_close (pts)", round(float((s.won - s.pn_C).mean() * 100), 2), "mean gap", round(float(s.gap.mean() * 100), 2))
    json.dump(res, open(os.path.join(HERE, "out", "maker_close.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
