"""Step 4: relative-value backtests of side markets against the 1X2-implied fair value (real prints, fees, game-bootstrap CIs).

Fair value: m1 (DC from 1X2) or m2 (DC from 1X2 + O/U 2.5 reference), each recalibrated on TRAIN ONLY with a per-market-key
logistic map  P(A) = sigmoid(a_k + b_k * logit(m))  (k = market key, e.g. 'totals|3.5|', 'spreads|-1.5|home').
Token X fair = f for A, 1-f for B.  Fee (taker, fee-enabled markets) = 0.05*q*(1-q) per share.

Strategies (each: at most one entry per game x market key x token; $100 per entry; size capped by the real print size):
 T  taker-hold : a real print shows a taker bought X at q; if fair_X - q - fee(q) > theta -> we buy X at q, hold to settlement.
 C  taker-flip : same entry; exit = the last real print before kickoff where a taker SOLD X at bid b (> entry time);
                 pnl = b - q - fee(q) - fee(b). No exit print -> held to settlement (reported as 'held' share).
 M  maker-hold : at every 15-min 1X2 grid point, a resting bid for X at floor_cent(fair_X - theta); filled only if within the
                 next 15 min a real print shows a taker SELLING X at a price STRICTLY below our bid (fill at our bid, no fee).
Grid tried: strategy {T,C,M} x fair {m1,m2} x theta {0.01,0.02,0.03,0.05,0.08} x window(min to kick) {(0,4320),(0,360),(0,60)}
            x markets {core, all} = 180 combos. Choice on TRAIN (< 2026-07-01) by return/$ with >= 150 entries.
Periods by kickoff: train < 2026-07-01, valid Jul-Sep 2026, test Oct 1 - Nov 18 2026.
Outputs: out/rv_grid.csv (all combos x periods), out/rv_best.json, out/calibration.csv
"""
import os, json, itertools, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CORE = ["totals|1.5|", "totals|2.5|", "totals|3.5|", "both_teams_to_score|0.0|", "spreads|-1.5|home", "spreads|-1.5|away",
        "soccer_team_totals|0.5|home", "soccer_team_totals|0.5|away", "soccer_team_totals|1.5|home", "soccer_team_totals|1.5|away"]
FEE = lambda q: 0.05 * q * (1 - q)
RNG = np.random.default_rng(0)
ENT = {}
KEEP = ["unique_id", "mkey", "xA", "ts", "mtk", "per", "cost", "pnl", "held", "px", "fees"]


def period(k):
    k = pd.to_datetime(k)
    return np.where(k < "2026-07-01", "train", np.where(k < "2026-10-01", "valid", np.where(k < "2026-11-19", "test", "live")))


def lg(p):
    p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))


def calibrate(p):
    """Per market key logistic recalibration of m1/m2 fitted on train last-print-per-game rows (one row per game/market)."""
    from sklearn.linear_model import LogisticRegression
    rows = []
    for m in ("m1", "m2"):
        p[m + "c"] = np.nan
        tr = p[(p.per == "train") & p[m].notna()].groupby(["unique_id", "mkey"]).tail(1)
        for k, d in p[p[m].notna()].groupby("mkey"):
            dt = tr[tr.mkey == k]
            if len(dt) >= 60 and dt.resA.nunique() == 2:
                lr = LogisticRegression(C=10.0).fit(lg(dt[m].to_numpy())[:, None], dt.resA.astype(int))
                a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
            else:
                a, b = 0.0, 1.0
            p.loc[d.index, m + "c"] = 1 / (1 + np.exp(-(a + b * lg(d[m].to_numpy()))))
            rows.append(dict(model=m, mkey=k, a=a, b=b, n_train=len(dt)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "calibration.csv"), index=False)
    return p


def summarize(e, label):
    """e: entries with unique_id, per, cost (=$ invested), pnl ($), day."""
    out = []
    for per, d in e.groupby("per"):
        g = d.groupby("unique_id").agg(c=("cost", "sum"), p=("pnl", "sum"))
        r = g.p.sum() / g.c.sum()
        bs = []
        C, P = g.c.to_numpy(), g.p.to_numpy()
        for _ in range(400):
            ix = RNG.integers(0, len(g), len(g)); bs.append(P[ix].sum() / C[ix].sum())
        months = max(1.0, (d.ts.max() - d.ts.min()) / (30 * 86400))
        daily = d.groupby("day").pnl.sum()
        out.append(dict(label=label, per=per, n=len(d), games=len(g), ret=r, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975),
                        per_month=len(d) / months, usd_month=d.pnl.sum() / months * (100 / d.cost.mean()) if len(d) else 0,
                        worst_day=daily.min(), held_share=d.held.mean() if "held" in d else np.nan))
    return out


def main():
    p = pd.read_parquet(os.path.join(HERE, "data", "panel.parquet"))
    p = p[p.resA.isin([0, 1])].copy()
    p["kick_min"] = pd.to_datetime(p.kick_utc).astype("datetime64[s]").astype("int64") // 60
    p["mtk"] = p.kick_min - p.minute
    p["per"] = period(p.kick_utc)
    p = p[p.per != "live"]
    p = p.sort_values(["unique_id", "mkey", "ts"]).reset_index(drop=True)
    p = calibrate(p)
    p["day"] = pd.to_datetime(p.ts, unit="s").dt.floor("D")
    p.to_parquet(os.path.join(HERE, "data", "panel_cal.parquet"))
    print("panel", len(p), p.groupby("per").unique_id.nunique().to_dict(), flush=True)
    # token-X view of every print: X = token the taker bought
    p["qx"] = np.where(p.buyA, p.pA, 1 - p.pA).round(4)
    p["xA"] = p.buyA
    p["won_x"] = np.where(p.xA, p.resA, 1 - p.resA)
    feeq = np.where(p.fees, FEE(p.qx), 0.0)
    sells = p[["unique_id", "mkey", "ts", "pA", "buyA"]].copy()
    sells = sells.assign(xA=~sells.buyA, bx=np.where(~sells.buyA, sells.pA, 1 - sells.pA).round(4))  # taker sold X == bought other
    last = sells.groupby(["unique_id", "mkey", "xA"]).agg(ts_exit=("ts", "max"))
    lastp = sells.merge(last.reset_index(), on=["unique_id", "mkey", "xA"])
    lastp = lastp[lastp.ts == lastp.ts_exit].groupby(["unique_id", "mkey", "xA"]).bx.min().rename("b").reset_index()
    lastp = lastp.merge(last.reset_index(), on=["unique_id", "mkey", "xA"])
    res = []
    for fair, theta, win, mk in itertools.product(["m1c", "m2c"], [0.01, 0.02, 0.03, 0.05, 0.08], [(0, 4320), (0, 360), (0, 60)], ["core", "all"]):
        base = p[p[fair].notna() & (p.mtk > win[0]) & (p.mtk <= win[1]) & ((mk == "all") | p.mkey.isin(CORE))]
        fx = np.where(base.xA, base[fair], 1 - base[fair])
        fq = np.where(base.fees, FEE(base.qx), 0.0)
        sig = base[(fx - base.qx - fq > theta) & (base.qx > 0.02) & (base.qx < 0.98)]
        ent = sig.groupby(["unique_id", "mkey", "xA"]).head(1).copy()
        if len(ent) == 0:
            continue
        sh = np.minimum(ent["size"], 100 / ent.qx)
        ent["cost"] = sh * ent.qx
        fee_e = np.where(ent.fees, FEE(ent.qx), 0.0)
        # T: hold
        eT = ent.assign(pnl=sh * (ent.won_x - ent.qx - fee_e), held=1.0, px=ent.qx)
        ENT[("T", fair, theta, str(win), mk)] = eT[KEEP]
        res += [dict(strategy="T", fair=fair, theta=theta, win=str(win), mk=mk, **r) for r in summarize(eT, "T")]
        # C: exit at last taker-sell print of X before kickoff after entry
        eC = ent.merge(lastp, on=["unique_id", "mkey", "xA"], how="left")
        ok = eC.ts_exit > eC.ts
        shC = np.minimum(eC["size"], 100 / eC.qx)
        feeb = np.where(eC.fees, FEE(eC.b.fillna(0)), 0.0)
        fee_c = np.where(eC.fees, FEE(eC.qx), 0.0)
        eC["pnl"] = np.where(ok, shC * (eC.b - eC.qx - fee_c - feeb), shC * (eC.won_x - eC.qx - fee_c))
        eC["held"] = (~ok).astype(float)
        eC["px"] = eC.qx
        ENT[("C", fair, theta, str(win), mk)] = eC[KEEP]
        res += [dict(strategy="C", fair=fair, theta=theta, win=str(win), mk=mk, **r) for r in summarize(eC, "C")]
    # M: maker bids on the 15-min grid
    res += maker(p, CORE)
    R = pd.DataFrame(res)
    # trades are a random sample of games: scale counts/$ to the full universe of games with a -more-markets event
    sm = pd.read_parquet(os.path.join(HERE, "data", "side_markets.parquet"), columns=["unique_id", "side_slug", "kick_utc", "vol"])
    sm = sm[sm.side_slug.str.endswith("-more-markets")].groupby("unique_id").agg(k=("kick_utc", "first"), v=("vol", "sum"))
    sm = sm[sm.v > 0]
    full = pd.Series(period(sm.k)).value_counts()
    samp = p.groupby("per").unique_id.nunique()
    R["sample_frac"] = R.per.map(samp / full)
    R["per_month_full"] = R.per_month / R.sample_frac
    R["usd_month_full"] = R.usd_month / R.sample_frac
    print("sample fraction", (samp / full).to_dict())
    R.to_csv(os.path.join(OUT, "rv_grid.csv"), index=False)
    tr = R[(R.per == "train") & (R.n >= 150)].sort_values("ret", ascending=False)
    print("combos tried:", R.groupby(["strategy", "fair", "theta", "win", "mk"]).ngroups)
    print(tr.head(15).to_string())
    best = {}
    for s in ["T", "C", "M"]:
        b = tr[tr.strategy == s].head(1)
        if len(b):
            key = b.iloc[0][["strategy", "fair", "theta", "win", "mk"]].to_dict()
            sel = R[(R.strategy == s) & (R.fair == key["fair"]) & (R.theta == key["theta"]) & (R.win == key["win"]) & (R.mk == key["mk"])]
            best[s] = dict(params={k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in key.items()},
                           rows=json.loads(sel.to_json(orient="records")))
            print(s, key); print(sel.to_string())
    be = [ENT[tuple(b["params"][k] for k in ["strategy", "fair", "theta", "win", "mk"])].assign(strategy=s) for s, b in best.items()]
    if be:
        pd.concat(be).to_parquet(os.path.join(OUT, "rv_best_entries.parquet"))
    json.dump(best, open(os.path.join(OUT, "rv_best.json"), "w"), indent=1, default=str)


def maker(p, core):
    """Resting bids at floor_cent(fair_X - theta) placed at each 15-min grid point; filled by strictly-through taker sells."""
    out = []
    # grid points: distinct (unique_id, minute-bucket) where fair is known; use prints themselves as the timeline:
    # we place/refresh the bid every 15 min using the latest fair value known at that time (last print's 1X2 fair).
    p = p.copy()
    p["slot"] = p.minute // 15
    for fair, theta, win, mk in itertools.product(["m1c", "m2c"], [0.01, 0.02, 0.03, 0.05, 0.08], [(0, 4320), (0, 360), (0, 60)], ["core", "all"]):
        b = p[p[fair].notna() & (p.mtk > win[0]) & (p.mtk <= win[1]) & ((mk == "all") | p.mkey.isin(core))]
        if not len(b):
            continue
        # fair at start of slot = fair of first print in the slot (computed from the 1X2 grid at/before that minute)
        f0 = b.groupby(["unique_id", "mkey", "slot"])[fair].transform("first")
        # taker SELL of A (buyA False) at pA < bidA fills our bid on A; taker BUY of A at pA > 1-bidB  fills our bid on B
        bidA = np.floor((f0 - theta) * 100 + 1e-9) / 100
        bidB = np.floor(((1 - f0) - theta) * 100 + 1e-9) / 100
        fillA = (~b.buyA) & (b.pA < bidA - 5e-5) & (bidA > 0.01)
        fillB = (b.buyA) & ((1 - b.pA) < bidB - 5e-5) & (bidB > 0.01)
        fa = b[fillA].assign(px=bidA[fillA], xA=True)
        fb = b[fillB].assign(px=bidB[fillB], xA=False)
        f = pd.concat([fa, fb]).sort_values("ts").groupby(["unique_id", "mkey", "xA"]).head(1)
        if not len(f):
            continue
        shares = np.minimum(f["size"], 100 / f.px)
        won = np.where(f.xA, f.resA, 1 - f.resA)
        f = f.assign(cost=shares * f.px, pnl=shares * (won - f.px), held=1.0)
        ENT[("M", fair, theta, str(win), mk)] = f[KEEP]
        out += [dict(strategy="M", fair=fair, theta=theta, win=str(win), mk=mk, **r) for r in summarize(f, "M")]
    return out


if __name__ == "__main__":
    main()
