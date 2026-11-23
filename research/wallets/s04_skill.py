"""s04: do wallets have PERSISTENT skill at predicting the pre-kickoff PRICE MOVE? (research questions 1, 3, 5)

Unit = one wallet in one game in one role (taker rows / maker rows), usd-weighted averages of:
  mv_c (move from the mid just after its trades to the close ~14 min pre-kick, in its direction), mv_1h, mv_6h,
  clv (close vs its own price), res (result vs its own price).
Walk-forward: for each kickoff month M, a wallet's score uses ONLY units of games that kicked off before M starts.
Scores (all shrunk to 0 with K pseudo-units): move = sum(mv_c)/(n+K); clv; res; tmove = t-stat of mv_c.
Then, in month M, units of scored wallets are bucketed by score and we measure their forward move.
Outputs: out/s04_persistence.csv, out/s04_corr.csv, out/s04_population.json, out/fig_s04_persistence.png
"""
import json, os
import numpy as np, pandas as pd
from common import D, OUT, period, boot_ci

K = 20
MINN = 20


def units(t):
    t = t[t.m0.notna()].copy()
    t["w"] = t.usd.astype(float)
    for c in ["mv_c", "mv_1h", "mv_6h", "clv", "res"]:
        t[c + "_w"] = t[c].astype(float) * t.w
    u = t.groupby(["wid", "unique_id", "taker"]).agg(
        w=("w", "sum"), n=("w", "size"), kick_ts=("kick_ts", "first"), ts=("ts", "min"),
        **{c: (c + "_w", "sum") for c in ["mv_c", "mv_1h", "mv_6h", "clv", "res"]}).reset_index()
    for c in ["mv_c", "mv_1h", "mv_6h", "clv", "res"]:
        u[c] = u[c] / u.w
    u["month"] = pd.to_datetime(u.kick_ts, unit="s").dt.to_period("M").dt.to_timestamp()
    return u


def walk_forward_scores(u, months, role):
    """Return DataFrame (month, wid, n, move, clv, res, tmove) using units with kick < month start."""
    uu = u[u.taker == role]
    out = []
    for m in months:
        h = uu[uu.kick_ts < m.timestamp()]
        if not len(h):
            continue
        a = h.groupby("wid").agg(n=("mv_c", "size"), s_mv=("mv_c", "sum"), s_mv2=("mv_c", lambda x: (x ** 2).sum()),
                                 s_clv=("clv", "sum"), s_res=("res", "sum"), usd=("w", "sum"))
        a["move"] = a.s_mv / (a.n + K)
        a["clv"] = a.s_clv / (a.n + K)
        a["res"] = a.s_res / (a.n + K)
        mu = a.s_mv / a.n
        sd = np.sqrt(np.maximum(a.s_mv2 / a.n - mu ** 2, 1e-6))
        a["tmove"] = mu / sd * np.sqrt(a.n)
        a["month"] = m
        out.append(a.reset_index()[["month", "wid", "n", "usd", "move", "clv", "res", "tmove"]])
    return pd.concat(out, ignore_index=True)


def main():
    t = pd.read_parquet(os.path.join(D, "trades_marked.parquet"),
                        columns=["wid", "unique_id", "taker", "usd", "ts", "kick_ts", "m0", "mv_c", "mv_1h", "mv_6h",
                                 "clv", "res"])
    u = units(t)
    u["period"] = period(u.kick_ts)
    u.to_parquet(os.path.join(D, "units.parquet"))
    months = pd.date_range("2025-11-01", "2026-11-01", freq="MS")
    pop = {}
    for role, name in [(True, "taker"), (False, "maker")]:
        r = u[u.taker == role]
        pop[name] = dict(units=int(len(r)), wallets=int(r.wid.nunique()),
                         wallets_ge20=int((r.groupby("wid").size() >= 20).sum()),
                         mean_mv_c=float(np.average(r.mv_c, weights=r.w)), mean_clv=float(np.average(r.clv, weights=r.w)))
    json.dump(pop, open(os.path.join(OUT, "s04_population.json"), "w"), indent=1)
    print(pop, flush=True)
    rows, corr = [], []
    for role, rname in [(True, "taker"), (False, "maker")]:
        sc = walk_forward_scores(u, months, role)
        sc.to_parquet(os.path.join(D, f"scores_{rname}.parquet"))
        ev = u[(u.taker == role)].merge(sc, on=["month", "wid"], how="inner", suffixes=("", "_s"))
        ev = ev[ev.n_s >= MINN] if "n_s" in ev else ev
        ev = ev.rename(columns={"n_s": "nhist"}) if "n_s" in ev else ev
        ev = ev[ev["n_y"] >= MINN] if "n_y" in ev else ev
        for score in ["move", "clv", "res", "tmove"]:
            col = score + "_s" if score + "_s" in ev else score
            # month-wise ranks among eligible wallets
            ev["rk"] = ev.groupby("month")[col].rank(pct=True)
            bins = [0, 0.01, 0.05, 0.2, 0.8, 0.95, 0.99, 1.0001]
            labs = ["bot1%", "bot1-5%", "bot5-20%", "mid", "top5-20%", "top1-5%", "top1%"]
            ev["b"] = pd.cut(ev.rk, bins, labels=labs, include_lowest=True)
            for per in ["train", "val", "test"]:
                e = ev[ev.period == per]
                for b in labs:
                    x = e[e.b == b]
                    if len(x) < 30:
                        continue
                    for tgt in ["mv_c", "mv_1h", "res"]:
                        m, lo, hi = boot_ci(x[tgt], x.unique_id, n=300)
                        rows.append(dict(role=rname, score=score, period=per, bucket=b, target=tgt, units=len(x),
                                         wallets=x.wid.nunique(), games=x.unique_id.nunique(), mean=m, lo=lo, hi=hi))
            # wallet-level persistence: score at val start (train-only) vs realised mean in val/test
            for per, m0 in [("val", pd.Timestamp("2026-07-01")), ("test", pd.Timestamp("2026-10-01"))]:
                s0 = sc[(sc.month == m0) & (sc.n >= MINN)].set_index("wid")[score].rename("score")
                rr = u[(u.taker == role) & (u.period == per)].groupby("wid").agg(n=("mv_c", "size"), mv=("mv_c", "mean"),
                                                                                  res=("res", "mean"))
                rr = rr[rr.n >= 10].join(s0, how="inner")
                rr = rr.rename(columns={"score": "sc"})
                if len(rr) > 20:
                    corr.append(dict(role=rname, score=score, period=per, wallets=len(rr),
                                     spearman_mv=rr.sc.corr(rr.mv, method="spearman"),
                                     spearman_res=rr.sc.corr(rr.res, method="spearman")))
        print(rname, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "s04_persistence.csv"), index=False)
    pd.DataFrame(corr).to_csv(os.path.join(OUT, "s04_corr.csv"), index=False)
    print(pd.DataFrame(corr).to_string())
    with pd.option_context("display.width", 250):
        print(res[(res.target == "mv_c")].pivot_table(index=["role", "score", "bucket"], columns="period",
                                                      values="mean").round(4).to_string())


if __name__ == "__main__":
    main()
