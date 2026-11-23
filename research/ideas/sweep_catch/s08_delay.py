"""s08: robustness of the final rules to (a) the bot's REACTION DELAY before the exit sell is posted, and
(b) concentration of profit in a few events.

(a) In s02 the exit limit sell counts from the fill second. A real bot only learns of the fill at its next run, so we
recompute the 'tmid_vk' exit requiring the exit print to come after fill + delay, delay in {60 s, 5 min, until the next
hourly run (R=60 rules), 60 min}. If the target is not hit, the same verified liquidation before kick is used.
(b) share of P&L from the top 1/5/10 fills and top 3 games, and results with the single best game removed.
Output: out/delay.csv, out/concentration.csv
"""
import os
import numpy as np, pandas as pd, duckdb
from s02_sim import D, LL, fee, floor_c
from s03_grid import OUT

TOL = 5e-5


def main():
    k = pd.read_parquet(os.path.join(OUT, "final_fills.parquet"))
    k = k[k.per != "after"]
    uids = list(k.uid.unique())
    con = duckdb.connect(config=dict(threads=2))
    con.register("u", pd.DataFrame(dict(uid=uids)))
    tp = con.sql(f"select unique_id uid, k, ts, ypx, mlong from '{os.path.join(D, 'tape.parquet')}' t "
                 f"join u on t.unique_id = u.uid").df()
    by = {key: g for key, g in tp.groupby(["uid", "k"])}
    out = []
    for r in k.itertuples():
        g = by.get((r.uid, r.k))
        ext = g[g.mlong == (r.s < 0)]; ent = g[g.mlong == (r.s > 0)]
        xx = np.round(ext.ypx.to_numpy() if r.s > 0 else 1 - ext.ypx.to_numpy(), 4); tx = ext.ts.to_numpy()
        xe = np.round(ent.ypx.to_numpy() if r.s > 0 else 1 - ent.ypx.to_numpy(), 4); te = ent.ts.to_numpy()
        T = round(float(max(r.L + 0.01, floor_c(r.c))), 4)
        R = 60 if "R60" in r.rule else 1
        next_run = (int(r.ts) // (R * 60) + 1) * R * 60 - int(r.ts)
        kick = int(r.ts) + int(r.mtk) * 60          # minute resolution, as in s02
        # verified liquidation (same as s02 'vk'): last bid-hit print in [kick-120m, kick-5m] after the fill, 1c worse
        vw = (te > r.ts) & (te >= kick - 7200) & (te <= kick - 300)
        if vw.any():
            vb = max(float(xe[vw][-1]) - 0.01, 0.001); vk = vb - fee(vb) - r.L
        else:
            vk = r.won - r.L
        row = dict(rule=r.rule, per=r.per, uid=r.uid, q=r.q, stake=r.stake, base=r.tmid_vk)
        for lab, dl in (("d0", 0), ("d60s", 60), ("d5m", 300), ("next_run", next_run), ("d60m", 3600)):
            hit = (tx > r.ts + dl) & (tx <= kick - 300) & (xx > T + TOL)
            row[lab] = (T - r.L) if hit.any() else vk
        out.append(row)
    o = pd.DataFrame(out)
    o.to_csv(os.path.join(OUT, "delay_raw.csv"), index=False)
    res = []
    for (rule, per), z in o.groupby(["rule", "per"]):
        for lab in ("base", "d0", "d60s", "d5m", "next_run", "d60m"):
            v = z[lab]
            res.append(dict(rule=rule, per=per, delay=lab, fills=len(z),
                            ret_per_usd=float((v * z.q).sum() / z.stake.sum())))
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(OUT, "delay.csv"), index=False)
    pd.set_option("display.width", 200)
    print(r.pivot_table(index=["rule", "per"], columns="delay", values="ret_per_usd").round(3).to_string())
    # concentration
    cr = []
    for (rule, per), z in k.groupby(["rule", "per"]):
        if not str(rule).startswith(("A_", "B_", "C_", "H_", "I_")):
            continue
        p = z.pnl.sort_values(ascending=False)
        gp = z.groupby("uid").pnl.sum().sort_values(ascending=False)
        cr.append(dict(rule=rule, per=per, fills=len(z), total=p.sum(), top1=p.iloc[0] / p.sum(),
                       top5=p.head(5).sum() / p.sum(), top3_games=gp.head(3).sum() / p.sum(),
                       ret_without_best_game=float(z[z.uid != gp.index[0]].pnl.sum() / z[z.uid != gp.index[0]].stake.sum())))
    c = pd.DataFrame(cr)
    c.to_csv(os.path.join(OUT, "concentration.csv"), index=False)
    print(c.round(3).to_string())


if __name__ == "__main__":
    main()
