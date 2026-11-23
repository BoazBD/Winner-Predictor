"""Step 9 (exploratory, NOT the pre-registered selection): the maker-vs-model family M / m2c / theta 0.02-0.05.

In s04 the train-selected rules failed out of sample, but the maker family with the 1X2+O/U2.5 fair value was mildly
positive in all three periods for theta 0.02-0.05. This script looks at it more closely so the report can say whether
it is worth paper trading: per period (CI), per market key, per month, fills per month scaled to all games.
Output: out/maker_family.csv, out/maker_family_by_mkey.csv, out/maker_family_monthly.csv
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s04_rv_backtest as s4

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")


def main():
    p = pd.read_parquet(os.path.join(HERE, "data", "panel_cal.parquet"))
    s4.ENT.clear()
    s4.maker(p, s4.CORE)
    rows, bym, mon = [], [], []
    for key, e in s4.ENT.items():
        _, fair, theta, win, mk = key
        if fair != "m2c" or theta > 0.05:
            continue
        for r in s4.summarize(e.assign(day=pd.to_datetime(e.ts, unit="s").dt.floor("D")), "M"):
            rows.append(dict(theta=theta, win=win, mk=mk, **r))
        if theta == 0.05 and win == "(0, 4320)" and mk == "all":
            for (k, per), d in e.groupby(["mkey", "per"]):
                bym.append(dict(mkey=k, per=per, n=len(d), ret=d.pnl.sum() / d.cost.sum()))
            e = e.assign(m=pd.to_datetime(e.ts, unit="s").dt.to_period("M").astype(str))
            mon = e.groupby("m").agg(n=("pnl", "size"), ret=("pnl", lambda x: x.sum() / e.loc[x.index, "cost"].sum())).reset_index()
            e.to_parquet(os.path.join(OUT, "maker_family_entries.parquet"))
    R = pd.DataFrame(rows).sort_values(["theta", "win", "mk", "per"])
    R.to_csv(os.path.join(OUT, "maker_family.csv"), index=False); print(R.round(4).to_string())
    B = pd.DataFrame(bym).pivot_table(index="mkey", columns="per", values=["n", "ret"]).round(3)
    B.to_csv(os.path.join(OUT, "maker_family_by_mkey.csv")); print(B.to_string())
    mon.to_csv(os.path.join(OUT, "maker_family_monthly.csv"), index=False); print(mon.round(3).to_string())


if __name__ == "__main__":
    main()
