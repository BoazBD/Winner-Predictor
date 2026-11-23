"""s06: informed vs uninformed sweeps. Can the sweeper's wallet history (known at the fill time) tell the losing fills
(price never comes back) from the winning ones?

Fills: mid family, R=1, d in {0.10,0.12,0.15,0.20}, Winner gap >= 6c, band 8-92c (the selected neighbourhood), and,
for contrast, the same depths WITHOUT the Winner filter. Sweeper = taker wallet of the first transaction that printed
through our level. Wallet features (no look-ahead): walk-forward taker skill scores of the kickoff month from
research/wallets (scores_taker: s_n games scored, s_move = avg move-after-trade in its direction, shrunk), wallet age =
fill time - first time the wallet appears in the trade data, sweep usd, levels in the tx.
Per-share outcome = tmid_vk pnl; 'never came back' = mark 60 min after the fill still below our price.
Output: out/informed.csv
"""
import os
import numpy as np, pandas as pd
from s03_grid import load, OUT

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wallets", "data")


def main():
    f = load()
    f = f[(f.g == 0) & (f.R == 1) & f.d.isin([0.10, 0.12, 0.15, 0.20]) & f.c.between(0.08, 0.92) & (f.per != "after")]
    f = f.assign(month=pd.to_datetime(f.kick_ts, unit="s").dt.to_period("M").dt.to_timestamp())
    sc = pd.read_parquet(os.path.join(W, "scores_taker.parquet"))[["month", "wid", "n", "move", "res"]]
    sc = sc.rename(columns={"wid": "twid", "n": "s_n", "move": "s_move", "res": "s_res"})
    f = f.merge(sc, on=["month", "twid"], how="left")
    life = pd.read_parquet(os.path.join(W, "wallet_life.parquet"))
    life = life.reset_index()[["wid", "first_ts"]].rename(columns={"wid": "twid"}) if "wid" in life.reset_index() else life
    f = f.merge(life, on="twid", how="left")
    f["age_d"] = (f.ts - f.first_ts) / 86400
    f["s_n"] = f.s_n.fillna(0)
    f["skill"] = np.where(f.s_n < 20, "unscored(<20 games)", np.where(f.s_move > 0.005, "skilled(move>0.5c)",
                                                                     np.where(f.s_move < -0.002, "anti-skilled", "neutral")))
    f["age"] = pd.cut(f.age_d, [-1, 1, 7, 30, 1e5], labels=["<1d", "1-7d", "7-30d", ">30d"])
    f["size"] = pd.cut(f.tusd, [-1, 10, 100, 1000, 1e9], labels=["<$10", "$10-100", "$100-1k", ">$1k"])
    f["levels"] = pd.cut(f.lev, [0, 1, 3, 1e4], labels=["1", "2-3", "4+"])
    f["wfilter"] = np.where(f.gap.fillna(-1) >= 0.06, "gap>=6c", "gap<6c")
    f["never_back"] = (f.x60.fillna(f.xc) < 0)
    rows = []
    for feat in ["skill", "age", "size", "levels"]:
        for (wfl, b), x in f.groupby(["wfilter", feat], observed=True):
            rows.append(dict(feature=feat, bin=str(b), wfilter=wfl, n=len(x), games=x.uid.nunique(),
                             ret=float(x.tmid_vk.sum() / x.L.sum()), never_back=float(x.never_back.mean()),
                             x5=float(x.x5.mean())))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "informed.csv"), index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string())
    print("\nwithin gap>=6c: who are the losers? share of losing fills by skill bin vs all fills")
    x = f[f.wfilter == "gap>=6c"]
    print(pd.DataFrame(dict(all=x.skill.value_counts(normalize=True),
                            losers=x[x.tmid_vk < 0].skill.value_counts(normalize=True))).round(3))


if __name__ == "__main__":
    main()
