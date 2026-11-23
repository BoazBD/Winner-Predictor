"""s05 - How fast does Polymarket absorb scheduled team news (confirmed lineups)?

No free source gives historical lineups with publish times, but lineups are released at a fixed offset before
kickoff (typically 60 min domestic leagues, ~75 min UEFA). We locate the release empirically from trades:
per league, mean |1-minute change| of the YES-implied price of the pre-match favourite vs minutes-to-kick.
Then, per game, take the favourite's price path around its league's spike minute L and measure what fraction
of the move from L-2 to L+45 (capped at kick-2) has happened by L+1, +2, +5, +10, +15, +30 min, and how many
cents are left after each delay (games with a >= 2c move).
Price per minute = median YES-implied trade price in that minute (ffill), float64 rounded to 4 dp.
Outputs: out/lineup_vol_profile.csv, out/lineup_speed.csv, out/fig_lineup_profile.png
"""
import os, json
import numpy as np, pandas as pd, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet"))
ids = ",".join(f"'{u}'" for u in s.unique_id)
t = duckdb.sql(f"""select unique_id, k, case when yes then q else 1-q end as py, size, rel_min
                   from '{R}/edge/data/master.parquet' where unique_id in ({ids}) and rel_min between -180 and 0""").df()
t["py"] = t.py.astype("float64").round(4)
t["m"] = np.floor(t.rel_min).astype(int)          # minute bucket, -180..-1
# favourite = outcome with highest median price in [-180,-120]
pre = t[t.rel_min < -120].groupby(["unique_id", "k"]).py.median().reset_index()
fav = pre.loc[pre.groupby("unique_id").py.idxmax()][["unique_id", "k"]]
t = t.merge(fav, on=["unique_id", "k"])
pm = t.groupby(["unique_id", "m"]).py.median().unstack()        # games x minutes
pm = pm.reindex(columns=range(-180, 0)).ffill(axis=1)
d = pm.diff(axis=1).abs()
lg = s.set_index("unique_id").league.reindex(pm.index)
comp = lg.map(lambda x: "UEFA" if x in ("ליגת האלופות", "הליגה האירופית") else "domestic")
prof = d.groupby(comp).mean().T
prof.index.name = "mins_to_kick_neg"
prof.to_csv(os.path.join(HERE, "out/lineup_vol_profile.csv"))
res = {}
rows = []
# per-league spike minute L (3-min smoothed mean |1-min change|, searched in [-95,-40])
Lmap = {}
for L_, g in d.groupby(lg):
    pr = g.mean().rolling(3, center=True).mean()
    Lmap[L_] = int(pr.loc[-95:-40].idxmax())
res["L_by_league"] = Lmap
H = list(range(-10, 46))
A = pd.DataFrame({h: [pm.loc[u, min(Lmap[lg[u]] + h, -2)] for u in pm.index] for h in H}, index=pm.index)
for c in ["domestic", "UEFA"]:
    sub = A[comp == c]
    tot = sub[45] - sub[-2]
    big = tot.abs() >= 0.02
    res[c] = dict(n_games=int((comp == c).sum()), n_move_ge_2c=int(big.sum()), share_games_move_ge_2c=float(big.mean()),
                  mean_abs_move_lineup_window_c=float(tot.abs().mean() * 100),
                  mean_abs_move_quiet_hour_c=float((pm.loc[sub.index, -120] - pm.loc[sub.index, -170]).abs().mean() * 100),
                  pre_move_10min_before_c=float(((sub[-2] - sub[-10]) * np.sign(tot))[big].mean() * 100))
    for h in [0, 1, 2, 3, 5, 10, 15, 20, 30, 45]:
        f = ((sub[h] - sub[-2]) / tot)[big]
        rows.append(dict(comp=c, h=h, n_big=int(big.sum()), frac_done_median=float(f.median()),
                         frac_done_mean=float(f.clip(-1, 2).mean()),
                         left_after_cents=float(((sub[45] - sub[h]) * np.sign(tot))[big].mean() * 100)))
sp = pd.DataFrame(rows)
sp.to_csv(os.path.join(HERE, "out/lineup_speed.csv"), index=False)
json.dump(res, open(os.path.join(HERE, "out/lineup_window.json"), "w"), indent=1)
print(json.dumps(res, indent=1)); print(sp.to_string())

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for c, cl in [("domestic", "#2a6fdb"), ("UEFA", "#d9822b")]:
    ax[0].plot(prof.index, prof[c] * 100, color=cl, label=f"{c} (n={res[c]['n_games']})", lw=1.2)
    q = sp[sp.comp == c]; ax[1].plot(q.h, q.frac_done_mean, "o-", color=cl, label=f"{c} (games with >=2c move: {res[c]['n_move_ge_2c']})")
ax[0].set_xlabel("minutes to kickoff"); ax[0].set_ylabel("mean |1-min change| of favourite, cents")
ax[0].set_title("Price activity before kickoff"); ax[0].legend(frameon=False)
ax[1].axhline(1, color="grey", lw=.5); ax[1].set_xlabel("minutes after lineup spike")
ax[1].set_ylabel("share of 45-min move done (mean)"); ax[1].set_title("How fast the lineup move happens"); ax[1].legend(frameon=False)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "out/fig_lineup_profile.png"), dpi=130)
