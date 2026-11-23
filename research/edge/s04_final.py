"""Final numbers + charts for the report. Rules were chosen on TRAIN (kickoff < 2026-07-01) only."""
import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0, "../lead_lag/.pylib")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SPL = lambda s: np.where(s < pd.Timestamp("2026-07-01"), "train", np.where(s < pd.Timestamp("2026-10-01"), "val", "test"))
R = {}
def ci(g):
    r = np.random.default_rng(0); i = r.integers(0, len(g), (2000, len(g))); e = g[i, 0].sum(1) / g[i, 1].sum(1); return [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]

# ---------- 1. who loses: first real taker buy per share, last 24h before kickoff
t = pd.read_parquet("data/master.parquet", columns=["event_id","k","yes","q","size","ts","rel_min","won","start"])
pre = t[(t.rel_min < -5) & (t.rel_min > -1440)].sort_values("ts").drop_duplicates(["event_id", "k", "yes"])
pre["cost"] = pre.q + 0.05 * pre.q * (1 - pre.q); pre["ret"] = (pre.won - pre.cost) / pre.cost; pre["split"] = SPL(pre.start)
pre["pb"] = pd.cut(pre.q, [.03, .08, .15, .30, .50, .70, .85, .97])
R["buyers"] = pre.groupby(["pb", "split"], observed=True).agg(n=("ret", "size"), ret=("ret", "mean")).reset_index().assign(pb=lambda d: d.pb.astype(str)).to_dict("records")
b = pre.groupby("pb", observed=True).ret.mean()
fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(b))
ax.bar(x, 100 * b.values, color=["#c42f2f" if v < 0 else "#1baf7a" for v in b.values], width=.7)
for i, v in enumerate(b.values): ax.text(i, 100 * v - (2.2 if v < 0 else -0.6), f"{100*v:+.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"{int(100*i.left)}–{int(100*i.right)}¢" for i in b.index]); ax.axhline(0, color="#555", lw=.8)
ax.set_ylabel("Buyer's average return per $"); ax.set_xlabel("Price paid for the share (before kickoff)"); ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(min(100 * b.values) - 5, 5); fig.tight_layout(); fig.savefig("out/fig_buyers.png", dpi=140); plt.close()

# ---------- 2. be the house on extreme longshots
h = pd.read_parquet("data/house.parquet"); h = h[h.won.notna()].copy(); h["split"] = SPL(h.start)
kick_mid = {}
def variant(name, T0, lo, hi, delta, fill):
    x = h[(h.T0 == T0) & (h.mid >= lo) & (h.mid <= hi) & (np.isclose(h.delta, delta))]
    f = x[x[fill]].copy(); f["p"] = f.a - f.won; f["c"] = 1 - f.a
    out = dict(name=name, rule=dict(T0=T0, lo=lo, hi=hi, delta=delta, fill=fill))
    for s in ("train", "val", "test", "all"):
        xs = x if s == "all" else x[x.split == s]; fs = f if s == "all" else f[f.split == s]
        if len(fs) < 5: continue
        g = fs.groupby("uid")[["p", "c"]].sum().to_numpy()
        out[s] = dict(orders=int(len(xs)), fills=int(len(fs)), fill_rate=float(len(fs) / len(xs)), price=float(fs.a.mean()), lost=int(fs.won.sum()),
                      win_rate=float(1 - fs.won.mean()), roi=float(g[:, 0].sum() / g[:, 1].sum()), ci=ci(g))
    return out, f
V = []
for nm, a in [("A: 3–8¢, sell 1¢ above mid, 6h before (strict fill)", (6, .03, .08, .01, "f_thr")),
              ("A, optimistic fill (trade at our price counts)", (6, .03, .08, .01, "f_tch")),
              ("B: 3–8¢, sell at mid, 6h before (strict fill)", (6, .03, .08, 0.0, "f_thr")),
              ("C: 3–8¢, sell at mid, 24h before (strict fill)", (24, .03, .08, 0.0, "f_thr")),
              ("D: 8–15¢, sell 1¢ above mid, 6h before (strict fill)", (6, .0801, .15, .01, "f_thr")),
              ("E: 15–30¢, sell 1¢ above mid, 6h before (strict fill)", (6, .1501, .30, .01, "f_thr"))]:
    o, f = variant(nm, *a); V.append(o)
    if nm.startswith("B"): fB = f
    if nm.startswith("A:"): fA = f
R["house"] = V
# exit-before-kickoff variant for B: buy the share back as a taker 5 minutes before kickoff
ex = []
for r in fB.itertuples():
    z = np.load(f"../lead_lag/data/series/{r.uid}.npz"); m = float(z["P"][-5, r.k])
    hs = 0.015 if r.poly_volume < 1e4 else 0.005
    ex.append(r.a - (m + hs + 0.05 * m * (1 - m)))
fB["p_exit"] = ex
R["house_exit"] = {s: dict(n=int(len(d)), roi=float(d.p_exit.sum() / d.c.sum()), win=float((d.p_exit > 0).mean())) for s, d in fB.groupby("split")}
# equity curve for B at 100 shares per order (about $94 at risk each)
for f, tag in ((fB, "B"), (fA, "A")):
    f.sort_values("start", inplace=True); f["usd"] = 100 * f.p; f["cum"] = f.usd.cumsum()
mon = fB.assign(m=fB.start.dt.to_period("M").astype(str)).groupby("m").agg(bets=("usd", "size"), lost=("won", "sum"), usd=("usd", "sum"), cap=("c", lambda c: 100 * c.sum()))
R["house_monthly"] = mon.reset_index().to_dict("records")
# chance of a losing month: resample months of ~70 bets from all fills
rng = np.random.default_rng(1); u = fB.usd.to_numpy(); sims = rng.choice(u, (20000, 70)).sum(1)
R["house_month_sim"] = dict(bets=70, mean=float(sims.mean()), p_loss=float((sims < 0).mean()), p5=float(np.percentile(sims, 5)), p95=float(np.percentile(sims, 95)))
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axvspan(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01"), color="#eda100", alpha=.12, lw=0); ax.axvspan(pd.Timestamp("2026-10-01"), fB.start.max(), color="#2a78d6", alpha=.12, lw=0)
ax.step(fB.start, fB.cum, where="post", color="#2a78d6", lw=2, label="B: sell at mid"); ax.step(fA.start, fA.cum, where="post", color="#eb6834", lw=2, label="A: sell 1¢ above mid")
ax.axhline(0, color="#555", lw=.8); ax.set_ylabel("Cumulative profit, $ (100 shares per bet)"); ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
ax.text(pd.Timestamp("2026-07-05"), ax.get_ylim()[0] * .9 if ax.get_ylim()[0] < 0 else 5, "check period", fontsize=8, color="#7a5300"); ax.text(pd.Timestamp("2026-10-03"), ax.get_ylim()[0] * .9 if ax.get_ylim()[0] < 0 else 5, "final test", fontsize=8, color="#1c5cab")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig("out/fig_house.png", dpi=140); plt.close()

# ---------- 3. Winner <-> Polymarket arbitrage (clean cases only)
a = pd.read_parquet("data/arb.parquet"); a = a[(a.edge > 0.005) & (a.edge < 0.10) & (a.q > 0.03) & (a.q < 0.97) & (a.rel_min > -1440)]
a["split"] = SPL(a.start)
g = a.groupby("unique_id").agg(start=("start", "first"), split=("split", "first"), edge=("edge", "median"), outlay=("outlay", "sum"), first=("rel_min", "min"), last=("rel_min", "max"), vol=("poly_volume", "first"))
g["stake"] = g.outlay.clip(upper=1000); g["profit"] = g.stake * g.edge
ngames = pd.read_parquet("../lead_lag/data/games_meta.parquet").query("ok").assign(split=lambda d: SPL(d.start)).groupby("split").size()
R["arb"] = {s: dict(games=int(len(d)), of_games=int(ngames[s]), median_edge=float(d.edge.median()), profit_1000cap=float(d.profit.sum()), median_minutes_before=float(-d["last"].median()),
                    median_window_min=float((d["last"] - d["first"]).median())) for s, d in g.groupby("split")}
R["arb"]["per_month_recent"] = float(len(g[g.start >= "2026-08-01"]) / 3.6); R["arb"]["profit_per_month_recent"] = float(g[g.start >= "2026-08-01"].profit.sum() / 3.6)
fig, ax = plt.subplots(figsize=(8, 3.2)); ax.hist(100 * g.edge, bins=np.arange(0.5, 10.5, .5), color="#2a78d6", rwidth=.9)
ax.set_xlabel("Locked-in profit per $ (%)"); ax.set_ylabel("Games"); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig("out/fig_arb.png", dpi=140); plt.close()

# ---------- 4. after the final whistle: resting bid for the winning share
t["split"] = SPL(t.start)
p = t[(t.rel_min >= 150) & (t.q <= 0.0015)]            # winner sold at >= 0.9985 to a resting bid, 150+ min after start
pu = p["size"] * (1 - p.q); pp = p["size"] * ((1 - p.won) - (1 - p.q))
R["post"] = {s: dict(usd=float(pu[p.split == s].sum()), profit=float(pp[p.split == s].sum()), lost_usd=float(pu[(p.split == s) & (p.won == 1)].sum()), games=int(p[p.split == s].event_id.nunique())) for s in ("train", "val", "test")}
rec = p[p.start >= "2026-08-01"]; R["post"]["usd_per_day_recent"] = float((rec["size"] * (1 - rec.q)).sum() / 110)
# late-game lottery tickets (1-2c) sold by makers, minutes 100-135
l = t[(t.rel_min >= 100) & (t.rel_min < 135) & (t.q > 0.01) & (t.q <= 0.02)]; sz = l["size"].clip(upper=2000)
R["late"] = {s: dict(roi=float((sz * (l.q - l.won))[l.split == s].sum() / (sz * (1 - l.q))[l.split == s].sum()), cap=float((sz * (1 - l.q))[l.split == s].sum())) for s in ("train", "val", "test")}
# momentum check is in x05.py (no effect); favourites check in x03.py
json.dump(R, open("out/results.json", "w"), indent=1, default=str)
for k in ("house", "house_exit", "house_month_sim", "arb", "post", "late"): print(k, json.dumps(R[k], default=str)[:1800])
print(mon.round(1).to_string())
