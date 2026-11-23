"""Step 7: statistics, bootstrap CIs (resampling games), placebos, breakdowns, charts.
Writes out/results.json, out/*.csv and out/*.png. REPORT.md quotes these numbers."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pylib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); O = os.path.join(HERE, "out"); os.makedirs(O, exist_ok=True)
B = 400
RNG = np.random.default_rng(1)
R = {}
COL = {"winner": "#2a6fdb", "poly": "#e8833a", "same": "#9aa0a6", "never": "#d0d4d9", "placebo": "#9aa0a6"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

meta = pd.read_parquet(os.path.join(D, "games_meta.parquet"))
meta = meta[meta.ok == True].copy()
# matching-quality filter: venues must roughly agree at kickoff (a wrong match shows up as a big level gap)
meta["good"] = meta.gap_kick <= 0.15
R["games_matched"] = int(len(meta)); R["games_good"] = int(meta.good.sum())
meta = meta[meta.good]
meta["month"] = pd.to_datetime(meta.kick_utc).dt.to_period("M").astype(str)
vb = pd.cut(meta.poly_volume, [-1, 1e4, 5e4, 2.5e5, 1e13], labels=["<10k", "10-50k", "50-250k", ">250k"])
meta["vol_bucket"] = vb.astype(str)
games = set(meta.unique_id)
M = meta.set_index("unique_id")


def boot(df, fn, b=B):
    """fn(df) -> scalar or array; CI by resampling games."""
    ids = df.unique_id.to_numpy(); u, inv = np.unique(ids, return_inverse=True)
    est = fn(df)
    out = []
    for _ in range(b):
        w = RNG.multinomial(len(u), np.ones(len(u)) / len(u))[inv]
        dd = df.loc[np.repeat(df.index.to_numpy(), w)]
        out.append(fn(dd))
    out = np.array(out)
    return est, np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0)


def fmt(e, lo, hi, pct=False, nd=1):
    f = (lambda x: f"{100 * x:.{nd}f}%") if pct else (lambda x: f"{x:.{nd + 1}f}")
    return f"{f(e)} [{f(lo)}, {f(hi)}]"


# ============================================================ events
ev = pd.read_parquet(os.path.join(D, "events.parquet"))
ev = ev[ev.unique_id.isin(games)].reset_index(drop=True)
for c in ("tier", "vol_bucket", "month", "poly_volume"):
    ev[c] = ev.unique_id.map(M[c])
ev["hour_il"] = (pd.to_datetime(ev.t_min * 60, unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Jerusalem").dt.hour)
ev["dow"] = pd.to_datetime(ev.t_min * 60, unit="s").dt.day_name().str[:3]
ev["ttk"] = pd.cut(ev.mins_to_kick / 60, [0, 1, 3, 6, 12, 24, 72], labels=["0-1h", "1-3h", "3-6h", "6-12h", "12-24h", "24-72h"]).astype(str)
ev["size"] = pd.cut(ev.M, [0, 0.05, 0.08, 0.12, 1], labels=["3-5", "5-8", "8-12", "12+"]).astype(str)
ev["lvl"] = pd.cut(ev.level, [0, 0.25, 0.5, 1], labels=["longshot <25%", "middle 25-50%", "favourite >50%"]).astype(str)


def classify(r):
    if r.pre_b >= 0.5:
        return "other_first"            # the other venue had already moved half of it in the 6 h before the base
    if r.b_half < 0:
        return "never"
    if np.isnan(r.p_time):
        return "never"
    # first-mover by time: Polymarket time vs Winner's (earliest, latest] window
    if r.p_time <= r.w_early:
        win_first = False
    elif r.p_time > r.w_late:
        win_first = True
    else:
        return "same"
    mover_is_winner = r.venue == "winner"
    return "self_first" if win_first == mover_is_winner else "other_first"


ev["cls"] = ev.apply(classify, axis=1)
# who moved first (venue names), for real events
ev["first"] = np.select([ev.cls == "same", ev.cls == "never",
                         (ev.cls == "self_first") == (ev.venue == "winner")],
                        ["same 15-min window", "other never followed", "Winner first"], "Polymarket first")
# lead of the other venue relative to the mover (minutes); Winner time uncertainty -> bounds
wmid = (ev.w_early + ev.w_late) / 2
ev["lead_mid"] = np.where(ev.venue == "winner", ev.p_time - wmid, wmid - ev.p_time)
ev["lead_lo"] = np.where(ev.venue == "winner", ev.p_time - ev.w_late, ev.w_early - ev.p_time)
ev["lead_hi"] = np.where(ev.venue == "winner", ev.p_time - ev.w_early, ev.w_late - ev.p_time)
# Polymarket time minus Winner time (positive = Winner earlier), from either trigger
ev["poly_minus_win_mid"] = np.where(ev.venue == "winner", ev.lead_mid, -ev.lead_mid)

real = ev[(ev.kind == "real") & ev.persist].copy()
plac = ev[ev.kind == "placebo"].copy()
shuf = ev[(ev.kind == "shuffled") & ev.persist].copy()
R["events"] = {f"{v}_{t}": int(len(real[(real.venue == v) & (real.thr == t)])) for v in ("winner", "poly") for t in (0.03, 0.05, 0.08)}
R["events_games"] = int(real.unique_id.nunique())
R["events_dropped_not_persistent"] = int(((ev.kind == "real") & ~ev.persist).sum())

order = ["Winner first", "same 15-min window", "Polymarket first", "other never followed"]
tab = []
for (v, t), g in real.groupby(["venue", "thr"]):
    row = dict(trigger=v, thr=t, n=len(g), games=g.unique_id.nunique())
    for o in order:
        e, lo, hi = boot(g, lambda d, o=o: (d["first"] == o).mean(), b=200)
        row[o] = fmt(e, lo, hi, pct=True)
        row[o + "_v"] = e
    for h in (0, 60, 180, 360):
        row[f"other_frac_{h}"] = g[f"fb_{h}"].median()
    row["other_frac_kick_med"] = g.fb_kick.median()
    row["self_frac_kick_med"] = g.fa_kick.median()
    sg = shuf[(shuf.venue == v) & (shuf.thr == t)]
    for o in order:                      # chance rates: the other venue's series taken from a random other game
        row["chance_" + o] = float((sg["first"] == o).mean())
    row["shuffled_other_frac_kick_med"] = float(sg.fb_kick.median())
    pg = plac[(plac.venue == v) & (plac.thr == t)]
    row["placebo_other_reach_half_6h"] = float((pg.fb_360 >= 0.5).mean())
    row["real_other_reach_half_6h"] = float((g.fb_360 >= 0.5).mean())
    row["placebo_other_frac_360_mean"] = float(pg.fb_360.mean()); row["real_other_frac_360_mean"] = float(g.fb_360.mean())
    tab.append(row)
tab = pd.DataFrame(tab); tab.to_csv(os.path.join(O, "who_first.csv"), index=False)

# conditional follow: the other venue had NOT moved half by the time the mover did; how often / how fast does it?
fol = []
for kind, src in (("real", real), ("chance (random other game)", shuf)):
    for (v, t), g in src.groupby(["venue", "thr"]):
        x = g[g.cls.isin(["self_first", "never"])]
        lag = np.where(x.b_half >= 0, x.b_half - x.i, np.inf)
        fol.append(dict(kind=kind, trigger=v, thr=t, n=len(x), share_of_moves=len(x) / len(g),
                        **{f"follow_within_{h}m": float((lag <= h).mean()) for h in (15, 60, 180, 360)},
                        follow_before_kick=float(np.isfinite(lag).mean())))
fol = pd.DataFrame(fol); fol.to_csv(os.path.join(O, "conditional_follow.csv"), index=False)
R["conditional_follow"] = fol.to_dict("records")
R["who_first"] = tab.drop(columns=[c for c in tab.columns if c.endswith("_v")]).to_dict("records")

# lead-time distributions (only where the other venue followed and neither 'pre-moved')
lt = []
for (v, t), g in real.groupby(["venue", "thr"]):
    f = g[(g.b_half >= 0) & (g.pre_b < 0.5)]
    for col in ("lead_lo", "lead_mid", "lead_hi"):
        q = f[col].quantile([.1, .25, .5, .75, .9]).to_dict()
        lt.append(dict(trigger=v, thr=t, measure=col, n=len(f), **{f"q{int(k * 100)}": round(x, 1) for k, x in q.items()},
                       share_within_15=float((f[col].abs() <= 15).mean()), share_neg=float((f[col] < 0).mean())))
lt = pd.DataFrame(lt); lt.to_csv(os.path.join(O, "lead_times.csv"), index=False)
R["lead_times"] = lt.to_dict("records")

# median lead CI (mid) for main configs
for v in ("winner", "poly"):
    g = real[(real.venue == v) & (real.thr == 0.03) & (real.b_half >= 0) & (real.pre_b < 0.5)]
    e, lo, hi = boot(g, lambda d: d.lead_mid.median(), b=200)
    R[f"median_lead_mid_{v}_3pt"] = [float(e), float(lo), float(hi)]

# ---- breakdowns (winner-triggered and poly-triggered, 3-pt)
bd_rows = []
for v in ("winner", "poly"):
    g0 = real[(real.venue == v) & (real.thr == 0.03)]
    for dim in ("tier", "ttk", "size", "lvl", "vol_bucket", "dow", "hour_il", "month"):
        for key, g in g0.groupby(dim):
            if len(g) < 30:
                continue
            bd_rows.append(dict(trigger=v, dim=dim, key=str(key), n=len(g), games=g.unique_id.nunique(),
                                winner_first=(g["first"] == "Winner first").mean(),
                                same=(g["first"] == "same 15-min window").mean(),
                                poly_first=(g["first"] == "Polymarket first").mean(),
                                never=(g["first"] == "other never followed").mean(),
                                other_frac_60=g.fb_60.median(), other_frac_360=g.fb_360.median(),
                                other_frac_kick=g.fb_kick.median(), mover_keeps_kick=g.fa_kick.median(),
                                gap_closing=(g.gap_before > 0.01).mean()))
bd = pd.DataFrame(bd_rows); bd.to_csv(os.path.join(O, "breakdowns.csv"), index=False)

# gap-closing split: did the mover move TOWARD the other venue's level (catching up) or away from it?
gc = []
for (v, t), g in real.groupby(["venue", "thr"]):
    for lab, s in (("toward other venue (gap >1pt)", g.gap_before > 0.01), ("away / no gap", g.gap_before <= 0.01)):
        x = g[s]
        gc.append(dict(trigger=v, thr=t, kind=lab, n=len(x), share=len(x) / len(g),
                       winner_first=(x["first"] == "Winner first").mean(), poly_first=(x["first"] == "Polymarket first").mean(),
                       never=(x["first"] == "other never followed").mean(), other_frac_kick=x.fb_kick.median()))
gc = pd.DataFrame(gc); gc.to_csv(os.path.join(O, "gap_closing.csv"), index=False)
R["gap_closing"] = gc.to_dict("records")

# ---- event-aligned average paths (chart)
offs = np.arange(-360, 361, 15) / 60
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, v in zip(axes, ("winner", "poly")):
    g = real[(real.venue == v) & (real.thr == 0.03)]
    pa = np.nanmean(np.stack(g.path_a.to_numpy()), 0); pb = np.nanmean(np.stack(g.path_b.to_numpy()), 0)
    pg = plac[(plac.venue == v) & (plac.thr == 0.03)]
    pp = np.nanmean(np.stack(pg.path_b.to_numpy()), 0)
    other = "poly" if v == "winner" else "winner"
    nm = {"winner": "Winner", "poly": "Polymarket"}
    ax.plot(offs, pa, color=COL[v], lw=2, label=f"{nm[v]} (the mover)")
    ax.plot(offs, pb, color=COL[other], lw=2, label=f"{nm[other]}")
    ps_ = np.nanmean(np.stack(shuf[(shuf.venue == v) & (shuf.thr == 0.03)].path_b.to_numpy()), 0)
    ax.plot(offs, pp, color=COL["placebo"], lw=1, ls="--", label=f"{nm[other]} at random times")
    ax.plot(offs, ps_, color="#555", lw=1, ls=":", label=f"{nm[other]} of a random other game")
    ax.axvline(0, color="#555", lw=.6); ax.axhline(0, color="#555", lw=.6)
    ax.set_title(f"{nm[v]} moves >=3 pts (n={len(g):,})"); ax.set_xlabel("hours from detection")
    ax.legend(frameon=False, fontsize=7)
axes[0].set_ylabel("share of the move made (1 = full move)")
fig.tight_layout(); fig.savefig(os.path.join(O, "fig_event_paths.png")); plt.close(fig)

# ---- who-first stacked bars
fig, ax = plt.subplots(figsize=(8, 3.2))
labs, y = [], 0
cmap = {"Winner first": COL["winner"], "same 15-min window": COL["same"], "Polymarket first": COL["poly"],
        "other never followed": COL["never"]}
for _, r in tab.iterrows():
    left = 0
    for o in order:
        ax.barh(y, r[o + "_v"], left=left, color=cmap[o], label=o if y == 0 else None)
        if r[o + "_v"] > 0.06:
            ax.text(left + r[o + "_v"] / 2, y, f"{100 * r[o + '_v']:.0f}%", ha="center", va="center", fontsize=7, color="white" if o != "other never followed" else "#333")
        left += r[o + "_v"]
    labs.append(f"{'Winner' if r.trigger == 'winner' else 'Polymarket'} moves >={int(r.thr * 100)} pts (n={r.n:,})"); y += 1
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.set_xlim(0, 1); ax.invert_yaxis()
ax.legend(ncol=4, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.4, 1.18)); ax.set_xlabel("share of major moves")
fig.tight_layout(); fig.savefig(os.path.join(O, "fig_who_first.png")); plt.close(fig)

# ---- lead-time histogram: Polymarket half-move time minus Winner half-move time (both triggers, 3pt)
g = real[(real.thr == 0.03) & (real.b_half >= 0) & (real.pre_b < 0.5)]
fig, ax = plt.subplots(figsize=(7, 3))
bins = np.arange(-360, 361, 15)
ax.hist(np.clip(g.poly_minus_win_mid, -360, 360), bins=bins, color=COL["poly"], alpha=.85)
ax.axvline(0, color="#333", lw=.8)
ax.set_xlabel("Polymarket time minus Winner time, minutes (Winner at mid-point of its 15-min window; clipped at ±6 h)\n<0: Polymarket moved first     >0: Winner moved first")
ax.set_ylabel("moves")
fig.tight_layout(); fig.savefig(os.path.join(O, "fig_lead_hist.png")); plt.close(fig)

# ============================================================ cross-correlation / impulse response
grid = pd.read_parquet(os.path.join(D, "grid.parquet"))
grid = grid[grid.unique_id.isin(games)].sort_values(["unique_id", "j"]).reset_index(drop=True)
gid = grid.unique_id.to_numpy()
same = np.r_[False, gid[1:] == gid[:-1]]
step = np.r_[0, np.diff(grid.j.to_numpy())]
valid = same & (step >= 10) & (step <= 20)
LAGS = np.arange(-24, 25)
u, ginv = np.unique(gid, return_inverse=True)
ng = len(u)
sums = {k: np.zeros((ng, len(LAGS))) for k in ("wp", "ww", "pp", "pw")}
sww = np.zeros(ng); spp = np.zeros(ng)
for k in (0, 1, 2):
    w = grid[f"w{k}"].to_numpy(); p = grid[f"p{k}"].to_numpy()
    dw = np.where(valid, np.r_[np.nan, np.diff(w)], np.nan); dp = np.where(valid, np.r_[np.nan, np.diff(p)], np.nan)
    dw = np.nan_to_num(dw); dp = np.nan_to_num(dp)
    sww += np.bincount(ginv, dw * dw, ng); spp += np.bincount(ginv, dp * dp, ng)
    for li, L in enumerate(LAGS):
        if L >= 0:
            a, b_ = dw[:len(dw) - L], dp[L:]; ga = ginv[:len(dw) - L]; ok = ginv[L:] == ga
        else:
            a, b_ = dw[-L:], dp[:len(dp) + L]; ga = ginv[-L:]; ok = ginv[:len(dp) + L] == ga
        sums["wp"][:, li] += np.bincount(ga, np.where(ok, a * b_, 0), ng)


def xc(wgt):
    return (wgt @ sums["wp"]) / np.sqrt((wgt @ sww) * (wgt @ spp))


def irf_p_on_w(wgt):   # Polymarket change at lag L per 1 pt Winner change now
    return (wgt @ sums["wp"]) / (wgt @ sww)


def irf_w_on_p(wgt):   # Winner change at lag -L per 1 pt Polymarket change now (reverse direction)
    return (wgt @ sums["wp"]) / (wgt @ spp)


one = np.ones(ng)
bs = [RNG.multinomial(ng, one / ng).astype(float) for _ in range(B)]
xce = xc(one); xcb = np.array([xc(w) for w in bs])
ire = irf_p_on_w(one); irb = np.array([irf_p_on_w(w) for w in bs])
iwe = irf_w_on_p(one); iwb = np.array([irf_w_on_p(w) for w in bs])
xcdf = pd.DataFrame(dict(lag_15min=LAGS, corr=xce, lo=np.percentile(xcb, 2.5, 0), hi=np.percentile(xcb, 97.5, 0)))
xcdf.to_csv(os.path.join(O, "xcorr.csv"), index=False)
pos = LAGS > 0; neg = LAGS < 0
cum_p = np.array([irf_p_on_w(w)[pos][:24].sum() for w in bs])
cum_w = np.array([irf_w_on_p(w)[neg][-24:].sum() for w in bs])
R["xcorr"] = dict(lag0=float(xce[LAGS == 0][0]), lag_p1=float(xce[LAGS == 1][0]), lag_m1=float(xce[LAGS == -1][0]),
                  sum_pos_1_8=float(xce[(LAGS >= 1) & (LAGS <= 8)].sum()), sum_neg_1_8=float(xce[(LAGS <= -1) & (LAGS >= -8)].sum()),
                  poly_follow_6h_per_winner_pt=[float(ire[pos][:24].sum()), float(np.percentile(cum_p, 2.5)), float(np.percentile(cum_p, 97.5))],
                  poly_same_interval_per_winner_pt=float(ire[LAGS == 0][0]),
                  winner_follow_6h_per_poly_pt=[float(iwe[neg][-24:].sum()), float(np.percentile(cum_w, 2.5)), float(np.percentile(cum_w, 97.5))],
                  winner_same_interval_per_poly_pt=float(iwe[LAGS == 0][0]), n_games=int(ng), n_intervals=int(valid.sum()))
fig, ax = plt.subplots(figsize=(7, 3))
ax.fill_between(LAGS * 15 / 60, xcdf.lo, xcdf.hi, color=COL["poly"], alpha=.25, lw=0)
ax.plot(LAGS * 15 / 60, xce, "o-", ms=3, color=COL["poly"])
ax.axhline(0, color="#555", lw=.6); ax.axvline(0, color="#555", lw=.6)
ax.set_xlabel("lag (hours).  >0: Winner change now vs Polymarket change LATER (Winner leads)\n<0: Polymarket change EARLIER vs Winner change now (Polymarket leads)")
ax.set_ylabel("correlation of 15-min changes")
fig.tight_layout(); fig.savefig(os.path.join(O, "fig_xcorr.png")); plt.close(fig)

# ---- error-correction: who closes a gap between the two venues?
ecm = []
grid["gkey"] = grid.unique_id
for h in (4, 12, 24):
    rows = []
    for k in (0, 1, 2):
        w = grid[f"w{k}"]; p = grid[f"p{k}"]
        gap = p - w
        gap = gap - gap.groupby(grid.unique_id).transform("mean")            # transient part of the gap
        fw = w.groupby(grid.unique_id).shift(-h) - w; fp = p.groupby(grid.unique_id).shift(-h) - p
        rows.append(pd.DataFrame(dict(unique_id=grid.unique_id, gap=gap, fw=fw, fp=fp)))
    df = pd.concat(rows).dropna().reset_index(drop=True)
    df = df[df.gap.abs() < 0.3]

    def slopes(d):
        g2 = (d.gap ** 2).sum()
        aw = (d.gap * d.fw).sum() / g2; ap = -(d.gap * d.fp).sum() / g2
        return np.array([aw, ap, aw / (aw + ap)])
    e, lo, hi = boot(df.sample(min(len(df), 400000), random_state=1), slopes, b=100)
    liq = df[df.unique_id.map(M.poly_volume) >= 2.5e5]
    el = slopes(liq) if len(liq) > 1000 else [np.nan] * 3
    ecm.append(dict(horizon_h=h / 4, liquid_winner_share=float(el[2]), liquid_poly_closes=float(el[1]),
                    liquid_winner_closes=float(el[0]), liquid_n=len(liq), winner_closes=e[0], winner_ci=[lo[0], hi[0]], poly_closes=e[1], poly_ci=[lo[1], hi[1]],
                    winner_share_of_adjustment=e[2], share_ci=[lo[2], hi[2]], n=len(df)))
R["ecm"] = ecm

# ============================================================ trading
tr = pd.read_parquet(os.path.join(D, "trades.parquet"))
tr = tr[tr.unique_id.isin(games)].reset_index(drop=True)
tr["tier"] = tr.unique_id.map(M.tier); tr["month"] = tr.unique_id.map(M.month)
tr["ev_vol"] = tr.unique_id.map(M.poly_volume)
tr["vol_bucket"] = tr.unique_id.map(M.vol_bucket)
res_k = {k: tr.unique_id.map(M[f"outcome_price{k}"]) for k in range(3)}
tr["won_yes"] = pd.to_numeric(np.select([tr.k == 0, tr.k == 1, tr.k == 2], [res_k[0], res_k[1], res_k[2]]), errors="coerce")
FEE = 0.05


def pnl(df, half_spread, fee_rate=None):
    """per-share and per-$ results of buying the token in the Winner-move direction (Yes if up, No if down)."""
    pe = np.where(df.d > 0, df.p_entry, 1 - df.p_entry)          # mid of the token we buy
    pc = np.where(df.d > 0, df.p_close, 1 - df.p_close)
    ask = np.clip(pe + half_spread, 0.001, 0.999)
    fee = (FEE if fee_rate is None else fee_rate) * ask * (1 - ask)
    cost = ask + fee
    clv = pc - cost                                                # value at kickoff price minus all-in cost
    won = np.where(df.d > 0, df.won_yes, 1 - df.won_yes)
    real_ = won - cost
    return pe, clv, clv / cost, real_ / cost


tr["hs"] = np.where(tr.ev_vol >= 1e4, 0.005, 0.015)          # half-spread model from live snapshot (s00_spreads.py)
pe, clv, roi_clv, roi_real = pnl(tr, tr.hs.to_numpy())
tr["pe"] = pe; tr["gross_move"] = np.where(tr.d > 0, tr.p_close - tr.p_entry, tr.p_entry - tr.p_close)
tr["clv"] = clv; tr["roi_clv"] = roi_clv; tr["roi_real"] = roi_real
tr["remaining_frac"] = tr.gross_move / tr.dW.abs()
tr["ttk"] = pd.cut(tr.mins_to_kick / 60, [0, 1, 3, 6, 12, 24, 72], labels=["0-1h", "1-3h", "3-6h", "6-12h", "12-24h", "24-72h"]).astype(str)
tr = tr[(tr.pe > 0.02) & (tr.pe < 0.98)]
tr["gap_now"] = tr.d * (tr.w_now - tr.p_entry)
tr["gap_bucket"] = pd.cut(tr.gap_now, [-1, 0, 0.02, 0.04, 1], labels=["Poly already past Winner", "0-2pt", "2-4pt", ">4pt"]).astype(str)
for h in (15, 60, 180, 360):
    tr[f"move_{h}"] = tr.d * (tr[f"p_{h}"] - tr.p_entry)
trows = []
for (cad, thr), g in tr.groupby(["cadence", "thr"]):
    e1 = boot(g, lambda d: d.gross_move.mean(), b=200)
    e2 = boot(g, lambda d: d.roi_clv.mean(), b=200)
    e3 = boot(g, lambda d: d.roi_real.mean(), b=200)
    trows.append(dict(cadence_min=cad, thr=thr, n=len(g), games=g.unique_id.nunique(),
                      gross_move_pts=100 * e1[0], gross_ci=[100 * e1[1], 100 * e1[2]],
                      remaining_frac_med=g.remaining_frac.median(),
                      roi_clv=e2[0], roi_clv_ci=[e2[1], e2[2]], roi_realized=e3[0], roi_real_ci=[e3[1], e3[2]],
                      roi_clv_zero_spread=float(pnl(g, 0.0)[2].mean()), roi_clv_2c_spread=float(pnl(g, 0.01)[2].mean()),
                      roi_clv_no_fee=float(pnl(g, g.hs.to_numpy(), 0.0)[2].mean()),
                      roi_clv_no_cost=float(pnl(g, 0.0, 0.0)[2].mean()),
                      hit_rate=float((g.gross_move > 0).mean()),
                      **{f"move_{h}m_pts": float(100 * g[f"move_{h}"].mean()) for h in (15, 60, 180, 360)}))
trd = pd.DataFrame(trows); trd.to_csv(os.path.join(O, "trading.csv"), index=False)
R["trading"] = trd.to_dict("records")

# recent period only (Polymarket taker fees switched on in April 2026; markets much deeper since then)
rec = []
for (cad, thr), g in tr[tr.month >= "2026-04"].groupby(["cadence", "thr"]):
    e2 = boot(g, lambda d: d.roi_clv.mean(), b=200)
    gg = g[g.gap_now > 0.04]
    e3 = boot(gg, lambda d: d.roi_clv.mean(), b=200) if len(gg) > 30 else (np.nan, np.nan, np.nan)
    rec.append(dict(cadence_min=cad, thr=thr, n=len(g), gross_move_pts=100 * g.gross_move.mean(), roi_clv=e2[0],
                    roi_clv_ci=[e2[1], e2[2]], roi_real=g.roi_real.mean(), n_gap4=len(gg), roi_clv_gap4=e3[0],
                    roi_clv_gap4_ci=[e3[1], e3[2]]))
R["trading_recent"] = rec
pd.DataFrame(rec).to_csv(os.path.join(O, "trading_recent.csv"), index=False)

# breakdowns of the 3-pt signal by cadence
tb_rows = []
for cad in (15, 60, 180):
    g0 = tr[(tr.cadence == cad) & (tr.thr == 0.03)]
    for dim in ("tier", "ttk", "vol_bucket", "month", "gap_bucket"):
        for key, g in g0.groupby(dim):
            if len(g) < 30:
                continue
            tb_rows.append(dict(cadence=cad, dim=dim, key=str(key), n=len(g), gross_pts=100 * g.gross_move.mean(),
                                roi_clv=g.roi_clv.mean(), roi_real=g.roi_real.mean()))
tbd = pd.DataFrame(tb_rows); tbd.to_csv(os.path.join(O, "trading_breakdowns.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 3))
x = 0; xt = []; xl = []
for thr in (0.02, 0.03, 0.05):
    for cad in (15, 60, 180):
        r = trd[(trd.cadence_min == cad) & (trd.thr == thr)].iloc[0]
        ax.bar(x, 100 * r.roi_clv, color={15: "#2a6fdb", 60: "#6c9be8", 180: "#b3c9f0"}[cad],
               label=f"check every {cad} min" if thr == 0.02 else None)
        ax.errorbar(x, 100 * r.roi_clv, yerr=[[100 * (r.roi_clv - r.roi_clv_ci[0])], [100 * (r.roi_clv_ci[1] - r.roi_clv)]], color="#333", lw=1, capsize=2)
        xt.append(x); xl.append(f">={int(thr * 100)}pt"); x += 1
    x += 0.6
ax.axhline(0, color="#333", lw=.7)
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7); ax.set_ylabel("return per $ staked (%)\nafter fee + half-spread")
ax.set_title("'Winner moved >=X pts in 6 h, Polymarket has not' - buy on Polymarket, value at kickoff price", fontsize=8)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(O, "fig_trading.png")); plt.close(fig)

# ============================================================ monthly stability + coverage
mon = []
for mth, g in real[real.thr == 0.03].groupby("month"):
    gw = g[g.venue == "winner"]; gp = g[g.venue == "poly"]
    t15 = tr[(tr.month == mth) & (tr.cadence == 60) & (tr.thr == 0.03)]
    mon.append(dict(month=mth, games=int((meta.month == mth).sum()), n_w=len(gw), n_p=len(gp),
                    w_first=(g["first"] == "Winner first").mean(), p_first=(g["first"] == "Polymarket first").mean(),
                    same=(g["first"] == "same 15-min window").mean(),
                    poly_follow_6h_after_winner=gw.fb_360.median(), winner_follow_6h_after_poly=gp.fb_360.median(),
                    trade_n_1h=len(t15), trade_roi_clv_1h=t15.roi_clv.mean() if len(t15) else np.nan))
mon = pd.DataFrame(mon); mon.to_csv(os.path.join(O, "monthly.csv"), index=False)
R["monthly"] = mon.to_dict("records")
fig, ax = plt.subplots(figsize=(8, 3))
mm = mon[mon.n_w + mon.n_p >= 50]
ax.plot(mm.month, mm.w_first, "o-", color=COL["winner"], label="Winner first")
ax.plot(mm.month, mm.p_first, "o-", color=COL["poly"], label="Polymarket first")
ax.plot(mm.month, mm.same, "o-", color=COL["same"], label="same 15-min window")
ax.set_ylabel("share of >=3-pt moves\n(both venues' moves pooled)"); ax.legend(frameon=False, fontsize=7); ax.tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(os.path.join(O, "fig_monthly.png")); plt.close(fig)

cov = meta.groupby("month").size()
allp = pd.read_parquet(os.path.join(D, "poly_events.parquet"))
allp_m = allp.groupby(allp.start.dt.to_period("M").astype(str)).size()
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(allp_m.index, allp_m.values, color="#d0d4d9", label="Polymarket 1X2 soccer events")
ax.bar(cov.index, cov.reindex(cov.index).values, color=COL["winner"], label="matched to Winner (used)")
ax.legend(frameon=False, fontsize=7); ax.tick_params(axis="x", rotation=60, labelsize=7); ax.set_ylabel("games")
fig.tight_layout(); fig.savefig(os.path.join(O, "fig_coverage.png")); plt.close(fig)
R["coverage_poly_by_month"] = allp_m.to_dict(); R["coverage_matched_by_month"] = cov.to_dict()
R["tiers"] = meta.tier.value_counts().to_dict(); R["vol_buckets"] = meta.vol_bucket.value_counts().to_dict()

json.dump(R, open(os.path.join(O, "results.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps(R, indent=1, default=str)[:6000])
