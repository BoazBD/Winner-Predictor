"""s06 - summary numbers (arb economics incl. capital lock-up / annualised return, bootstrap CIs) + simple SVG charts.

Inputs: data/arb_prints.parquet (s04), data/matches.parquet (s02), out/leadlag_predict.csv (s05), data/candles/ (s03)
Arb accounting: we copy each qualifying Polymarket print size (shares) and hedge the same number of Kalshi contracts.
  capital = size*cost ; pnl = size*(1-cost) ; lock-up days = (kickoff - print time) + 0.25 d (settlement on both venues)
  annualised = (1+edge)^(365/lockup)-1 per trade, capital-weighted mean reported as simple edge*365/lockup.
Scaling: candles were downloaded for a random subset of matched games -> $/month scaled by matched/quoted ratio.
Outputs: out/summary.json, out/arb_events.csv, out/chart_*.svg
"""
import glob, json
import numpy as np, pandas as pd

x = pd.read_parquet("data/arb_prints.parquet")
mt = pd.read_parquet("data/matches.parquet")
n_dl = len(glob.glob("data/candles/*.parquet"))
E = x[x.edge.notna() & x.clean].copy()      # clean = no Kalshi fair-price settlement, print direction sane
E["lock_d"] = -E.rel_min / 1440 + 0.25
S = {"matched_games": len(mt), "games_downloaded": n_dl, "games_with_quote_and_prints": int(E.unique_id.nunique()),
     "prints_with_quote": len(E)}
scale = len(mt) / max(n_dl, 1)
months = (mt.kick_utc.max() - mt.kick_utc.min()).days / 30.4


def boot_games(z, f, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    ids = z.unique_id.unique()
    if len(ids) < 3:
        return [np.nan, np.nan]
    grp = {k: v for k, v in z.groupby("unique_id")}
    vals = [f(pd.concat([grp[i] for i in rng.choice(ids, len(ids))])) for _ in range(n)]
    return list(np.percentile(vals, [2.5, 97.5]))


# distinct arb "events": consecutive qualifying prints in same game/outcome/direction within 30 min
rows = []
for th in (0.0, 0.005, 0.01, 0.02):
    tcol = "ts"
    y = E[(E.edge > th) & (E.edge_next > 0)].sort_values(tcol)
    y = y.assign(key=y.unique_id + y.j.astype(str) + y.yes.astype(str))
    y["new"] = (y.groupby("key")[tcol].diff().fillna(1e9) > 1800)
    y["ev"] = y.groupby("key").new.cumsum()
    y["first_ts"] = y.groupby(["key", "ev"])[tcol].transform("min")
    for lag in (5, 15, 60):     # what a slower poller could still catch: prints >= lag minutes after the arb opened
        y[f"pnl_after{lag}"] = np.where(y[tcol] - y.first_ts >= lag * 60, y.pnl, 0.0)
    ev = y.groupby(["key", "ev"]).agg(pnl_after5=("pnl_after5", "sum"), pnl_after15=("pnl_after15", "sum"),
                                      pnl_after60=("pnl_after60", "sum"),unique_id=("unique_id", "first"), series=("series", "first"),
                                      split=("split", "first"), start=(tcol, "min"), end=(tcol, "max"),
                                      prints=("q", "size"), cap=("usd", "sum"), pnl=("pnl", "sum"),
                                      max_edge=("edge", "max"), lock_d=("lock_d", "mean")).reset_index()
    ev["dur_min"] = (ev.end - ev.start) / 60
    if th == 0.005:
        ev.to_csv("out/arb_events.csv", index=False)
    ann = (ev.pnl.sum() / ev.cap.sum()) * 365 / max((ev.cap * ev.lock_d).sum() / max(ev.cap.sum(), 1e-9), 1e-9)
    rows.append(dict(th=th, events=len(ev), games=ev.unique_id.nunique(),
                     games_share=ev.unique_id.nunique() / S["games_with_quote_and_prints"],
                     capital=ev.cap.sum(), pnl=ev.pnl.sum(), ret_on_cap=ev.pnl.sum() / max(ev.cap.sum(), 1e-9),
                     pnl_after5=ev.pnl_after5.sum(), pnl_after15=ev.pnl_after15.sum(),
                     pnl_after60=ev.pnl_after60.sum(), events_dur_ge15=int((ev.dur_min >= 15).sum()),
                     median_event_cap=ev.cap.median(), median_dur_min=ev.dur_min.median(),
                     median_lock_d=ev.lock_d.median(), annualised_simple=ann,
                     pnl_per_month_scaled=ev.pnl.sum() * scale / months,
                     pnl_ci=boot_games(y, lambda d: d.pnl.sum()) if len(y) else [np.nan, np.nan]))
S["arb"] = rows
print(pd.DataFrame(rows).to_string())

# distribution of the cross-venue gap (print vs Kalshi same-side best price) - how close are the two books?
S["gap_quantiles_cents"] = {str(p): float(np.round(100 * E.gap.abs().quantile(p), 2)) for p in (0.5, 0.9, 0.99)}
S["cost_quantiles"] = {str(p): float(E.cost.quantile(p)) for p in (0.001, 0.01, 0.05, 0.5)}
S["by_split_prints_edge_gt0"] = E.assign(a=E.edge > 0).groupby("split").agg(prints=("a", "size"), arb=("a", "mean"),
                                                                              games=("unique_id", "nunique")).to_dict("index")
json.dump(S, open("out/summary.json", "w"), indent=1, default=float)


def svg_bars(labels, vals, title, fn, fmt="{:.1f}", ylab=""):
    W, H, L, B, T = 640, 300, 60, 60, 40
    vmax = max(max(vals), 0); vmin = min(min(vals), 0)
    rng = (vmax - vmin) or 1
    bw = (W - L - 20) / len(vals)
    y0 = T + (H - T - B) * vmax / rng
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>',
           f'<text x="{L}" y="20" font-size="14" fill="#0b0b0b">{title}</text>',
           f'<text x="10" y="{T - 6}" fill="#52514e">{ylab}</text>',
           f'<line x1="{L}" x2="{W - 20}" y1="{y0}" y2="{y0}" stroke="#c3c2b7"/>']
    for i, (lab, v) in enumerate(zip(labels, vals)):
        h = (H - T - B) * abs(v) / rng
        y = y0 - h if v >= 0 else y0
        xx = L + i * bw + 2
        out.append(f'<rect x="{xx:.1f}" y="{y:.1f}" width="{bw - 4:.1f}" height="{max(h, 1):.1f}" rx="2" fill="#2a78d6"/>')
        out.append(f'<text x="{xx + (bw - 4) / 2:.1f}" y="{(y - 4) if v >= 0 else (y + h + 12):.1f}" text-anchor="middle" fill="#0b0b0b">{fmt.format(v)}</text>')
        out.append(f'<text x="{xx + (bw - 4) / 2:.1f}" y="{H - B + 16}" text-anchor="middle" fill="#52514e" transform="rotate(30 {xx + (bw - 4) / 2:.1f} {H - B + 16})">{lab}</text>')
    out.append("</svg>")
    open(fn, "w").write("\n".join(out))


mm = mt.groupby(mt.kick_utc.dt.to_period("M")).size()
svg_bars([str(p) for p in mm.index], list(mm.values), "Polymarket games matched to a Kalshi market, by month",
         "out/chart_matches_by_month.svg", "{:.0f}", "games")
bins = [-1, -0.05, -0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 1]
h = pd.cut(1 - E.cost, bins).value_counts(normalize=True).sort_index()
svg_bars([str(i) for i in h.index], list(100 * h.values), "Hedged cost: profit per $1 payoff (1 - cost), % of prints",
         "out/chart_arb_margin_hist.svg", "{:.2f}", "% prints")
try:
    p = pd.read_csv("out/leadlag_predict.csv")
    p = p[(p.split == "train") & (p.H == 60)]
    svg_bars(list(p.b), list(100 * p.mean_pnl), "Lead/lag (train, H=60 min): Poly round-trip P&L by Kalshi gap",
             "out/chart_leadlag_train.svg", "{:.1f}", "cents/share")
except FileNotFoundError:
    pass
print(json.dumps({k: v for k, v in S.items() if k != "arb"}, indent=1, default=float))
