"""Step 7: summary tables + charts for the report.

 out/deviation_by_htk.csv   median |side price - fair| by hours-to-kick (mid proxy = mean of last taker-buy and last
                            taker-sell print in the hour, both required), fair = m2c (1X2 + O/U 2.5, train-calibrated)
 out/convergence.csv        does a deviation at T-6..-3h shrink by the last hour? slope of dev_last on dev_early
 out/live_scan_summary.csv  all live LP scans
 charts: out/coverage.png, out/live_lp_cost.png, out/deviation_by_htk.png, out/cum_pnl.png
"""
import os, glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "axes.edgecolor": "#8a8984", "font.size": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def coverage():
    c = pd.read_csv(os.path.join(OUT, "inventory_coverage_by_month.csv"), index_col=0)
    c = c[c.index >= "2025-09"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(c))
    ax.bar(x, c.games, color="#d9d8d2", label="1X2 games (ok)", width=0.8)
    ax.plot(x, c.totals, color=C1, lw=2, marker="o", ms=4, label="with totals/BTTS/spreads")
    ax.plot(x, c.soccer_exact_score, color=C2, lw=2, marker="o", ms=4, label="with exact score")
    ax.set_xticks(x); ax.set_xticklabels(c.index, rotation=45, ha="right"); ax.set_ylabel("games per month")
    ax.set_title("Side-market coverage of Polymarket soccer games", loc="left"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "coverage.png"), dpi=130); plt.close(fig)


def live():
    fs = sorted(glob.glob(os.path.join(OUT, "live_scan_2*.csv")))
    if not fs:
        return
    L = pd.concat([pd.read_csv(f).assign(scan=os.path.basename(f)[10:-4]) for f in fs])
    s = L.groupby("scan").agg(games=("game", "size"), lp_min=("lp_cost", "min"), lp_med=("lp_cost", "median"),
                              locks=("lp_cost", lambda x: int((x < 1 - 1e-9).sum())), ht_min=("ht_yes_sum", "min"),
                              es_min=("es_yes_sum", "min"), dev_med=("mean_abs_dev", "median")).reset_index()
    s.to_csv(os.path.join(OUT, "live_scan_summary.csv"), index=False); print(s.to_string())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(L.lp_cost.dropna().clip(upper=1.08), bins=np.arange(0.995, 1.081, 0.0025), color=C1, edgecolor="#fcfcfb")
    ax.axvline(1.0, color="#52514e", lw=1, ls="--"); ax.text(1.0005, ax.get_ylim()[1] * 0.9, "cost 1.00 = break-even", fontsize=9, color="#52514e")
    ax.set_xlabel("cheapest guaranteed-$1 portfolio from live best asks (+taker fees)"); ax.set_ylabel("game snapshots")
    ax.set_title(f"No locks in live books ({len(fs)} scans, {L.lp_cost.notna().sum()} game snapshots)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "live_lp_cost.png"), dpi=130); plt.close(fig)


def deviation():
    p = pd.read_parquet(os.path.join(HERE, "data", "panel_cal.parquet"))
    p = p[p.m2c.notna()]
    p["htk"] = (p.mtk // 60).astype(int)
    p = p[p.htk < 24]
    last = p.sort_values("ts").groupby(["unique_id", "mkey", "htk", "buyA"]).pA.last().unstack("buyA").dropna()
    last.columns = ["sell", "buy"]
    last["mid"] = (last.sell + last.buy) / 2
    last["spr"] = last.buy - last.sell
    f = p.groupby(["unique_id", "mkey", "htk"]).m2c.last()
    d = last.join(f).reset_index()
    d = d[(d.spr >= -0.005) & (d.spr <= 0.10)]
    d["dev"] = d.mid - d.m2c
    t = d.groupby("htk").agg(n=("dev", "size"), med_abs_dev=("dev", lambda x: x.abs().median()), mean_dev=("dev", "mean"),
                             med_spread=("spr", "median")).reset_index()
    t.to_csv(os.path.join(OUT, "deviation_by_htk.csv"), index=False); print(t.to_string())
    e = d[d.htk.between(3, 6)].sort_values("htk").groupby(["unique_id", "mkey"]).dev.last().rename("early")
    l = d[d.htk == 0].groupby(["unique_id", "mkey"]).dev.last().rename("late")
    j = pd.concat([e, l], axis=1).dropna()
    rows = []
    for k, x in [("all", j)] + [(k, x) for k, x in j.groupby(level="mkey") if len(x) >= 100]:
        b = np.polyfit(x.early, x.late, 1)[0] if len(x) > 5 else np.nan
        rows.append(dict(mkey=k, n=len(x), slope_late_on_early=b, med_abs_early=x.early.abs().median(), med_abs_late=x.late.abs().median()))
    cv = pd.DataFrame(rows); cv.to_csv(os.path.join(OUT, "convergence.csv"), index=False); print(cv.to_string())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(t.htk, t.med_abs_dev * 100, color=C1, lw=2, marker="o", ms=4, label="median |mid - fair|")
    ax.plot(t.htk, t.med_spread * 100, color=C2, lw=2, marker="o", ms=4, label="median print spread (buy - sell)")
    ax.invert_xaxis(); ax.set_xlabel("hours to kickoff"); ax.set_ylabel("cents"); ax.legend(frameon=False)
    ax.set_title("Side markets sit within ~1-2c of the 1X2+O/U2.5 fair value", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "deviation_by_htk.png"), dpi=130); plt.close(fig)


def cum():
    f = os.path.join(OUT, "rv_best_entries.parquet")
    if not os.path.exists(f):
        return
    e = pd.read_parquet(f)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for (s, col) in zip(sorted(e.strategy.unique()), [C1, C2, C3, C4]):
        x = e[e.strategy == s].sort_values("ts")
        ax.plot(pd.to_datetime(x.ts, unit="s"), (x.pnl / x.cost * 100).cumsum(), color=col, lw=2, label=f"{s} (cum. \\$ if every trade were \\$100)")
    for d in ["2026-07-01", "2026-10-01"]:
        ax.axvline(pd.Timestamp(d), color="#8a8984", lw=1, ls="--")
    ax.set_ylabel("$"); ax.legend(frameon=False); ax.set_title("Chosen rules, cumulative P&L (train | valid | test)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cum_pnl.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    coverage(); live(); deviation(); cum()
