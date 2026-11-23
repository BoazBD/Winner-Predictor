"""s07: final numbers for the rules chosen on TRAIN in s04 (+ two robustness variants), per period.

Rules (mid family, Winner gap filter applied at the order; exit = limit sell at the pre-sweep level floor(c), if not
filled before kick: VERIFIED liquidation 1c under the last real bid-hit print in the final 2 h, else hold to result):
  A  'continuous'  R=1  d=0.20 gap>=6c band 8-92c  (s04 pick for R=1)
  B  'hourly'      R=60 d=0.20 gap>=10c band 8-92c (s04 pick for R=60)
  C  'ladder'      R=1  three orders of S/3 at d=0.10, 0.15, 0.20 (gap>=6c), robustness (not selected)
Size: order S shares; fill q = min(S, shares printed strictly through our level within 60 s). S in {100, 1000}.
Stats: return per $ (sum pnl / sum L*q) with 95% CI by bootstrapping games (1000x), fills/month, $/month, worst day,
share of fills held to the result, share of fills whose sell target hit, collateral (from s05) -> return per locked $.
Outputs: out/final_summary.csv, out/final_monthly.csv, out/final_fills.parquet, charts fig_*.png
"""
import os, json
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lead_lag", ".pylib"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from s03_grid import load, OUT

MONTHS = {"train": 10.0, "val": 3.0, "test": 1.6}
RULES = {
    "A_continuous_R1_d20": dict(R=1, ds=[0.20], gap=0.06, w=[1.0]),
    "B_hourly_R60_d20": dict(R=60, ds=[0.20], gap=0.10, w=[1.0]),
    "C_ladder_R1_d10-15-20": dict(R=1, ds=[0.10, 0.15, 0.20], gap=0.06, w=[1 / 3] * 3),
    "D_hourly_ladder_R60": dict(R=60, ds=[0.10, 0.15, 0.20], gap=0.06, w=[1 / 3] * 3),
    # capital-efficient versions: quote only games with >= $10k printed in the previous 6 h (chosen on train in s05:
    # best $ earned per $ of collateral locked)
    "E_liquid_R1_d20": dict(R=1, ds=[0.20], gap=0.06, w=[1.0], liq=1e4),
    "F_liquid_ladder_R1": dict(R=1, ds=[0.10, 0.15, 0.20], gap=0.06, w=[1 / 3] * 3, liq=1e4),
    "G_liquid_ladder_R60": dict(R=60, ds=[0.10, 0.15, 0.20], gap=0.06, w=[1 / 3] * 3, liq=1e4),
    # capital-efficient by time: quote only in the last 3 h before kickoff (s05: best $ per locked $ on train)
    "H_last3h_R1_d20": dict(R=1, ds=[0.20], gap=0.06, w=[1.0], tmax=180),
    "I_last3h_ladder_R1": dict(R=1, ds=[0.10, 0.15, 0.20], gap=0.06, w=[1 / 3] * 3, tmax=180),
}
EXIT = "tmid_vk"


def boot(pn, st, gid, n=1000, seed=1):
    gi, gu = pd.factorize(gid)
    sp = np.bincount(gi, pn, len(gu)); ss = np.bincount(gi, st, len(gu))
    rng = np.random.default_rng(seed); G = len(gu); est = []
    for _ in range(n):
        c = np.bincount(rng.integers(0, G, G), minlength=G)
        est.append((c * sp).sum() / max((c * ss).sum(), 1e-9))
    return sp.sum() / ss.sum(), np.percentile(est, 2.5), np.percentile(est, 97.5)


def rule_fills(f, spec, S):
    parts = []
    for d, w in zip(spec["ds"], spec["w"]):
        x = f[(f.g == 0) & (f.R == spec["R"]) & np.isclose(f.d, d) & f.c.between(0.08, 0.92)
              & (f.gap.fillna(-1) >= spec["gap"]) & (f.liq6 >= spec.get("liq", 0))
              & (f.mtk <= spec.get("tmax", 10 ** 6))].copy()
        x["q"] = np.minimum(S * w, x.sh_thr.astype(float))
        parts.append(x)
    x = pd.concat(parts)
    x["pnl"] = x[EXIT] * x.q; x["stake"] = x.L * x.q
    x["day"] = pd.to_datetime(x.ts, unit="s").dt.floor("D")
    return x


def main():
    f = load()
    coll = pd.read_csv(os.path.join(OUT, "collateral.csv")) if os.path.exists(os.path.join(OUT, "collateral.csv")) else None
    rows, mon, keep = [], [], []
    for name, spec in RULES.items():
        for S in (100, 1000):
            x = rule_fills(f, spec, S)
            for per in ["train", "val", "test", "after"]:
                z = x[x.per == per]
                if not len(z):
                    continue
                r, lo, hi = boot(z.pnl.to_numpy(), z.stake.to_numpy(), z.uid.to_numpy())
                mo = MONTHS.get(per, np.nan)
                daily = z.groupby("day").pnl.sum()
                lock = np.nan
                if coll is not None and per in MONTHS:
                    # collateral table has gap 0 and 0.06; for gap>=10c the 6c figure is a (slight) upper bound
                    cc = coll[(coll.per == per) & np.isclose(coll.gap, min(spec["gap"], 0.06))]
                    if spec.get("tmax", 10 ** 6) <= 180:
                        cc = cc[cc.tb == 0]
                    if spec.get("liq", 0) >= 1e4:
                        cc = cc[cc.lb == 3]
                    # average $ locked = sum(L per share-minute) * S*w / minutes in period
                    lock = sum(cc[np.isclose(cc.d, d)].usdmin.sum() * S * w for d, w in zip(spec["ds"], spec["w"])) \
                        / (mo * 30 * 24 * 60)
                rows.append(dict(rule=name, S=S, period=per, fills=len(z), games=z.uid.nunique(),
                                 fills_per_month=len(z) / mo, ret_per_usd=r, ci_lo=lo, ci_hi=hi,
                                 usd_per_month=z.pnl.sum() / mo, stake_per_month=z.stake.sum() / mo,
                                 median_fill_shares=float(z.q.median()), worst_day=float(daily.min()),
                                 best_day=float(daily.max()), win_rate=float((z[EXIT] > 0).mean()),
                                 target_hit=float((z.tmid_hit >= 0).mean()), held_to_result=float(z.vk_res.mean()),
                                 short_k_share=float((z.s == -1).mean()), avg_locked_usd=lock,
                                 ret_per_locked_usd_month=(z.pnl.sum() / mo) / lock if lock and lock > 0 else np.nan))
            if S == 1000:
                m = x[x.per != "after"].assign(month=pd.to_datetime(x.ts, unit="s").dt.strftime("%Y-%m"))
                mm = m.groupby("month").agg(fills=("pnl", "size"), usd=("pnl", "sum"), stake=("stake", "sum"))
                mm["ret"] = mm.usd / mm.stake; mm["rule"] = name
                mon.append(mm.reset_index())
                keep.append(x.assign(rule=name))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "final_summary.csv"), index=False)
    pd.concat(mon).to_csv(os.path.join(OUT, "final_monthly.csv"), index=False)
    k = pd.concat(keep)
    k[["rule", "uid", "k", "s", "ts", "mtk", "per", "c", "L", "wf", "gap", "q", "sh_thr", "xmin", "tusd", "lev", "liq6",
       EXIT, "tmid_hit", "vk_res", "won", "pnl", "stake"]].to_parquet(os.path.join(OUT, "final_fills.parquet"))
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(s.round(3).to_string())
    print(pd.concat(mon).round(3).to_string())
    charts(f, k)


def charts(f, k):
    # 1 cumulative $ (S=1000)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, col in [("A_continuous_R1_d20", "#3366cc"), ("C_ladder_R1_d10-15-20", "#dd8800"),
                      ("B_hourly_R60_d20", "#22aa66"), ("F_liquid_ladder_R1", "#aa3399")]:
        z = k[(k.rule == name) & (k.per != "after")].sort_values("ts")
        ax.plot(pd.to_datetime(z.ts, unit="s"), z.pnl.cumsum(), label=name, color=col)
    for t, lab in [("2026-07-01", "val"), ("2026-10-01", "test")]:
        ax.axvline(pd.Timestamp(t), color="grey", ls="--", lw=0.8); ax.text(pd.Timestamp(t), ax.get_ylim()[1] * 0.95, lab)
    ax.set_title("Cumulative P&L, deep resting bids (S=1000 shares per order, fills capped by real through-volume)")
    ax.set_ylabel("$"); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_cum.png"), dpi=120)
    # 2 return per $ by depth: with vs without Winner filter, per period (R=1, tmid_vk, 1 share)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, per in zip(axs, ["train", "val", "test"]):
        for lab, gp, col in [("no Winner filter", -9, "#999999"), ("Winner gap>=0", 0.0, "#88aadd"),
                             ("Winner gap>=6c", 0.06, "#3366cc")]:
            x = f[(f.g == 0) & (f.R == 1) & (f.per == per) & f.c.between(0.08, 0.92)]
            if gp > -1:
                x = x[x.gap.fillna(-1) >= gp]
            r = x.groupby("d").apply(lambda z: z[EXIT].sum() / z.L.sum())
            ax.plot(r.index * 100, r.values * 100, marker="o", label=lab, color=col)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(per); ax.set_xlabel("depth below mid (c)")
    axs[0].set_ylabel("return per $ (%)"); axs[0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_depth.png"), dpi=120)
    # 3 mark path after the fill: mid - our price, Winner filter vs against
    fig, ax = plt.subplots(figsize=(6, 4))
    x = f[(f.g == 0) & (f.R == 1) & np.isclose(f.d, 0.15) & f.c.between(0.08, 0.92) & (f.per != "after")]
    for lab, m, col in [("gap>=6c (sweep pushed price AWAY from Winner)", x.gap >= 0.06, "#3366cc"),
                        ("gap<0 (sweep moved TOWARD/past Winner)", x.gap < 0, "#cc3333")]:
        z = x[m]
        ax.plot([0, 5, 15, 60, 120], [0, z.x5.mean(), z.x15.mean(), z.x60.mean(), z.xc.mean()], marker="o",
                label=f"{lab} n={len(z)}", color=col)
    ax.set_xticks([0, 5, 15, 60, 120]); ax.set_xticklabels(["fill", "+5m", "+15m", "+60m", "close"])
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("mid minus our buy price ($/share)")
    ax.set_title("After a 15c-deep fill"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_marks.png"), dpi=120)


if __name__ == "__main__":
    main()
