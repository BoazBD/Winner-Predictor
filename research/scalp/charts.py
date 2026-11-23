"""Chart helpers for s07_report.py: matplotlib -> base64 PNG (light theme, fixed categorical order)."""
import io, base64, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lead_lag", ".pylib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SPLIT_BG = {"train": "#f1f0ec", "val": "#e6eef9", "test": "#fbeee6"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlecolor": INK})
TRAIN_END, VAL_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01")


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def shade(ax, lo, hi):
    for a, b, lab in ((lo, TRAIN_END, "train"), (TRAIN_END, VAL_END, "val"), (VAL_END, hi, "test")):
        ax.axvspan(a, b, color=SPLIT_BG[lab], zorder=0, lw=0)
        ax.text(a + (b - a) / 2, 1.0, {"train": "train (tuning)", "val": "validation", "test": "final test"}[lab],
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK2)


def cum_pnl(tr, names, title):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    lo, hi = tr.kick.min(), tr.kick.max()
    shade(ax, lo, hi)
    for i, n in enumerate(names):
        x = tr[tr.name == n].sort_values("kick")
        ax.plot(x.kick, (x.ret * 100).cumsum(), color=SER[i % 8], lw=2, label=n)
        ax.annotate(n, (x.kick.iloc[-1], (x.ret * 100).sum()), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("cumulative profit, \\$ (at \\$100 per trade)")
    ax.set_title(title, loc="left", pad=16)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    return b64(fig)


def fill_bars(fr):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    rules = list(fr.rule.unique())
    ents = ["taker", "taker_v", "bid60", "bid240", "mid60"]
    lab = {"taker": "taker (at ask)", "taker_v": "taker, real print", "bid60": "bid, 1 h", "bid240": "bid, 4 h", "mid60": "mid, 1 h"}
    for ax, col, ttl in ((axes[0], "entry_fill", "Buy order filled"), (axes[1], "exit_fill_2c", "Sell at +2c filled before kickoff")):
        w = 0.8 / len(rules)
        for i, r in enumerate(rules):
            y = [fr[(fr.rule == r) & (fr.ent == e)][col].mean() * 100 for e in ents]
            ax.bar(np.arange(len(ents)) + i * w - 0.4 + w / 2, y, width=w - 0.03, color=SER[i], label=r)
        ax.set_xticks(range(len(ents))); ax.set_xticklabels([lab[e] for e in ents], rotation=25, ha="right", fontsize=8)
        ax.set_title(ttl, loc="left"); ax.set_ylim(0, 105)
    axes[0].set_ylabel("% of orders")
    axes[1].legend(fontsize=8, loc="upper right")
    return b64(fig)


def by_d(bd, names):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, n in enumerate(names):
        for sp, ls in (("train", "-"), ("val", "--")):
            x = bd[(bd.name == n) & (bd.split == sp) & (bd.d > 0)].sort_values("d")
            if len(x):
                ax.plot(x.d * 100, x.ret * 100, ls, marker="o", ms=5, color=SER[i % 8], lw=2,
                        label=f"{n} ({'train' if sp == 'train' else 'validation'})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("profit target d (cents above the buy price)"); ax.set_ylabel("return per $ staked, %")
    ax.set_title("Return by profit target (train solid, validation dashed)", loc="left")
    ax.legend(fontsize=7, ncol=2)
    return b64(fig)


def monthly(mm, name):
    x = mm[mm.name == name].sort_values("month")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cols = {"train": SER[0], "val": SER[2], "test": SER[1]}
    ax.bar(range(len(x)), x.usd, color=[cols[s] for s in x.split], width=0.7)
    for i, (u, n) in enumerate(zip(x.usd, x.n)):
        ax.text(i, u + (3 if u >= 0 else -3), f"n={n}", ha="center", va="bottom" if u >= 0 else "top", fontsize=7, color=INK2)
    ax.set_xticks(range(len(x))); ax.set_xticklabels(x.month, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("profit, \\$ (at \\$100 per trade)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[k], label=l) for k, l in (("train", "train"), ("val", "validation"), ("test", "final test"))], fontsize=8)
    ax.set_title(f"{name}: profit by month", loc="left")
    return b64(fig)


def mid_vs_trades(sens, names):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    vars_ = ["midpoint c=0", "midpoint c=1", "trades touch", "trades through (base)", "trades through, $100 must trade"]
    w = 0.8 / len(vars_)
    for i, v in enumerate(vars_):
        y = [sens[(sens.name == n) & (sens.variant == v) & (sens.split.isin(["train", "val"]))].ret.mean() * 100 for n in names]
        ax.bar(np.arange(len(names)) + i * w - 0.4 + w / 2, y, width=w - 0.03, color=SER[i], label=v)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_ylabel("return per $, % (train+val avg)")
    ax.set_title("Same strategy, different fill assumptions", loc="left"); ax.legend(fontsize=7, ncol=2)
    return b64(fig)
