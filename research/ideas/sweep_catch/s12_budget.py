"""s12: the sweep catcher under a fixed bankroll (Polymarket locks the full cost of every resting buy order).

Step 1 (build): for every 5-minute clock slot, the total price sum(L) of all orders the bot could rest, split by
hours-to-kickoff (0..71), side (Yes / No) and level (A, B)  ->  data/s12_lsum.npy (+ t0 of every game).
Same order rules as production: price band 8-92c, L >= 2c, Winner fair - L >= 6c, stop 10 min before kickoff.
Step 2 (evaluate): an allocation policy turns the bankroll into shares per order at each slot; fills come from the
s10 candidates (R=5), fill size = min(shares, through-volume), exits as in s11.
Policies:
  fixed(S, W)          S shares on every order within W hours (what production does; bankroll ignored)
  uniform(W, Smax)     S = min(Smax, budget / sum(L) of all orders within W hours): always fully deployed
  nearest(S0, Wmax)    hours-to-kickoff buckets are filled nearest first with S0 shares until the budget runs out
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); LL = os.path.join(HERE, "..", "..", "lead_lag", "data")
sys.path.insert(0, HERE)
from s10_detail import level, floor_c, LAG
LEV = {"A": 15.4, "B": 20.5}
SLOT0 = int(pd.Timestamp("2025-08-25").timestamp() // 300)
NSLOT = int(pd.Timestamp("2026-11-20").timestamp() // 300) - SLOT0
MONTHS = {0: 10.0, 1: 3.0, 2: 1.6}
VAL0 = pd.Timestamp("2026-07-01").timestamp(); TEST0 = pd.Timestamp("2026-10-01").timestamp()


def one(uid):
    try:
        z = np.load(os.path.join(LL, "series", f"{uid}.npz"))
    except FileNotFoundError:
        return None
    t0 = int(z["t0"]); P = np.round(z["P"].astype(np.float64), 4); wt = z["wt"].astype(np.float64); Q = z["Q"].astype(np.float64)
    n = len(P)
    if not len(wt):
        return None
    sl = np.arange(5, n - 10, 5)
    gs = (t0 + sl) // 5 - SLOT0
    hb = np.minimum((n - sl) // 60, 71)
    wi = np.searchsorted(wt, t0 + sl - 30, side="right") - 1
    out = []
    for k in range(3):
        for si, s in enumerate((1, -1)):
            X = P[:, k] if s > 0 else np.round(1 - P[:, k], 4)
            c = X[sl - LAG]
            wf = np.where(wi >= 0, (Q[:, k] if s > 0 else 1 - Q[:, k])[np.clip(wi, 0, None)], np.nan)
            for li, d in enumerate(LEV.values()):
                L = level("hyb", d, c)
                with np.errstate(invalid="ignore"):
                    ok = ~np.isnan(c) & (c >= 0.08) & (c <= 0.92) & (L >= 0.02) & (wf - L >= 0.06 - 1e-9) & (gs >= 0) & (gs < NSLOT)
                out.append((gs[ok], hb[ok], si, li, L[ok].astype(np.float32)))
    return uid, t0, out


def build():
    t = time.time()
    gm = pd.read_parquet(os.path.join(LL, "games_meta.parquet")); gm = gm[gm.ok == True]
    kt = pd.read_parquet(os.path.join(D, "tape.parquet"), columns=["unique_id"]).unique_id.unique()
    uids = sorted(set(gm.unique_id) & set(kt))
    LS = np.zeros((NSLOT, 72, 2, 2), np.float32); T0 = {}
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(one, uids, chunksize=32)):
            if r is None:
                continue
            uid, t0, out = r; T0[uid] = t0
            for gs, hb, si, li, L in out:
                np.add.at(LS[:, :, si, li], (gs, hb), L)
            if i % 3000 == 0:
                print(i, round(time.time() - t), "s", flush=True)
    np.save(os.path.join(D, "s12_lsum.npy"), LS)
    pd.Series(T0).to_frame("t0").to_parquet(os.path.join(D, "s12_t0.parquet"))
    print("done", round(time.time() - t), "s", "avg sum(L) within 3h:", LS[:, :3].sum((1, 2, 3)).mean())


_C = {}


def load():
    if not _C:
        f = pd.read_parquet(os.path.join(D, "s10_cand.parquet"))
        f = f[(f.R == 5) & (f.fam == "hyb") & (np.isclose(f.d, 15.4) | np.isclose(f.d, 20.5))].copy()
        t0 = pd.read_parquet(os.path.join(D, "s12_t0.parquet")).t0
        f["gs"] = ((f.uid.map(t0) + f.slot) // 5 - SLOT0).astype(int)
        f["hb"] = np.minimum(f.mtk // 60, 71).astype(int)      # same bucket as the order (slot within the same hour +-)
        f["si"] = (f.s < 0).astype(int); f["li"] = np.isclose(f.d, 20.5).astype(int)
        f = f[(f.c.between(0.08, 0.92)) & ((f.wf30 - f.L).fillna(-9) >= 0.06 - 1e-9) & (f.mtk >= 10) & (f.gs >= 0) & (f.gs < NSLOT)]
        _C["f"] = f.sort_values(["uid", "k", "s", "d", "ts"]).reset_index(drop=True)
        _C["LS"] = np.load(os.path.join(D, "s12_lsum.npy"))
    return _C["f"], _C["LS"]


def alloc(policy, sides=(0, 1), levels=(0, 1), budget=800.0):
    """shares per order as an array [slot, hb] (0 = not quoted); orders outside `sides` / `levels` get none."""
    f, LS = load()
    tot = LS[:, :, list(sides)][:, :, :, list(levels)].sum((2, 3))          # [slot, hb] sum of L in scope
    kind = policy[0]
    S = np.zeros(tot.shape, np.float32)
    if kind == "fixed":
        _, S0, W = policy; S[:, :W] = S0
    elif kind == "uniform":
        _, W, Smax = policy
        den = tot[:, :W].sum(1)
        S[:, :W] = np.minimum(Smax, budget / np.maximum(den, 1e-9))[:, None]
    elif kind == "nearest":
        _, S0, Wmax = policy
        cum = np.cumsum(tot[:, :Wmax] * S0, 1)                               # $ needed up to each bucket
        prev = cum - tot[:, :Wmax] * S0
        frac = np.clip((budget - prev) / np.maximum(tot[:, :Wmax] * S0, 1e-9), 0, 1)
        S[:, :Wmax] = S0 * frac
    used = (S * tot).sum(1)
    return S, used


def evaluate(policy, sides=(0, 1), levels=(0, 1), budget=800.0, name=None):
    f, LS = load()
    S, used = alloc(policy, sides, levels, budget)
    x = f[f.si.isin(sides) & f.li.isin(levels)].copy()
    x["S"] = S[x.gs.to_numpy(), x.hb.to_numpy()]
    x = x[x.S >= 5]                                                           # Polymarket minimum order: 5 shares
    keep = np.zeros(len(x), bool)
    U, K, S_, DD, TS, SL = x.uid.to_numpy(), x.k.to_numpy(), x.s.to_numpy(), x.d.to_numpy(), x.ts.to_numpy(), x.slot.to_numpy()
    prev = None; nxt = 0; last = -1
    for i in range(len(x)):
        key = (U[i], K[i], S_[i], DD[i])
        if key != prev:
            prev = key; nxt = 0; last = -1
        if TS[i] >= nxt and SL[i] > last:
            keep[i] = True; nxt = TS[i] + 3600; last = SL[i]
    x = x[keep].copy()
    q = np.minimum(x.S, x.sh_tx).astype(float)
    T = x.T_tm.to_numpy(); h = x.h_tm_5.to_numpy(); v = x.v_tm_5.to_numpy()
    qx = np.where(h >= 0, np.minimum(q, v), 0.0)
    x["q"] = q; x["pnl"] = qx * (T - x.L) + (q - qx) * x.vk.to_numpy(float); x["stake"] = x.L * q
    row = dict(name=name or str(policy))
    for p, lab in ((0, "tr"), (1, "va"), (2, "te")):
        z = x[x.per == p]
        row[f"fills_{lab}"] = round(len(z) / MONTHS[p], 1)
        row[f"ret_{lab}"] = round(100 * z.pnl.sum() / max(z.stake.sum(), 1e-9))
        row[f"usd_{lab}"] = round(z.pnl.sum() / MONTHS[p], 1)
    t = (np.arange(NSLOT) + SLOT0) * 300
    live = LS.sum((1, 2, 3)) > 0
    for lab, m in (("tr", t < VAL0), ("va", (t >= VAL0) & (t < TEST0))):
        row[f"avg_used_{lab}"] = round(float(used[m & live].mean())); row[f"max_used_{lab}"] = round(float(used[m].max()))
    row["max_stake_fill"] = round(float(x.stake.max()), 0) if len(x) else 0
    return row, x


if __name__ == "__main__":
    build()


def evaluate_priority(cells, S0, budget=800.0, name=None, levels=(0, 1)):
    """cells: [(side index, hour bucket)] in priority order; each gets S0 shares per order until the budget runs out
    (the cell where it runs out is scaled down)."""
    f, LS = load()
    S = np.zeros((NSLOT, 72, 2), np.float32); left = np.full(NSLOT, budget, np.float64); used = np.zeros(NSLOT)
    for si, hb in cells:
        need = LS[:, hb, si][:, list(levels)].sum(1) * S0
        frac = np.clip(left / np.maximum(need, 1e-9), 0, 1)
        S[:, hb, si] = S0 * frac
        used += need * frac; left -= need * frac
    x = f[f.li.isin(levels)].copy()
    x["S"] = S[x.gs.to_numpy(), x.hb.to_numpy(), x.si.to_numpy()]
    x = x[x.S >= 5]
    keep = np.zeros(len(x), bool)
    U, K, S_, DD, TS, SL = x.uid.to_numpy(), x.k.to_numpy(), x.s.to_numpy(), x.d.to_numpy(), x.ts.to_numpy(), x.slot.to_numpy()
    prev = None; nxt = 0; last = -1
    for i in range(len(x)):
        key = (U[i], K[i], S_[i], DD[i])
        if key != prev:
            prev = key; nxt = 0; last = -1
        if TS[i] >= nxt and SL[i] > last:
            keep[i] = True; nxt = TS[i] + 3600; last = SL[i]
    x = x[keep].copy()
    q = np.minimum(x.S, x.sh_tx).astype(float)
    T = x.T_tm.to_numpy(); h = x.h_tm_5.to_numpy(); v = x.v_tm_5.to_numpy()
    qx = np.where(h >= 0, np.minimum(q, v), 0.0)
    x["q"] = q; x["pnl"] = qx * (T - x.L) + (q - qx) * x.vk.to_numpy(float); x["stake"] = x.L * q
    row = dict(name=name)
    for p, lab in ((0, "tr"), (1, "va"), (2, "te")):
        z = x[x.per == p]
        row[f"fills_{lab}"] = round(len(z) / MONTHS[p], 1); row[f"ret_{lab}"] = round(100 * z.pnl.sum() / max(z.stake.sum(), 1e-9))
        row[f"usd_{lab}"] = round(z.pnl.sum() / MONTHS[p], 1)
    t = (np.arange(NSLOT) + SLOT0) * 300; live = LS.sum((1, 2, 3)) > 0
    row["avg_used_tr"] = round(float(used[(t < VAL0) & live].mean())); row["avg_used_va"] = round(float(used[(t >= VAL0) & (t < TEST0) & live].mean()))
    row["max_fill_stake"] = round(float(x.stake.max())) if len(x) else 0
    return row, x
