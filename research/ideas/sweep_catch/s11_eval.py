"""s11: evaluate production variants of the sweep catcher on the s10 candidates.

rule spec keys (defaults = the s07 'continuous' rule):
  R refresh min | fam 'mid'/'rel' | d depth | lo, hi price band of c | gap min Winner gap (wf - L) | wlag Winner data lag
  W quote window start (hours before kick) | E stop quoting (min before kick) | side 0 both / 1 / -1 | cool min after a
  fill | T exit target label | dl fill-detection delay (min) | S order shares | cap 'tx' or '60' (fill size source)
  xcap True: the target sell is capped by the shares printed above T | ex 'vk' (liquidate at kick) or 'hold'
  maxpos: max fills per (game, outcome, side); tier filter; vmin min game volume
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "out")
MONTHS = {0: 10.0, 1: 3.0, 2: 1.6}
PER = {0: "train", 1: "val", 2: "test"}
BASE = dict(R=1, fam="mid", d=0.20, lo=0.08, hi=0.92, gap=0.06, wlag=1, W=72, E=10, side=0, cool=60, T="tm", dl=0,
            S=1000, cap="tx", xcap=True, ex="vk", maxpos=99, tier=None, vmin=0, liqmin=0)
_F = {}


def cands():
    if "f" not in _F:
        f = pd.read_parquet(os.path.join(D, "s10_cand.parquet"))
        _F["f"] = f.sort_values(["uid", "k", "s", "R", "fam", "d", "ts"]).reset_index(drop=True)
    return _F["f"]


def lockdf():
    if "l" not in _F:
        _F["l"] = pd.read_parquet(os.path.join(OUT, "s10_lock.parquet"))
    return _F["l"]


def fills(spec):
    sp = {**BASE, **spec}
    f = cands()
    x = f[(f.R == sp["R"]) & (f.fam == sp["fam"]) & np.isclose(f.d, sp["d"])]
    gap = x[f"wf{sp['wlag']}"] - x.L
    m = x.c.between(sp["lo"], sp["hi"]) & (gap.fillna(-9) >= sp["gap"] - 1e-9) & (x.mtk <= sp["W"] * 60) & (x.mtk >= sp["E"])
    if sp["side"]:
        m &= x.s == sp["side"]
    if sp["tier"]:
        m &= x.tier.isin(sp["tier"])
    m &= (x.vol >= sp["vmin"]) & (x.liq6 >= sp["liqmin"])
    x = x[m]
    # one fill per refresh slot, cooldown after a fill, max fills per (game, outcome, side)
    keep = np.zeros(len(x), bool)
    U, K, S_, TS, SL = x.uid.to_numpy(), x.k.to_numpy(), x.s.to_numpy(), x.ts.to_numpy(), x.slot.to_numpy()
    prev = None; nxt_ts = 0; last_slot = -1; cnt = 0
    for i in range(len(x)):
        key = (U[i], K[i], S_[i])
        if key != prev:
            prev = key; nxt_ts = 0; last_slot = -1; cnt = 0
        if TS[i] >= nxt_ts and SL[i] > last_slot and cnt < sp["maxpos"]:
            keep[i] = True; nxt_ts = TS[i] + sp["cool"] * 60; last_slot = SL[i]; cnt += 1
    x = x[keep].copy()
    q = np.minimum(sp["S"], x.sh_tx if sp["cap"] == "tx" else x.sh60).astype(float)
    T = x[f"T_{sp['T']}"].to_numpy(); h = x[f"h_{sp['T']}_{sp['dl']}"].to_numpy(); v = x[f"v_{sp['T']}_{sp['dl']}"].to_numpy()
    qx = np.where(h >= 0, np.minimum(q, v) if sp["xcap"] else q, 0.0)
    rest = x[sp["ex"]].to_numpy(float)
    x["q"] = q; x["q_exit"] = qx
    x["pnl"] = qx * (T - x.L) + (q - qx) * rest
    x["stake"] = x.L * q
    x["hit"] = h >= 0
    return x, sp


def boot(pn, st, gid, n=500, seed=1):
    gi, gu = pd.factorize(gid)
    sp = np.bincount(gi, pn, len(gu)); ss = np.bincount(gi, st, len(gu))
    rng = np.random.default_rng(seed); G = len(gu)
    if G < 3:
        return sp.sum() / max(ss.sum(), 1e-9), np.nan, np.nan
    est = [(c * sp).sum() / max((c * ss).sum(), 1e-9) for c in (np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(n))]
    return sp.sum() / ss.sum(), np.percentile(est, 2.5), np.percentile(est, 97.5)


def locked(sp):
    """average $ locked by resting buy orders of S shares (all qualifying slots; gap buckets are 3c wide)."""
    l = lockdf()
    gb_min = int(np.ceil(sp["gap"] * 100 / 3 - 1e-9)) if sp["gap"] > -1 else -1
    m = (l.R == sp["R"]) & (l.fam == sp["fam"]) & np.isclose(l.d, sp["d"]) & (l.hb < sp["W"]) & (l.gb >= gb_min)
    if sp["lo"] >= 0.08:
        m &= l.bb == 1
    z = l[m].groupby("per").usdmin.sum()
    return {p: z.get(p, 0) * sp["S"] / (MONTHS[p] * 30 * 24 * 60) for p in MONTHS}


def evaluate(spec, name=None, pers=(0, 1), ci=True):
    x, sp = fills(spec)
    lk = locked(sp)
    rows = []
    for p in pers:
        z = x[x.per == p]
        r, lo, hi = boot(z.pnl.to_numpy(), z.stake.to_numpy(), z.uid.to_numpy()) if (ci and len(z)) else \
            (z.pnl.sum() / max(z.stake.sum(), 1e-9), np.nan, np.nan)
        rows.append(dict(name=name or str(spec), per=PER[p], n=len(z), games=z.uid.nunique(),
                         n_m=len(z) / MONTHS[p], ret=r, lo=lo, hi=hi, usd_m=z.pnl.sum() / MONTHS[p],
                         q_med=z.q.median() if len(z) else np.nan, hit=z.hit.mean() if len(z) else np.nan,
                         exit_full=(z.q_exit >= z.q - 1e-6).mean() if len(z) else np.nan,
                         res=z.vk_res.mean() if len(z) else np.nan, lock=lk[p],
                         top5=(z.pnl.nlargest(5).sum() / z.pnl.sum()) if len(z) and z.pnl.sum() > 0 else np.nan))
    return pd.DataFrame(rows)


def table(specs, pers=(0, 1), ci=True):
    out = pd.concat([evaluate(s, n, pers, ci) for n, s in specs])
    w = out.pivot(index="name", columns="per", values=["n_m", "ret", "lo", "usd_m", "lock"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.loc[[n for n, _ in specs]], out
