"""Step 3: historical backtest of the TRADING side of a two-sided reward quoter (fills, adverse selection, inventory, exit).

Quoter, per game and per outcome market k (Yes-k book), re-quoting every CAD minutes from the 15-min grid midpoint:
  bid = floor((mid - 0.005 - j*0.01)), ask = ceil((mid + 0.005 + j*0.01))   (j = ticks behind the touch; j=0 joins
  the touch of a 1c book), N shares each side, refreshed to N every re-quote, inventory capped at |I| <= IMAX*N
  (the side that would grow inventory is capped), quoting window [TSTART, TEND] minutes before kickoff,
  only when 0.05 <= mid <= 0.95.
Fills: STRICT = a real trade printed strictly through our price (y < bid - 5e-5 for taker sells, y > ask + 5e-5 for
  taker buys); fill qty = min(remaining N, trade size summed). AT = also counts trades AT our price (optimistic, queue).
Exit at TEND: residual inventory crossed at grid mid -/+ half spread H(volume) minus taker fee 0.05 p(1-p).
  (No game-result risk: nothing is held past TEND.)
PnL components: edge at fill vs mid at fill-slot, markout to exit mid, exit cost. Maker rebate reported separately
  (0 for us below $200k 30-day maker-weighted volume; 'opt' = 15% of the taker fee our fills generated).
Vectorised over all (game,k) series: dense arrays [series, slot]. Writes out/sim_games.parquet (per game x config).
"""
import os, itertools, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
NS = 100  # 15-min slots: slot i covers minutes-to-kick (14+15(i-1), 14+15i]; slot 0 = last 14 minutes
EPS = 5e-5


def load():
    f = pd.read_parquet(os.path.join(HERE, "data", "fills_base.parquet"))
    g = pd.read_parquet(os.path.join(HERE, "data", "grid24.parquet"))
    meta = pd.read_parquet(os.path.join(R, "lead_lag/data/games_meta.parquet"),
                           columns=["unique_id", "ok", "start", "tier", "series", "poly_volume"])
    meta = meta[meta.ok]
    uids = sorted(set(f.unique_id) & set(meta.unique_id))
    uix = {u: i for i, u in enumerate(uids)}
    meta = meta.set_index("unique_id").loc[uids].reset_index()
    G = len(uids)
    mid = np.full((G, 3, NS), np.nan)
    g = g[g.unique_id.isin(uix)]
    gi = g.unique_id.map(uix).values; si = np.rint((g.mtk.values - 14) / 15).astype(int)
    ok = (si >= 0) & (si < NS)
    for k in range(3):
        mid[gi[ok], k, si[ok]] = g[f"p{k}"].values[ok]
    f = f[f.unique_id.isin(uix)]
    t = dict(g=f.unique_id.map(uix).values, k=f.k.values.astype(int),
             s=np.clip(np.rint((f.slot_mtk.values - 14) / 15).astype(int), 0, NS - 1),
             side=f.side.values, y=f.y.values.astype(np.float64).round(4), size=f["size"].values.astype(np.float64))
    return meta, mid, t


def quotes(mid, j):
    bid = np.floor(np.round((mid - 0.005 - 0.01 * j) * 100, 6) + 1e-6) / 100
    ask = np.ceil(np.round((mid + 0.005 + 0.01 * j) * 100, 6) - 1e-6) / 100
    return bid.round(4), ask.round(4)


def qmid(mid, cad):
    """mid used for the quote in each slot: refreshed every cad minutes (slot blocks of cad/15, quote set at block start
    = the earliest/highest-minutes slot of the block)."""
    b = cad // 15
    if b == 1: return mid
    q = mid.copy()
    for s in range(NS):
        top = min(NS - 1, (s // b) * b + b - 1)
        q[..., s] = mid[..., top]
    return q


def fill_arrays(meta, mid, t, cad, j):
    """shares available to fill our bid/ask per [g,k,slot] (strict and at-price)."""
    qm = qmid(mid, cad)
    bid, ask = quotes(qm, j)
    b = bid[t["g"], t["k"], t["s"]]; a = ask[t["g"], t["k"], t["s"]]
    y, side, sz = t["y"], t["side"], t["size"]
    out = {}
    shape = mid.shape
    for name, cond in [("B_s", (side < 0) & (y < b - EPS)), ("A_s", (side > 0) & (y > a + EPS)),
                       ("B_a", (side < 0) & (y < b + EPS)), ("A_a", (side > 0) & (y > a - EPS))]:
        arr = np.zeros(shape)
        c = cond & np.isfinite(b)
        np.add.at(arr, (t["g"][c], t["k"][c], t["s"][c]), sz[c])
        out[name] = arr
    return qm, bid, ask, out


def run(meta, mid, qm, bid, ask, F, mode, N, IMAX, TSTART, TEND, H, UNW=0):
    G = mid.shape[0]
    s_hi = (TSTART - 14) // 15; s_lo = int(np.ceil((TEND - 14) / 15))
    s_unw = int(np.ceil((TEND + UNW - 14) / 15))   # slots below this: quote only the inventory-reducing side
    if mode.startswith("q"):   # queue-aware: at-price volume only reaches us after X shares queued ahead
        X = float(mode[1:])
        Bav = F["B_s"] + np.maximum(0, F["B_a"] - F["B_s"] - X); Aav = F["A_s"] + np.maximum(0, F["A_a"] - F["A_s"] - X)
    else:
        Bav, Aav = F["B_" + mode], F["A_" + mode]
    inv = np.zeros((G, 3)); cash = np.zeros((G, 3)); nfill = np.zeros((G, 3)); notional = np.zeros((G, 3))
    edge = np.zeros((G, 3)); rebate = np.zeros((G, 3)); qhours = np.zeros((G, 3))
    for s in range(s_hi, s_lo - 1, -1):           # walk forward in time
        m, bq, aq = qm[:, :, s], bid[:, :, s], ask[:, :, s]
        live = np.isfinite(m) & (m >= 0.05) & (m <= 0.95)
        im = 0 if (UNW and s < s_unw) else IMAX
        capb = np.clip(im * N - inv, 0, N); capa = np.clip(im * N + inv, 0, N)
        fb = np.where(live, np.minimum(capb, Bav[:, :, s]), 0); fa = np.where(live, np.minimum(capa, Aav[:, :, s]), 0)
        fb = np.nan_to_num(fb); fa = np.nan_to_num(fa)
        bq0 = np.nan_to_num(bq); aq0 = np.nan_to_num(aq); m0 = np.nan_to_num(m)
        inv += fb - fa; cash += -fb * bq0 + fa * aq0
        nfill += (fb > 0).astype(int) + (fa > 0).astype(int); notional += fb * bq0 + fa * (1 - aq0)
        edge += fb * (m0 - bq0) + fa * (aq0 - m0)
        rebate += 0.15 * 0.05 * (fb * bq0 * (1 - bq0) + fa * aq0 * (1 - aq0))
        qhours += live * 0.25
    # exit mid = grid mid at the end of the window (slot s_lo-1, or the last known mid before it)
    me = mid[:, :, max(s_lo - 1, 0)]
    for s in range(s_lo, s_hi + 1):
        me = np.where(np.isfinite(me), me, mid[:, :, s])
    me = np.nan_to_num(me, nan=0.5)
    mtm = cash + inv * me
    exit_px = np.where(inv > 0, me - H[:, None], me + H[:, None])
    fee = 0.05 * np.clip(exit_px, 0.01, 0.99) * (1 - np.clip(exit_px, 0.01, 0.99))
    exit_cost = np.abs(inv) * (H[:, None] + fee)
    pnl = mtm - exit_cost
    return dict(pnl=pnl.sum(1), mtm=mtm.sum(1), edge=edge.sum(1), exit_cost=exit_cost.sum(1), rebate=rebate.sum(1),
                nfill=nfill.sum(1), notional=notional.sum(1), qhours=qhours.sum(1), absinv_end=np.abs(inv).sum(1))


def main():
    meta, mid, t = load()
    vol = meta.poly_volume.fillna(0).values
    H = np.where(vol > 1e4, 0.005, np.where(vol > 1e3, 0.01, 0.02))   # half spread at exit (live snapshot medians)
    rows = []
    import sys
    big = len(sys.argv) > 1 and sys.argv[1] == "big"
    grid = dict(cad=[15, 60], j=[0, 1, 2, 3], N=[100, 1000], IMAX=[1, 3], TSTART=[1440, 360, 120], TEND=[15, 60])
    queue = len(sys.argv) > 1 and sys.argv[1] == "queue"
    if queue:  # third pass: queue-position-aware fills + passive unwind window, EPL-relevant sizes
        grid = dict(cad=[15, 60], j=[0, 1], N=[1000, 5000, 20000], IMAX=[1, 3], TSTART=[900], TEND=[15, 60], UNW=[0, 120],
                    modes=["q5000", "q20000", "q50000", "s", "a"])
    if big:   # second pass: large sizes for reward-share scaling, match-day window (~15h)
        grid = dict(cad=[15, 60], j=[0, 1], N=[1000, 5000, 20000], IMAX=[1, 3], TSTART=[900, 360, 120], TEND=[15, 60])
    n = 0
    for cad, j in itertools.product(grid["cad"], grid["j"]):
        qm, bid, ask, F = fill_arrays(meta, mid, t, cad, j)
        for mode, N, IMAX, TS, TE, UW in itertools.product(grid.get("modes", ["s", "a"]), grid["N"], grid["IMAX"],
                                                           grid["TSTART"], grid["TEND"], grid.get("UNW", [0])):
            r = run(meta, mid, qm, bid, ask, F, mode, N, IMAX, TS, TE, H, UW)
            d = pd.DataFrame(r).astype("float32"); d["unique_id"] = meta.unique_id.values
            d["cad"], d["j"], d["mode"], d["N"], d["IMAX"], d["TSTART"], d["TEND"], d["UNW"] = cad, j, mode, N, IMAX, TS, TE, UW
            rows.append(d); n += 1
        print("done cad", cad, "j", j, flush=True)
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(meta[["unique_id", "start", "tier", "series", "poly_volume"]], on="unique_id")
    out.to_parquet(os.path.join(HERE, "out", "sim_games_queue.parquet" if queue else "sim_games_bigN.parquet" if big else "sim_games.parquet"))
    print("configs", n, "rows", len(out))


if __name__ == "__main__":
    main()
