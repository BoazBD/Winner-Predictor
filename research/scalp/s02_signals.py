"""Step 2: generate candidate signals for every matched game and simulate buy -> limit-sell execution on the 1-minute
Polymarket MIDPOINT series (research/lead_lag/data/series/*.npz; grid ends 1 minute before kickoff, nothing after).

Decisions are taken at 15-minute check times j (UTC minutes divisible by 15) using only data at or before j
(Polymarket mid at j, last Winner snapshot with scrape time <= j). Orders go in at minute j+1.

Signal families (dir=+1 buy Yes, dir=-1 buy No; X = mid of the share bought = P or 1-P):
  F1 winner_move : Winner prob moved >= thr over 6 h, Polymarket moved < half of it the same way -> follow Winner
  F2 value_gap   : Winner margin-free prob - Polymarket mid >= thr (dir-adjusted)            -> buy the cheap side
  F3 overreact   : Polymarket mid moved >= thr in the last 60 min, Winner moved < 1 pt that way in 90 min -> fade
  F4 stale_sum   : sum of the 3 Yes mids off 1 by >= thr; buy the outcome that did not move in the last hour
                   (Yes if sum < 1, No if sum > 1)
  F6 clock       : fixed check points 24 h and 6 h before kickoff, every outcome, both sides (no signal: the
                   control group / time-to-kickoff drift study)
One signal per (family, thr, outcome, side) per 6 hours.

Execution variants simulated per signal (all on the bought share's mid X, half-spread hs from the volume model):
  entry  : taker  -> pay X+hs + taker fee 0.05*p*(1-p), filled at once
           bid60 / bid240 -> resting buy at the best bid L = X-hs for 60 / 240 minutes, maker (no fee)
           mid60  -> resting buy at floor_tick(X) (inside the spread when hs >= 1c), 60 minutes
  fill rule c: a resting buy at L fills when mid <= L - c*hs; a resting sell at S fills when mid >= S + c*hs
           c=1 "book moved through our price" (conservative), c=0 "mid touched our price" (optimistic)
  target d: limit sell at entry + d, d in 1,2,3,5 cents, and 'fair' (Winner prob at signal, F1/F2 only, min +1c)
  if the target is not hit: pnl_kick (taker sell at kickoff-5 min: mid - hs - fee), pnl_hold (hold to settlement),
           pnl_t180 / pnl_t720 (taker sell 3 h / 12 h after entry, or at kickoff-5 if earlier)
Decision-time book-quality features: psum (sum of the 3 Yes mids; far from 1 = empty/incoherent book),
upd (share of minutes with a mid change over the last 6 h), age (minutes since the first Polymarket price).
Outputs: data/signals.parquet (one row per signal, features) and data/sims.parquet (signal x entry x c x d)
"""
import os, glob
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.join(HERE, "..", "lead_lag")
FEE = 0.05
COOL = 360
D_LIST = (0.01, 0.02, 0.03, 0.05, -1.0)          # -1 = sell at Winner fair value
ENTRIES = ("taker", "bid60", "bid240", "mid60")
THR = dict(F1=(0.02, 0.03, 0.05), F2=(0.03, 0.05, 0.08), F3=(0.05, 0.08, 0.12), F4=(0.03, 0.05, 0.08))


def half_spread(vol):
    return 0.02 if vol < 1e4 else 0.01 if vol < 5e4 else 0.0075 if vol < 2.5e5 else 0.005


def fee(p):
    return FEE * p * (1 - p)


def simulate(X, j, n, hs, fair, settle):
    """X: mid of the bought share (n,), signal at j. Returns list of dict rows (entry x c x d)."""
    out = []
    a = j + 1
    x0 = X[a]
    last = n - 5                      # last minute we act before kickoff
    if a >= last or np.isnan(x0):
        return out
    for ent in ENTRIES:
        for c in (1, 0):
            if ent == "taker":
                pe = min(x0 + hs, 0.999); fe = fee(pe); me = a
            else:
                L = np.floor((x0 - hs) * 100 + 1e-6) / 100 if ent.startswith("bid") else np.floor(x0 * 100 + 1e-6) / 100
                if L <= 0.0:
                    continue
                win = 60 if ent.endswith("60") else 240
                seg = X[a + 1:min(a + 1 + win, last)]
                hit = np.where(seg <= L - c * hs + 1e-9)[0]
                if not hit.size:
                    out.append(dict(ent=ent, c=c, d=np.nan, filled=False))
                    continue
                me = a + 1 + hit[0]; pe = L; fe = 0.0
            for d in D_LIST:
                if d < 0:
                    if np.isnan(fair):
                        continue
                    S = max(pe + 0.01, np.round(fair, 2))
                else:
                    S = pe + d
                if S >= 1.0:
                    continue
                seg = X[me + 1:last + 1]
                h = np.where(seg >= S + c * hs - 1e-9)[0]
                mh = me + 1 + h[0] if h.size else -1
                win_pnl = S - pe - fe

                def taker_exit(m):
                    px = max(X[m] - hs, 0.001)
                    return px - fee(px) - pe - fe
                pk = win_pnl if mh >= 0 else taker_exit(last)
                ph = win_pnl if mh >= 0 else settle - pe - fe
                r = dict(ent=ent, c=c, d=d, filled=True, me=me - j, pe=pe, S=S, hit=mh >= 0,
                         t_hit=(mh - me) if mh >= 0 else -1, pnl_kick=pk, pnl_hold=ph,
                         mark=X[n - 1] - pe - fe)
                for T in (180, 720):
                    mt = min(me + T, last)
                    r[f"pnl_t{T}"] = win_pnl if (mh >= 0 and mh <= mt) else taker_exit(mt)
                out.append(r)
    return out


def process(args):
    f, meta = args
    uid = os.path.basename(f)[:-4]
    z = np.load(f)
    t0, kick, P, wt, Q = int(z["t0"]), int(z["kick"]), z["P"].astype(np.float64), z["wt"], z["Q"].astype(np.float64)
    n = len(P)
    if n < 400:
        return None
    mins = t0 + np.arange(n)
    widx = np.searchsorted(wt, mins, side="right") - 1
    W = np.where(widx[:, None] >= 0, Q[np.clip(widx, 0, None)], np.nan)
    hs = half_spread(meta["poly_volume"])
    settle = meta["settle"]
    okP = ~np.isnan(P).any(1)
    first_p = int(np.argmax(okP)) if okP.any() else n
    sigs, sims = [], []
    taken = {}
    checks = np.where((mins % 15) == 0)[0]
    checks = checks[(checks >= first_p + 60) & (checks < n - 10)]
    # Winner change time: last minute at which Winner value changed (for age-of-Winner-quote feature)

    def add(fam, thr, k, dr, j, feat):
        key = (fam, thr, k, dr)
        if key in taken and j - taken[key] < COOL:
            return
        taken[key] = j
        X = P[:, k] if dr > 0 else 1 - P[:, k]
        fair = W[j, k] if dr > 0 else 1 - W[j, k]
        rows = simulate(X, j, n, hs, fair if fam in ("F1", "F2") else np.nan, settle[k] if dr > 0 else 1 - settle[k])
        if not rows:
            return
        sid = len(sigs)
        lo = max(first_p, j - 360)
        upd = float(np.mean(np.abs(np.diff(P[lo:j + 1, k])) > 0)) if j - lo >= 30 else np.nan
        sigs.append(dict(sid=sid, fam=fam, thr=thr, k=k, dir=dr, j=j, t_min=int(mins[j]), mtk=int(kick - mins[j]),
                         psum=float(P[j].sum()), upd=upd, age=int(j - first_p),
                         x0=float(X[j + 1]), w_fair=float(fair), gap=float(fair - X[j]), **feat))
        for r in rows:
            r["sid"] = sid
            sims.append(r)

    for j in checks:
        if not okP[j]:
            continue
        hasW = not np.isnan(W[j]).any()
        p = P[j]
        # ---- F1 / F2 need Winner
        if hasW and j >= 360 and not np.isnan(W[j - 360]).any() and okP[j - 360]:
            for k in range(3):
                dW = W[j, k] - W[j - 360, k]; dP = P[j, k] - P[j - 360, k]
                dr = 1 if dW > 0 else -1
                for thr in THR["F1"]:
                    if abs(dW) >= thr and dr * dP < 0.5 * abs(dW):
                        add("F1", thr, k, dr, j, dict(sig=float(abs(dW)), aux=float(dr * dP)))
        if hasW:
            for k in range(3):
                g = W[j, k] - p[k]
                dr = 1 if g > 0 else -1
                jj = max(j - 60, first_p)
                dP60 = dr * (P[j, k] - P[jj, k])
                for thr in THR["F2"]:
                    if abs(g) >= thr:
                        add("F2", thr, k, dr, j, dict(sig=float(abs(g)), aux=float(dP60)))
        # ---- F3 overreaction (Winner confirmation optional if Winner missing -> skip)
        if hasW and j - 90 >= first_p and okP[j - 60]:
            jw = max(j - 90, 0)
            for k in range(3):
                dP = P[j, k] - P[j - 60, k]
                if np.isnan(W[jw, k]):
                    continue
                dWk = W[j, k] - W[jw, k]
                up = 1 if dP > 0 else -1
                for thr in THR["F3"]:
                    if abs(dP) >= thr and up * dWk < 0.01:
                        add("F3", thr, k, -up, j, dict(sig=float(abs(dP)), aux=float(up * (p[k] - W[j, k]))))
        # ---- F4 stale sum
        s = p.sum() - 1
        if j - 60 >= first_p and okP[j - 60]:
            ch = np.abs(P[j] - P[j - 60])
            mover = int(np.argmax(ch))
            if ch[mover] >= 0.02:
                rest = [k for k in range(3) if k != mover]
                k = rest[int(np.argmin(ch[rest]))]
                dr = 1 if s < 0 else -1
                for thr in THR["F4"]:
                    if abs(s) >= thr:
                        add("F4", thr, k, dr, j, dict(sig=float(abs(s)), aux=float(ch[mover])))
    # ---- F6 clock
    for h in (24, 6):
        j = n - 1 - h * 60
        if j < first_p:
            continue
        j = int(j - (mins[j] % 15))
        if j < first_p or not okP[j]:
            continue
        for k in range(3):
            for dr in (1, -1):
                add("F6", float(h), k, dr, j, dict(sig=float(h), aux=float(W[j, k] if not np.isnan(W[j, k]) else np.nan)))
    if not sigs:
        return None
    S = pd.DataFrame(sigs); S["uid"] = uid
    M = pd.DataFrame(sims); M["uid"] = uid
    return S, M


if __name__ == "__main__":
    g = pd.read_parquet(os.path.join(LL, "data", "games_meta.parquet"))
    g = g[g.ok == True]
    metas = {}
    for r in g.itertuples():
        st = np.array([float(r.outcome_price0), float(r.outcome_price1), float(r.outcome_price2)])
        if r.swapped:
            st = st[::-1]
        metas[r.unique_id] = dict(poly_volume=float(r.poly_volume), settle=st)
    files = [f for f in sorted(glob.glob(os.path.join(LL, "data", "series", "*.npz"))) if os.path.basename(f)[:-4] in metas]
    jobs = [(f, metas[os.path.basename(f)[:-4]]) for f in files]
    Ss, Ms = [], []
    off = 0
    with Pool(7) as pool:
        for i, res in enumerate(pool.imap_unordered(process, jobs, chunksize=20)):
            if res is None:
                continue
            S, M = res
            S["sid"] += off; M["sid"] += off; off += len(S)
            Ss.append(S); Ms.append(M)
            if i % 2000 == 0:
                print(i, off, flush=True)
    S = pd.concat(Ss, ignore_index=True)
    M = pd.concat(Ms, ignore_index=True)
    for col in M.columns:
        if M[col].dtype == np.float64:
            M[col] = M[col].astype(np.float32)
    S.to_parquet(os.path.join(HERE, "data", "signals.parquet"))
    M.to_parquet(os.path.join(HERE, "data", "sims.parquet"))
    print("signals", len(S), "sim rows", len(M))
    print(S.groupby(["fam", "thr"]).size())
