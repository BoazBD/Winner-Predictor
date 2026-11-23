"""s04 - (a) empirical check of the football-data early-odds collection time, (b) which venue is sharpest.

(a) For main-league games, mean |anchor_early_k - poly_k(T + h)| over outcomes for hourly offsets h in [-36, +36] around
    the assumed T (Tue/Fri 18:00 London). If the early odds were taken at time tau, the distance should be smallest
    near tau (both venues are informative and move together). Same for Winner, and for the CLOSING anchor (control).
(b) Multiclass log loss and Brier of each venue on common samples: at close (Polymarket last grid row <=20 min,
    Winner last row, Betfair Exchange closing, Pinnacle closing, average-bookmaker closing) and at T (early anchors vs
    Polymarket/Winner at T). Plus a stacked multinomial logit (fit on train, scored on val+test) to see which venue
    carries the weight when combined. CIs: 2,000 game bootstraps of log-loss differences.
Outputs out/timestamp_profile.csv, out/fig_timestamp.png, out/sharpness.csv, out/stacking.json.
"""
import os, json
import numpy as np, pandas as pd
import sys; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lead_lag", ".pylib"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(0)


def split(k):
    return np.where(k < "2026-07-01", "train", np.where(k < "2026-10-01", "val", np.where(k < "2026-11-19", "test", "live")))


def wide(P, col):
    return P.pivot(index="unique_id", columns="k", values=col)


def norm3(x):
    x = np.clip(x.astype(float), 1e-4, None)
    return x / x.sum(axis=1, keepdims=True)


def ll(prob, y):
    return -np.log(np.clip(prob[np.arange(len(y)), y], 1e-6, 1))


def boot_mean(v, n=2000):
    v = np.asarray(v); idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    P = pd.read_parquet(os.path.join(HERE, "data", "panel.parquet"))
    prof = pd.read_parquet(os.path.join(HERE, "data", "profile.parquet"))
    G = P[P.k == 0].set_index("unique_id")
    # ---------- (a) timestamp profile
    rows = []
    main_ids = G.index[G.main_league & G.bfe.notna() & (G.h_T_to_kick >= 6)]
    pr = prof[prof.unique_id.isin(main_ids)]
    for an in ["bfe", "ps", "avg", "b365", "bfec"]:
        A = P[P.unique_id.isin(main_ids)].pivot(index="unique_id", columns="k", values=an)
        A.columns = [f"a{c}" for c in A.columns]
        m = pr.merge(A, left_on="unique_id", right_index=True).dropna(subset=["a0", "p0"])
        for nm, cols in [("poly", ["p0", "p1", "p2"]), ("winner", ["w0", "w1", "w2"])]:
            pp = norm3(m[cols].to_numpy(np.float64))
            m["d"] = np.abs(pp - m[["a0", "a1", "a2"]].to_numpy()).mean(axis=1)
            # balanced panel: only games with all offsets present between -24..+12
            s = m.groupby("off_h").agg(dist=("d", "mean"), n=("d", "size")).reset_index()
            s["anchor"], s["venue"] = an, nm
            rows.append(s)
    TP = pd.concat(rows)
    TP.to_csv(os.path.join(OUT, "timestamp_profile.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for i, v in enumerate(["poly", "winner"]):
        for an in ["bfe", "ps", "avg", "bfec"]:
            s = TP[(TP.venue == v) & (TP.anchor == an) & (TP.n >= 100)]
            ax[i].plot(s.off_h, s.dist * 100, label=an)
        ax[i].axvline(0, color="k", lw=.6, ls=":")
        ax[i].set_title(f"{v} vs football-data odds"); ax[i].set_xlabel("hours from assumed T (Tue/Fri 18:00 London)")
        ax[i].set_ylabel("mean |prob diff| (pts)"); ax[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_timestamp.png"), dpi=120)
    for an in ["bfe", "ps", "avg"]:
        s = TP[(TP.venue == "poly") & (TP.anchor == an) & (TP.n >= 100)]
        if len(s):
            print(f"[{an}] poly distance minimised at offset {s.loc[s.dist.idxmin(), 'off_h']} h "
                  f"(dist {s.dist.min()*100:.2f} pts; at 0h {s[s.off_h==0].dist.mean()*100:.2f}; at -12h {s[s.off_h==-12].dist.mean()*100:.2f}; at +12h {s[s.off_h==12].dist.mean()*100:.2f})")

    # ---------- (b) sharpness
    y = G.won  # k=0 won flag is not the result; rebuild result
    W = P.pivot(index="unique_id", columns="k", values="won")
    res = W.to_numpy().argmax(axis=1); res = pd.Series(res, index=W.index)
    kick = G.kick.reindex(W.index)
    venues_close = {"poly_close": "p_C", "winner_close": "w_C", "bfe_close": "bfec", "pin_close": "psc", "avg_close": "avgc", "b365_close": "b365c"}
    venues_T = {"poly_T": "p_T", "winner_T": "w_T", "bfe_T": "bfe", "pin_T": "ps", "avg_T": "avg", "b365_T": "b365"}
    out = []

    def compare(vmap, need, tag, ids_filter=None):
        mats = {v: wide(P, c).reindex(W.index) for v, c in vmap.items()}
        ok = np.ones(len(W), bool)
        for v in need:
            ok &= mats[v].notna().all(axis=1).to_numpy()
        if ids_filter is not None:
            ok &= ids_filter
        yy = res.to_numpy()[ok]
        lls = {}
        for v in need:
            p = norm3(mats[v].to_numpy(np.float64)[ok])
            lls[v] = ll(p, yy)
            br = ((p - np.eye(3)[yy]) ** 2).sum(axis=1)
            out.append(dict(sample=tag, venue=v, n_games=int(ok.sum()), logloss=float(lls[v].mean()), brier=float(br.mean())))
        ref = need[0]
        for v in need[1:]:
            m, lo, hi = boot_mean(lls[v] - lls[ref])
            out.append(dict(sample=tag, venue=f"{v} minus {ref}", n_games=int(ok.sum()), logloss=m, brier=np.nan, ci_lo=lo, ci_hi=hi))

    late = (split(kick.astype(str).to_numpy()) != "live")
    compare(venues_close, ["poly_close", "winner_close", "bfe_close", "avg_close"], "close: all matched (Jul25-Nov26)", late)
    compare(venues_close, ["poly_close", "winner_close", "bfe_close", "avg_close", "pin_close"], "close: Pinnacle era (to Jan26)", late)
    compare(venues_T, ["poly_T", "winner_T", "bfe_T", "avg_T"], "at T: main leagues", late)
    compare(venues_T, ["poly_T", "winner_T", "bfe_T", "avg_T", "pin_T"], "at T: Pinnacle era", late)
    fee_era = (kick >= "2026-04-01").to_numpy()
    compare(venues_close, ["poly_close", "bfe_close", "winner_close"], "close: Apr-Nov 2026", late & fee_era)
    compare(venues_T, ["poly_T", "bfe_T", "winner_T"], "at T: Apr-Nov 2026", late & fee_era)
    S = pd.DataFrame(out)
    S.to_csv(os.path.join(OUT, "sharpness.csv"), index=False)
    pd.set_option("display.width", 200)
    print(S.to_string())

    # ---------- stacking: multinomial logit on log-probs, weights per venue (shared across outcomes) + draw intercept
    def stack(vmap, need, tag):
        mats = {v: norm3(wide(P, vmap[v]).reindex(W.index).to_numpy(np.float64)) for v in need}
        ok = np.ones(len(W), bool)
        for v in need:
            ok &= ~np.isnan(mats[v]).any(axis=1)
        sp = split(kick.astype(str).to_numpy())
        tr, te = ok & (sp == "train"), ok & np.isin(sp, ["val", "test"])
        if tr.sum() < 300:
            tr = ok & (sp != "live"); te = tr  # fallback (flagged)
        L = np.stack([np.log(mats[v]) for v in need], axis=2)  # n x 3 x V
        yy = res.to_numpy()

        def nll(b, idx):
            z = (L[idx] * b[:-1]).sum(axis=2); z[:, 1] += b[-1]
            z -= z.max(axis=1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return -lp[np.arange(idx.sum()), yy[idx]].mean()
        b0 = np.r_[np.ones(len(need)) / len(need), 0.0]
        r = minimize(nll, b0, args=(tr,), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-7))
        single = {v: float(-np.log(np.clip(mats[v][te][np.arange(te.sum()), yy[te]], 1e-6, 1)).mean()) for v in need}
        return dict(sample=tag, venues=need, weights=dict(zip(need + ["draw_bias"], np.round(r.x, 3).tolist())),
                    n_train=int(tr.sum()), n_eval=int(te.sum()), ll_stack_eval=float(nll(r.x, te)), ll_single_eval=single)
    st = [stack(venues_close, ["poly_close", "bfe_close", "winner_close"], "close"),
          stack(venues_T, ["poly_T", "bfe_T", "winner_T"], "at T"),
          stack(venues_close, ["poly_close", "pin_close", "bfe_close"], "close, Pinnacle era (in-sample)")]
    json.dump(st, open(os.path.join(OUT, "stacking.json"), "w"), indent=1)
    for s in st:
        print(s)


if __name__ == "__main__":
    main()
