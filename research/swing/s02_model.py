"""Swing research, step 2: learn when a buy-then-sell order is worth placing, and with which buy level / profit target.

Splits by kickoff: train < 2026-07-01, val < 2026-10-01, test >= 2026-10-01 (to 2026-11-18).
Selection uses ONLY train: models fit on kickoffs before 2026-04-01, configs/thresholds chosen on Apr-Jun 2026.
Then the chosen model is refit on all of train and scored on val (check) and test (reported once).

Policy (what production would do): walk each (game, outcome, direction) in time; when the model's predicted profit
is above the threshold and we have no live order/position there, place the order. An unfilled order frees the slot
after its 2 h window; a filled one keeps it until the position is closed (at the latest 5 min before kickoff).
"""
import os, sys, json, numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
X_C, D_C = (0, 1, 2), (1, 2, 3, 5)
TRAIN_END, VAL_END, FIT_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01"), pd.Timestamp("2026-04-01")
W = 2 * 3600
FEATS = ["mid", "w", "gap", "psum", "rank", "mtk", "dmid_1h", "dmid_3h", "dmid_6h", "dmid_24h", "dw_1h", "dw_3h",
         "dw_6h", "dw_24h", "absmove_6h", "n_tr", "lognot", "n_1h", "not_1h", "buyshare_1h", "n_6h", "not_6h",
         "buyshare_6h", "last_gap", "since_last_h", "lastbuy_gap", "lastsell_gap", "spread_px", "dir", "is_draw",
         "tier_top", "hour", "tickpos"]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=1000, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=8, seed=0)
ROUNDS = 300


def load():
    df = pd.read_parquet(os.path.join(HERE, "data", "orders.parquet"))
    df["gap"] = df.w - df.mid
    df["spread_px"] = df.lastbuy_gap - df.lastsell_gap
    df["is_draw"] = (df.k == 1).astype(int)
    df["tier_top"] = df.tier_top.astype(int)
    df["hour"] = pd.to_datetime(df.t, unit="s").dt.hour
    df["tickpos"] = df.mid * 100 - np.floor(df.mid * 100 + 1e-6)
    df["split"] = np.where(df.kick < TRAIN_END, "train", np.where(df.kick < VAL_END, "val", "test"))
    df = df.sort_values(["uid", "k", "dir", "t"]).reset_index(drop=True)
    return df


def policy(df, score, thr, x):
    """Indices of orders the policy places (see module doc)."""
    fill = df[f"fill_x{x}"].to_numpy(); t = df.t.to_numpy(); key = (df.uid + "_" + df.k.astype(str) + df.dir.astype(str)).to_numpy()
    ok = score > thr
    out, busy, last = [], -1, None
    for i in np.flatnonzero(ok):
        if key[i] != last:
            last, busy = key[i], -1
        if t[i] < busy:
            continue
        out.append(i)
        busy = t[i] + W if not fill[i] else np.inf
    return np.array(out, dtype=int)


def stats(df, idx, x, d, n_boot=1000):
    s = df.iloc[idx]
    f = s[s[f"fill_x{x}"]]
    pnl, cost = f[f"pnl_x{x}_d{d}"].to_numpy(), f[f"L_x{x}"].to_numpy()
    r = dict(orders=len(s), fills=len(f), fill_rate=len(f) / max(len(s), 1),
             hit=float(f[f"hit_x{x}_d{d}"].mean()) if len(f) else np.nan,
             c_per_share=float(pnl.mean() * 100) if len(f) else np.nan,
             roi=float(pnl.sum() / cost.sum()) if len(f) else np.nan, games=int(f.uid.nunique()))
    if len(f) > 5:
        g = f.assign(p=pnl, c=cost).groupby("uid")[["p", "c"]].sum().to_numpy()
        rng = np.random.default_rng(0); ii = rng.integers(0, len(g), (n_boot, len(g)))
        e = g[ii, 0].sum(1) / g[ii, 1].sum(1)
        r["ci"] = [float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))]
    else:
        r["ci"] = [np.nan, np.nan]
    if len(f):
        months = max((s.kick.max() - s.kick.min()).days / 30.4, 0.5)
        r["fills_per_month"] = len(f) / months
        r["usd_per_month_100"] = float((pnl / cost * 100).sum() / months)
    return r


def fit(d, target):
    return lgb.train(PARAMS, lgb.Dataset(d[FEATS], d[target] * 100), ROUNDS)


def main():
    df = load()
    print("orders", len(df), df.split.value_counts().to_dict(), flush=True)
    tr = df[df.split == "train"]
    fitp, tune = tr[tr.kick < FIT_END], tr[tr.kick >= FIT_END].reset_index(drop=True)
    res = {"baseline": {}, "tune": [], "imp": {}}
    # baselines: place every order (no model), per config, on each split
    for sp in ("train", "val", "test"):
        s = df[df.split == sp].reset_index(drop=True)
        for x in X_C:
            idx = policy(s, np.ones(len(s)), 0, x)
            for d in D_C:
                res["baseline"][f"{sp}_x{x}_d{d}"] = stats(s, idx, x, d, n_boot=200)
    print("baseline (no model) ROI:", {k: round(v["roi"] * 100, 2) for k, v in res["baseline"].items()}, flush=True)
    # tuning on Apr-Jun with models fit before Apr
    models = {}
    for x in X_C:
        for d in D_C:
            m = fit(fitp, f"pnl_x{x}_d{d}"); models[(x, d)] = m
            sc = m.predict(tune[FEATS])
            for thr in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
                idx = policy(tune, sc, thr, x)
                r = stats(tune, idx, x, d, n_boot=300); r.update(x=x, d=d, thr=thr)
                res["tune"].append(r)
                print(f"tune x{x} d{d} thr{thr}: fills {r['fills']} roi {r['roi']*100:.2f}% ci {np.round(np.array(r['ci'])*100,2)} c/sh {r['c_per_share']:.2f}", flush=True)
    T = pd.DataFrame(res["tune"])
    T["score"] = T.ci.str[0]
    T = T[T.fills >= 100]
    best = T.sort_values("score", ascending=False).iloc[0]
    print("BEST on tune:", best.to_dict(), flush=True)
    x, d, thr = int(best.x), int(best.d), float(best.thr)
    m = fit(tr, f"pnl_x{x}_d{d}")
    res["chosen"] = dict(x=x, d=d, thr=thr)
    for sp in ("val", "test"):
        s = df[df.split == sp].reset_index(drop=True)
        sc = m.predict(s[FEATS])
        for th in (sorted({0.0, thr, 0.1, 0.3}) if sp == "val" else [thr]):   # test: the chosen threshold only
            r = stats(s, policy(s, sc, th, x), x, d)
            res[f"{sp}_thr{th}"] = r
            print(f"{sp} x{x} d{d} thr{th}: fills {r['fills']} roi {r['roi']*100:.2f}% ci {np.round(np.array(r['ci'])*100,2)} hit {r['hit']:.2f} c/sh {r['c_per_share']:.2f} $/mo {r.get('usd_per_month_100', 0):.0f}", flush=True)
    res["imp"] = dict(zip(FEATS, m.feature_importance("gain").round(0).tolist()))
    m.save_model(os.path.join(HERE, "out", "model.txt"))
    json.dump(res, open(os.path.join(HERE, "out", "results.json"), "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
