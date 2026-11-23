"""Swing research, step 3: does the chosen rule (x0 = buy at floor(mid), d1 = sell 1c higher, model threshold) survive
  - delay: the order starts 15 min after the decision (guards against midpoints that already contain the next minutes)
  - q200: a fill / target counts only once >= 200 shares traded through our level (about $100 orders)
  - stale Winner: Winner features taken 3 h old (production may only refresh Winner odds every 3 h)
  - by month, by direction, by outcome type, by time to kickoff
Same model recipe as s02 (fit on all train), same threshold. Output: out/robust.json
"""
import os, json, numpy as np, pandas as pd
import s02_model as S

HERE = os.path.dirname(os.path.abspath(__file__))
X, D, THR = 0, 1, 0.5


def prep(name):
    df = pd.read_parquet(os.path.join(HERE, "data", name + ".parquet"))
    df["gap"] = df.w - df.mid
    df["spread_px"] = df.lastbuy_gap - df.lastsell_gap
    df["is_draw"] = (df.k == 1).astype(int); df["tier_top"] = df.tier_top.astype(int)
    df["hour"] = pd.to_datetime(df.t, unit="s").dt.hour
    df["tickpos"] = df.mid * 100 - np.floor(df.mid * 100 + 1e-6)
    df["split"] = np.where(df.kick < S.TRAIN_END, "train", np.where(df.kick < S.VAL_END, "val", "test"))
    return df.sort_values(["uid", "k", "dir", "t"]).reset_index(drop=True)


def stale(df, hours=3):
    """Winner features as they were `hours` earlier (same game/outcome/direction series, 30-min steps)."""
    df = df.copy(); g = df.groupby(["uid", "k", "dir"], sort=False)
    n = hours * 2
    for c in ("w", "dw_1h", "dw_3h", "dw_6h", "dw_24h"):
        df[c] = g[c].shift(n)
    df["gap"] = df.w - df.mid
    return df


def run(df, tag, out, extra=False):
    tr = df[df.split == "train"]
    m = S.fit(tr, f"pnl_x{X}_d{D}")
    for sp in ("val", "test"):
        s = df[df.split == sp].reset_index(drop=True)
        sc = m.predict(s[S.FEATS]); idx = S.policy(s, sc, THR, X)
        r = S.stats(s, idx, X, D); out[f"{tag}_{sp}"] = r
        print(f"{tag:10s} {sp:5s} fills {r['fills']:5d} roi {r['roi']*100:6.2f}% ci {np.round(np.array(r['ci'])*100,2)} hit {r['hit']:.2f} c/sh {r['c_per_share']:.2f} fills/mo {r.get('fills_per_month',0):.0f}", flush=True)
        if extra:
            f = s.iloc[idx]; f = f[f[f"fill_x{X}"]].copy()
            f["pnl"] = f[f"pnl_x{X}_d{D}"]; f["cost"] = f[f"L_x{X}"]
            f["month"] = f.kick.dt.to_period("M").astype(str)
            f["hk"] = pd.cut(f.mtk / 60, [0, 3, 6, 12, 24, 48])
            f["side"] = np.where(f.is_draw == 1, "draw", "team") + np.where(f.dir == 1, "_yes", "_no")
            for col in ("month", "side", "hk"):
                t = f.groupby(col, observed=True).agg(n=("pnl", "size"), pnl=("pnl", "sum"), cost=("cost", "sum"))
                t["roi%"] = (t.pnl / t.cost * 100).round(2)
                print(t[["n", "roi%"]].T.to_string(), flush=True)
                out[f"{tag}_{sp}_by_{col}"] = t[["n", "roi%"]].reset_index().astype(str).to_dict("records")
            out[f"{tag}_{sp}_trades"] = f[["uid", "k", "dir", "t", "mtk", "mid", "w", "L_x0", "pnl", "kick"]].astype(str).to_dict("records")
    return m


def main():
    out = {}
    base = prep("orders")
    run(base, "base", out, extra=True)
    run(stale(base), "staleW3h", out)
    for nm in ("orders_delay", "orders_q200"):
        if os.path.exists(os.path.join(HERE, "data", nm + ".parquet")):
            run(prep(nm), nm.replace("orders_", ""), out)
    json.dump(out, open(os.path.join(HERE, "out", "robust.json"), "w"), default=str, indent=1)


if __name__ == "__main__":
    main()
