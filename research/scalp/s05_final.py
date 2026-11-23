"""Step 5: finalists -> validation -> ONE evaluation on the untouched TEST period (kickoff >= 2026-10-01).

Protocol (fixed before looking at test):
  * source of truth = real-trade fills, mode 'through' (s04 / s03 trades grid)
  * two tracks: A "closed before kickoff" (exit = taker sell at kickoff-5 / after 3 h / after 12 h if the limit sell
    has not filled) and B "anything goes" (also allows holding unfilled shares to settlement)
  * finalist per family and track = best train lower bound (mean - 2 SE, game-clustered)
  * a finalist is "validated" if its validation return is > 0 AND its validation lower bound > 0
  * every finalist (validated or not) is evaluated on test exactly once, here.
Outputs: out/final_summary.csv, out/final_trades.parquet (per trade, all splits), out/final_monthly.csv,
         out/final_sensitivity.csv, out/byd.csv, out/fill_rates.csv
"""
import os, json
import numpy as np, pandas as pd
from common import load_all, HERE, boot_ci, EXITS

S, M = load_all()
S["fav_ok"] = (S.x0 >= 0.15) & (S.x0 <= 0.85)
S["mid_px"] = (S.x0 >= 0.25) & (S.x0 <= 0.75)
S["far"] = S.htk > 12
S["vol_mid"] = (S.series_vol >= 1e4) & (S.series_vol <= 2.5e5)
S["nontop"] = S.tier != "top"
S["coh"] = S.psum.between(0.98, 1.05)
FMAP = {"time": {">12h": "far", "<=12h": "~far"}, "price": {"15-85c": "fav_ok", "25-75c": "mid_px"},
        "vol": {"series 10k-250k": "vol_mid"}, "tier": {"non-top": "nontop"}, "book": {"coherent": "coh"}}
T = pd.read_parquet(os.path.join(HERE, "data", "sims_trades.parquet"))
T["filled"] = T.filled.astype(bool)
Mid = M[M.c == 1].drop(columns="uid")
Mid0 = M[M.c == 0].drop(columns="uid")
G = pd.read_csv(os.path.join(HERE, "out", "grid_trades.csv"), low_memory=False)
G["sub"] = G["sub"].astype(str)
HS = {"<10k": 0.02, "10-50k": 0.01, "50-250k": 0.0075, ">250k": 0.005}
S["hs"] = np.where(S.poly_volume < 1e4, 0.02, np.where(S.poly_volume < 5e4, 0.01, np.where(S.poly_volume < 2.5e5, 0.0075, 0.005)))


def sig_mask(cfg):
    q = (S.fam == cfg["fam"]) & (S.thr == cfg["thr"]) & S.fav_ok
    for f in ("time", "price", "vol", "tier", "book"):
        col = FMAP[f].get(cfg[f])
        if col:
            q &= ~S[col.strip("~")] if col.startswith("~") else S[col]
    if str(cfg["sub"]) not in ("None", "nan"):
        kk, dd = eval(str(cfg["sub"]))
        q &= (S.k == kk) & (S.dir == dd)
    return S[q]


def trades_for(cfg, src=T, mode="through", d=None, ent=None):
    s = sig_mask(cfg)
    m = src
    if "mode" in m.columns:
        m = m[m["mode"] == mode]
    m = m[(m.ent == (ent or cfg["ent"]))]
    attempts = m[m.sid.isin(s.sid)].groupby("sid").filled.any()      # entry orders placed (incl. unfilled)
    f = m[m.filled & (m.d == (cfg["d"] if d is None else d))].merge(
        s[["sid", "uid", "split", "t_utc", "day", "month", "mtk", "x0", "hs", "k", "dir", "kick"]], on="sid")
    f["hit"] = f.hit.astype(bool)
    f["pnl"] = f[cfg["exit"]].astype(float)
    f["ret"] = f.pnl / f.pe
    return f, attempts


def stats(f, attempts, cfg, label):
    out = dict(label=label, n_orders=len(attempts), entry_fill=float(attempts.mean()) if len(attempts) else np.nan,
               n=len(f), games=f.uid.nunique())
    if not len(f):
        return out
    lo, hi = boot_ci(f.assign(one=1.0), "ret")
    out.update(exit_fill=float(f.hit.mean()), closed_pre_kick=float(f.hit.mean()) if cfg["exit"] == "pnl_hold" else 1.0,
               held=float((~f.hit).mean()) if cfg["exit"] == "pnl_hold" else 0.0,
               ret=float(f.ret.mean()), ret_lo=lo, ret_hi=hi, cents=float(f.pnl.mean() * 100),
               cents_median=float(f.pnl.median() * 100), win=float((f.pnl > 0).mean()),
               worst_cents=float(f.pnl.min() * 100), worst_ret=float(f.ret.min()),
               avg_loss_unfilled_cents=float(f[~f.hit].pnl.mean() * 100) if (~f.hit).any() else np.nan,
               usd_per_100=float(f.ret.mean() * 100))
    daily = f.groupby("day").ret.sum() * 100                     # $100 per trade
    out["worst_day_usd"] = float(daily.min())
    out["days"] = int(daily.size)
    span_m = max((f.t_utc.max() - f.t_utc.min()).days / 30.4, 1)
    out["trades_per_month"] = len(f) / span_m
    out["usd_per_month_at_100"] = out["trades_per_month"] * out["usd_per_100"]
    # capital: open positions over time ($100 each) from entry-order time to exit (target hit or kickoff)
    st = f.t_utc
    en = f.t_utc + pd.to_timedelta(np.where(f.hit, f.me + f.t_hit, f.mtk), unit="m")
    ev = pd.concat([pd.Series(1, index=st), pd.Series(-1, index=en)]).sort_index()
    out["peak_open_positions"] = int(ev.cumsum().max())
    out["capital_usd_at_100"] = out["peak_open_positions"] * 100
    return out


if __name__ == "__main__":
    key = ["fam", "thr", "ent", "d", "exit", "time", "price", "vol", "tier", "book", "sub"]
    tracks = {"A_closed_pre_kickoff": ["pnl_kick", "pnl_t180", "pnl_t720"], "B_any": EXITS}
    rows, alltr, monthly, sens, byd, fills = [], [], [], [], [], []
    # Realistic-entry rule (from s04b_entry_reality.py, train+val only): when the 3 Yes mids sum to > 1.05 the book is
    # empty/incoherent and real takers paid 4-21c MORE than 'mid + half-spread' (and rarely traded at all), so a plain
    # 'taker' entry is only trusted in coherent books. Maker entries and 'taker_v' are checked against real prints.
    G["real_entry"] = (G.book == "coherent") | (G.ent != "taker")
    finals = []
    for track, exits in tracks.items():
        W = G[G.exit.isin(exits) & G.real_entry].dropna(subset=["lb_train"])
        finals.append((track, W.sort_values("lb_train", ascending=False).groupby("fam").head(1)))
    # for transparency: the overall best config WITHOUT the realistic-entry rule (a midpoint artefact)
    W = G.dropna(subset=["lb_train"]).sort_values("lb_train", ascending=False).head(1)
    finals.append(("Z_artefact", W))
    for track, fin in finals:
        for _, c in fin.iterrows():
            cfg = c[key].to_dict()
            name = f"{track[:1]}-{cfg['fam']}"
            f, att = trades_for(cfg)
            for sp in ("train", "val", "test"):
                r = stats(f[f.split == sp], att[att.index.isin(S[S.split == sp].sid)], cfg, name)
                rows.append(dict(track=track, name=name, split=sp, **cfg, **r))
            f["name"] = name; f["track"] = track
            alltr.append(f[["name", "track", "uid", "split", "t_utc", "day", "month", "pnl", "ret", "hit", "pe", "S", "kick"]])
            mm = f.groupby(["month", "split"]).agg(n=("ret", "size"), ret=("ret", "mean"), usd=("ret", lambda x: 100 * x.sum())).reset_index()
            mm["name"] = name; monthly.append(mm)
            # sensitivity (all splits): fill rules, size, spread x2 on taker legs
            for lab, (src, mode) in {"trades through (base)": (T, "through"), "trades touch": (T, "touch"),
                                     "trades through, $100 must trade": (T, "through100"),
                                     "midpoint c=1": (Mid, None), "midpoint c=0": (Mid0, None)}.items():
                g2, a2 = trades_for(cfg, src, mode or "through")
                for sp in ("train", "val", "test"):
                    x = g2[g2.split == sp]
                    sens.append(dict(name=name, variant=lab, split=sp, n=len(x), ret=x.ret.mean() if len(x) else np.nan,
                                     exit_fill=x.hit.mean() if len(x) else np.nan,
                                     entry_fill=float(a2[a2.index.isin(S[S.split == sp].sid)].mean()) if len(a2) else np.nan))
            extra_taker = int(str(cfg["ent"]).startswith("taker"))
            x = f.copy()
            x["pnl2"] = x.pnl - x.hs * (extra_taker + (~x.hit & (cfg["exit"] != "pnl_hold")).astype(int))
            for sp in ("train", "val", "test"):
                y = x[x.split == sp]
                sens.append(dict(name=name, variant="spread x2 on taker legs", split=sp, n=len(y),
                                 ret=(y.pnl2 / y.pe).mean() if len(y) else np.nan, exit_fill=y.hit.mean() if len(y) else np.nan))
            # profit by target d (train + val only: test is not used for anything but the finalist)
            for d in sorted(T.d.dropna().unique()):
                g3, a3 = trades_for(cfg, d=d)
                for sp in ("train", "val"):
                    y = g3[g3.split == sp]
                    byd.append(dict(name=name, d=d, split=sp, n=len(y), ret=y.ret.mean() if len(y) else np.nan,
                                    exit_fill=y.hit.mean() if len(y) else np.nan))
    R = pd.DataFrame(rows)
    piv = R.pivot_table(index="name", columns="split", values=["ret", "ret_lo"], aggfunc="first")
    valid = [n for n in piv.index if piv.loc[n, ("ret", "val")] > 0 and piv.loc[n, ("ret_lo", "val")] > 0]
    R["validated"] = R.name.isin(valid)
    R.to_csv(os.path.join(HERE, "out", "final_summary.csv"), index=False)
    pd.concat(alltr).to_parquet(os.path.join(HERE, "out", "final_trades.parquet"))
    pd.concat(monthly).to_csv(os.path.join(HERE, "out", "final_monthly.csv"), index=False)
    pd.DataFrame(sens).to_csv(os.path.join(HERE, "out", "final_sensitivity.csv"), index=False)
    pd.DataFrame(byd).to_csv(os.path.join(HERE, "out", "byd.csv"), index=False)
    # entry and exit fill rates for every entry style (all F1-F4 signals, price 15-85c, d=2c, train+val)
    q = S[S.fav_ok & S.fam.isin(["F1", "F2", "F3", "F4"]) & (S.split != "test")][["sid", "fam"]]
    for lab, (src, mode) in {"real trades (through)": (T, "through"), "real trades (touch)": (T, "touch"),
                             "midpoint c=1": (Mid, None), "midpoint c=0": (Mid0, None)}.items():
        m = src[src.sid.isin(q.sid)]
        if mode:
            m = m[m["mode"] == mode]
        ef = m.groupby(["ent", "sid"]).filled.any().groupby("ent").mean()
        xf = m[m.filled & (m.d == 0.02)].groupby("ent").hit.mean()
        for e in ef.index:
            fills.append(dict(rule=lab, ent=e, entry_fill=ef[e], exit_fill_2c=xf.get(e, np.nan)))
    pd.DataFrame(fills).to_csv(os.path.join(HERE, "out", "fill_rates.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(R[["name", "split", "thr", "ent", "d", "exit", "time", "price", "vol", "tier", "book", "sub", "n", "games",
             "entry_fill", "exit_fill", "ret", "ret_lo", "ret_hi", "cents", "win", "worst_cents", "validated"]].to_string())
    print(pd.DataFrame(sens).pivot_table(index=["name", "variant"], columns="split", values="ret").round(4).to_string())
    print(pd.DataFrame(fills).round(3).to_string())
