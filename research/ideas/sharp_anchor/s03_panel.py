"""s03 - build the per-game x outcome panel.

For each matched game (s02; kept only if FD result == Polymarket resolution and score >= 1.2):
  * T = assumed collection time of the football-data EARLY odds. Notes: weekend games collected Friday afternoon,
    midweek games Tuesday afternoon. Collection day = latest Tue/Fri on or before the London kick date (Sat/Sun/Mon ->
    Fri; Wed/Thu -> Tue). Conservative (latest plausible) T = 18:00 Europe/London that day. Game kept only if
    kick - T >= 60 min and T is inside the 72 h Polymarket grid.
  * p_T / w_T: Polymarket midpoint / Winner margin-free prob at the first grid row with t_min >= T (within 20 min).
  * p_X: exit midpoint at the last grid row with mins_to_kick >= 15.  p_C / w_C: last grid row (<= 20 min to kick).
  * Real trades (edge/data/master.parquet): ask_T = first taker BUY-Yes price of outcome k in [T, T+60 min]
    (a real print at the ask); bid_X = 1 - q of the last taker BUY-No trade of outcome k in [kick-75, kick-5] min.
  * Profile: p at hourly offsets -36..+36 h around T (for the empirical collection-time check, s04).
Floats cast to float64 and rounded to 4 dp. Output data/panel.parquet (one row per game x outcome) + data/profile.parquet.
"""
import os, json
import numpy as np, pandas as pd, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
GRID = os.path.join(R, "lead_lag", "data", "grid.parquet")
MASTER = os.path.join(R, "edge", "data", "master.parquet")
ANCH = ["ps", "psc", "bfe", "bfec", "avg", "avgc", "b365", "b365c"]


def collection_T(kick_utc, hour=18):
    loc = pd.Timestamp(kick_utc).tz_localize("UTC").tz_convert("Europe/London")
    d = loc.normalize()
    wd = d.weekday()  # Mon=0
    back = {0: 3, 1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}[wd]
    day = (d - pd.Timedelta(days=back)).tz_localize(None)
    T = pd.Timestamp(day.strftime("%Y-%m-%d") + f" {hour:02d}:00").tz_localize("Europe/London").tz_convert("UTC").tz_localize(None)
    return T


def main():
    fm = pd.read_parquet(os.path.join(HERE, "data", "fd_matched.parquet"))
    n0 = len(fm)
    fm = fm[(fm.poly_res_k == fm.result_k) & (fm.score >= 1.2)].drop_duplicates("unique_id").copy()
    print("matched", n0, "kept", len(fm))
    fm["T"] = [collection_T(k) for k in fm.kick_fd.where(fm.has_time, fm.kick)]
    fm["coll_min"] = (fm["T"].values.astype("datetime64[m]").astype(np.int64))
    fm["kick_min"] = fm.kick.values.astype("datetime64[m]").astype(np.int64)
    fm["h_T_to_kick"] = (fm.kick_min - fm.coll_min) / 60
    con = duckdb.connect()
    con.register("fm", fm[["unique_id", "coll_min", "kick_min"]])
    g = con.execute(f"""select g.unique_id, g.mins_to_kick, g.t_min,
            round(g.w0::double,4) w0, round(g.w1::double,4) w1, round(g.w2::double,4) w2,
            round(g.p0::double,4) p0, round(g.p1::double,4) p1, round(g.p2::double,4) p2, fm.coll_min
            from '{GRID}' g join fm using(unique_id)""").df()
    print("grid rows", len(g))
    g = g.sort_values(["unique_id", "t_min"])
    # T row
    gt = g[(g.t_min >= g.coll_min) & (g.t_min <= g.coll_min + 20)].groupby("unique_id").head(1)
    gx = g[g.mins_to_kick >= 15].groupby("unique_id").tail(1)
    gc = g[g.mins_to_kick <= 20].groupby("unique_id").tail(1)
    gfirst = g.groupby("unique_id").t_min.min().rename("grid_t0")
    # hourly profile around T
    g["off_h"] = (g.t_min - g.coll_min) / 60.0
    prof = g[(g.off_h.abs() <= 36) & ((g.t_min - g.coll_min) % 60 < 15)].copy()
    prof["off_h"] = np.floor(prof.off_h).astype(int)
    prof = prof.groupby(["unique_id", "off_h"]).head(1)[["unique_id", "off_h", "mins_to_kick", "p0", "p1", "p2", "w0", "w1", "w2"]]
    prof.to_parquet(os.path.join(HERE, "data", "profile.parquet"))

    # real trades
    tr = con.execute(f"""select m.unique_id, m.k, m.yes, round(m.q,4) q, m.size, m.rel_min, m.ts
            from '{MASTER}' m join fm using(unique_id) where m.rel_min < 0""").df()
    tr["t_min"] = tr.ts // 60 if tr.ts.max() > 1e11 / 60 else tr.ts
    # ts units check
    if tr.ts.max() > 1e12:
        tr["t_min"] = tr.ts // 60000
    else:
        tr["t_min"] = tr.ts // 60
    tr = tr.merge(fm[["unique_id", "coll_min", "kick_min"]], on="unique_id")
    a = tr[tr.yes & (tr.t_min >= tr.coll_min) & (tr.t_min <= tr.coll_min + 60)].sort_values("t_min").groupby(["unique_id", "k"]).head(1)
    a = a.rename(columns={"q": "ask_T_trade", "size": "ask_T_size"})[["unique_id", "k", "ask_T_trade", "ask_T_size"]]
    b = tr[(~tr.yes) & (tr.rel_min >= -75) & (tr.rel_min <= -5)].sort_values("t_min").groupby(["unique_id", "k"]).tail(1).copy()
    b["bid_X_trade"] = (1 - b.q).round(4)
    b = b[["unique_id", "k", "bid_X_trade"]]
    # traded volume of outcome k in [T, kick) for capacity
    vol = tr[(tr.t_min >= tr.coll_min) & (tr.t_min < tr.kick_min)].assign(usd=lambda d: d.q * d["size"]).groupby(["unique_id", "k"]).usd.sum().rename("usd_after_T").reset_index()

    rows = []
    for nm, df in [("T", gt), ("X", gx), ("C", gc)]:
        df = df.set_index("unique_id")
        for k in range(3):
            pass
    base = fm.set_index("unique_id")
    out = []
    for k in range(3):
        d = pd.DataFrame(index=base.index)
        d["k"] = k
        for nm, df in [("T", gt), ("X", gx), ("C", gc)]:
            df = df.set_index("unique_id")
            d[f"p_{nm}"] = df[f"p{k}"]; d[f"w_{nm}"] = df[f"w{k}"]
            d[f"psum_{nm}"] = df[["p0", "p1", "p2"]].sum(axis=1)
            d[f"mtk_{nm}"] = df["mins_to_kick"]
        for an in ANCH:
            o = base[[f"{an}_0", f"{an}_1", f"{an}_2"]].astype(float)
            inv = 1 / o
            d[f"{an}"] = (inv[f"{an}_{k}"] / inv.sum(axis=1)).round(4)    # proportional margin removal
            d[f"{an}_ovr"] = inv.sum(axis=1).round(4)
        d["won"] = (base.result_k == k).astype(int)
        for c in ["event_id", "kick", "T", "h_T_to_kick", "series", "tier", "fees0", "ev_volume", "fd_div", "fd_file", "title", "has_time"]:
            d[c] = base[c]
        out.append(d.reset_index())
    P = pd.concat(out, ignore_index=True)
    P = P.merge(a, on=["unique_id", "k"], how="left").merge(b, on=["unique_id", "k"], how="left").merge(vol, on=["unique_id", "k"], how="left")
    P = P.merge(gfirst.reset_index(), on="unique_id", how="left")
    P["main_league"] = ~P.fd_file.str.startswith("new")
    P.to_parquet(os.path.join(HERE, "data", "panel.parquet"))
    G = P[P.k == 0]
    print("games", len(G), "with p_T", G.p_T.notna().sum(), "with bfe early & p_T", (G.p_T.notna() & G.bfe.notna()).sum(),
          "with ps early & p_T", (G.p_T.notna() & G.ps.notna()).sum(), "with close p", G.p_C.notna().sum())
    print("hours T->kick (games with p_T):\n", G[G.p_T.notna()].h_T_to_kick.describe().to_string())
    print("main-league games missing p_T because T before grid start:", ((G.main_league) & G.p_T.isna() & (G.grid_t0 > G['T'].values.astype('datetime64[m]').astype(np.int64))).sum())
    print("ask_T trade coverage among p_T rows:", P[P.p_T.notna() & P.bfe.notna()].ask_T_trade.notna().mean())


if __name__ == "__main__":
    main()
