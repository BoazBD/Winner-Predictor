"""s04 - Polymarket x Kalshi hedged arbitrage, measured on REAL Polymarket prints against the Kalshi top of book.

For each pre-kickoff Polymarket print (master.parquet: "a taker bought token X of outcome j at q", rel_min < -2):
  Kalshi quote = close of the last 1-min Kalshi candle that ENDED at or before the print's minute (no look-ahead);
  quote must be <= 30 min old (else unknown) and two-sided (0 < bid < ask < 1).
  A) print bought YES_j on Poly at q   -> we buy YES_j on Poly at q  + buy NO_j on Kalshi at (1 - bidK_j)
       cost = q + fP(q) + (1-bidK) + fK(bidK)          payoff exactly 1   (one of the two legs wins)
  B) print bought NO_j  on Poly at q   -> we buy NO_j on Poly at q   + buy YES_j on Kalshi at askK_j
       cost = q + fP(q) + askK + fK(askK)
  Kalshi 'not j' can also be bought as YES on the other two Kalshi outcomes (sum of their asks + fees); use the min.
  edge = 1/cost - 1  (return on capital, held to settlement).
Fees: Polymarket taker 0.05*p*(1-p) (applied to every game, conservative: many older games were fee-free);
      Kalshi taker 0.07*p*(1-p) + 0.0001 rounding allowance per contract.
Persistence check: the Kalshi quote of the NEXT minute (candle ending <= t+60s+60s) must still give edge > 0.
Size: $ notional = print size * q (Poly side we copy); Kalshi depth unknown (candles have no size) -> also report
      Kalshi volume traded in the +-5 min window as a depth proxy.
Also: Kalshi-internal 3-way basket (sum of 3 asks + fees < 1) from candles alone.
Outputs: data/arb_prints.parquet, out/arb_summary.json, out/arb_by_split.csv, out/arb_by_league.csv
"""
import glob, json
import numpy as np, pandas as pd

R4 = lambda x: np.round(np.asarray(x, dtype="float64"), 4)
fP = lambda p: 0.05 * p * (1 - p)
fK = lambda p: 0.07 * p * (1 - p) + 0.0001

mt = pd.read_parquet("data/matches.parquet")
gm = pd.read_parquet("../../lead_lag/data/games_meta.parquet", columns=["unique_id", "swapped", "kick_utc"])
gm2 = pd.read_parquet("../../lead_lag/data/games_meta.parquet", columns=["unique_id", "outcome_price0", "outcome_price1",
                                                                       "outcome_price2"])
mt = mt.merge(gm[["unique_id", "swapped"]], on="unique_id").merge(gm2, on="unique_id")
# settlement cross-check: drop games whose Kalshi and Polymarket results disagree (matching errors / rule differences)
pw = np.argmax(mt[["outcome_price0", "outcome_price1", "outcome_price2"]].astype(float).to_numpy(), 1)
kr = np.select([np.where(mt.flipped, mt.result_away, mt.result_home) == "yes", mt.result_tie == "yes",
                np.where(mt.flipped, mt.result_home, mt.result_away) == "yes"], [0, 1, 2], -1)
mt["settle_agree"] = np.where(kr >= 0, pw == kr, np.nan)
print("settlement agreement", mt.settle_agree.value_counts(dropna=False).to_dict(), flush=True)
mt = mt[mt.settle_agree != 0]
# ---- Kalshi candles -> per (unique_id, poly outcome j) quote table
cs = []
for fn in glob.glob("data/candles/*.parquet"):
    c = pd.read_parquet(fn)
    if len(c):
        cs.append(c)
c = pd.concat(cs, ignore_index=True)
tk = pd.concat([mt[["unique_id", "kt0"]].rename(columns={"kt0": "ticker"}).assign(j=0),
                mt[["unique_id", "kt1"]].rename(columns={"kt1": "ticker"}).assign(j=1),
                mt[["unique_id", "kt2"]].rename(columns={"kt2": "ticker"}).assign(j=2)])
c = c.merge(tk, on="ticker")
for col in ("bid_c", "ask_c"):
    c[col] = R4(c[col])
c = c.sort_values(["unique_id", "j", "ts"])
c["bid_c"] = c.groupby(["unique_id", "j"]).bid_c.ffill()
c["ask_c"] = c.groupby(["unique_id", "j"]).ask_c.ffill()
print("candles", len(c), "games with candles", c.unique_id.nunique(), flush=True)

# wide quote table per unique_id at each candle ts (forward-filled per outcome)
wide = c.pivot_table(index=["unique_id", "ts"], columns="j", values=["bid_c", "ask_c", "vol"], aggfunc="last")
wide.columns = [f"{a}{b}" for a, b in wide.columns]
wide = wide.reset_index().sort_values(["unique_id", "ts"])
for col in [x for x in wide.columns if x.startswith(("bid", "ask"))]:
    wide[col] = wide.groupby("unique_id")[col].ffill()
for col in [x for x in wide.columns if x.startswith("vol")]:
    wide[col] = wide[col].fillna(0)
wide["qts"] = wide.ts
wide = wide.drop(columns="ts")

# ---- Kalshi internal basket
A = wide[["ask_c0", "ask_c1", "ask_c2"]].to_numpy("float64")
basket = (A + fK(A)).sum(1)
wide["kbasket"] = basket

# ---- Polymarket prints
m = pd.read_parquet("../../edge/data/master.parquet", columns=["unique_id", "k", "yes", "q", "size", "ts", "rel_min",
                                                               "start", "fees0", "won"])
m = m[m.unique_id.isin(mt.unique_id) & (m.rel_min < -2)].merge(mt[["unique_id", "swapped", "series", "league",
                                                                     "event_ticker", "fee_type"]], on="unique_id")
m["j"] = np.where(m.swapped & (m.k != 1), 2 - m.k, m.k)
m["q"] = R4(m.q)
m = m[(m.q > 0) & (m.q < 1)]
m["minute_ts"] = (m.ts // 60) * 60                          # candle with end_period_ts <= minute start
m = m.sort_values("minute_ts")
wide = wide.sort_values("qts")
x = pd.merge_asof(m, wide, left_on="minute_ts", right_on="qts", by="unique_id", direction="backward")
x = x[x.qts.notna() & (x.minute_ts - x.qts <= 1800)]
# next-minute quote for persistence
x["next_ts"] = x.minute_ts + 60
x = x.sort_values("next_ts")
nxt = wide[["unique_id", "qts"] + [f"{a}{j}" for a in ("bid_c", "ask_c") for j in range(3)]].rename(
    columns=lambda s: s if s == "unique_id" else "n_" + s)
x = pd.merge_asof(x, nxt.sort_values("n_qts"), left_on="next_ts", right_on="n_qts", by="unique_id",
                  direction="backward")
print("prints with a Kalshi quote", len(x), "games", x.unique_id.nunique(), flush=True)


def leg(df, pre):
    """cost of Kalshi hedge leg for each row, given prefix '' or 'n_'."""
    j = df.j.to_numpy()
    B = df[[f"{pre}bid_c{i}" for i in range(3)]].to_numpy("float64")
    A = df[[f"{pre}ask_c{i}" for i in range(3)]].to_numpy("float64")
    idx = np.arange(len(df))
    bid, ask = B[idx, j], A[idx, j]
    okq = (bid > 0) & (ask < 1) & (bid < ask)
    no_direct = (1 - bid) + fK(bid)                    # buy NO_j at 1-bid
    Af = A + fK(A)
    others = Af.sum(1) - Af[idx, j]                    # buy YES on the two other outcomes
    no_cost = np.fmin(np.where(bid > 0, no_direct, np.inf), np.where(np.isnan(others), np.inf, others))
    yes_cost = np.where(ask < 1, ask + fK(ask), np.inf)
    hedge = np.where(df.yes.to_numpy(), no_cost, yes_cost)
    return np.where(okq, hedge, np.nan), bid, ask


x["hedge"], x["kbid"], x["kask"] = leg(x, "")
x["hedge_next"], _, _ = leg(x, "n_")
x["poly_cost"] = x.q + fP(x.q)
x["cost"] = x.poly_cost + x.hedge
x["edge"] = 1 / x.cost - 1
x["edge_next"] = 1 / (x.poly_cost + x.hedge_next) - 1
x["edge_nofeeP"] = 1 / (x.q + x.hedge) - 1
x["usd"] = x["size"] * x.cost                     # capital for a fully hedged copy of the print
x["pnl"] = x["size"] * (1 - x.cost)
x["kick"] = pd.to_datetime(x.start)
x["split"] = np.select([x.kick < "2026-07-01", x.kick < "2026-10-01", x.kick < "2026-11-19"],
                       ["train", "val", "test"], "live")
x["kvol5"] = x[["vol0", "vol1", "vol2"]].sum(axis=1)
x["kmid_j"] = (x.kbid + x.kask) / 2
x["poly_yes_px"] = np.where(x.yes, x.q, 1 - x.q)            # implied YES_j price of the print
x["gap"] = x.kmid_j - x.poly_yes_px
# ---- sanity filters (added after inspecting outliers):
# (1) games Kalshi settled at a 'fair price' (postponed/cancelled) are a settlement-rule divergence, not an arb
scal = set(mt[(mt.result_home == "scalar") | (mt.result_tie == "scalar") | (mt.result_away == "scalar")].unique_id)
x["kalshi_scalar"] = x.unique_id.isin(scal)
# (2) print-direction errors in master (sweeps logged as NO at the YES price): compare print's implied YES_j price
#     with the Polymarket 15-min grid midpoint of the same outcome (independent of Kalshi); flag |diff| > 8c
gr = pd.read_parquet("../../lead_lag/data/grid.parquet", columns=["unique_id", "mins_to_kick", "p0", "p1", "p2"])
gr = gr[gr.unique_id.isin(x.unique_id.unique())].merge(mt[["unique_id", "swapped"]], on="unique_id")
for i in range(3):   # grid p is in Winner orientation -> poly orientation
    src = np.where(gr.swapped & (i != 1), 2 - i, i)
    gr[f"pm{i}"] = R4(gr[["p0", "p1", "p2"]].to_numpy("float64")[np.arange(len(gr)), src])
gr["mtk"] = gr.mins_to_kick.astype("float64")
x["mtk"] = (-x.rel_min).astype("float64")
x = pd.merge_asof(x.sort_values("mtk"), gr[["unique_id", "mtk", "pm0", "pm1", "pm2"]].sort_values("mtk"), on="mtk",
                  by="unique_id", direction="forward", tolerance=20.0)      # grid point at or before the print
pm = x[["pm0", "pm1", "pm2"]].to_numpy("float64")[np.arange(len(x)), x.j.to_numpy()]
x["poly_mid_j"] = pm
x["print_bad"] = np.abs(x.poly_yes_px - pm) > 0.08 + 5e-5
x["clean"] = ~x.kalshi_scalar & ~(x.print_bad.fillna(False)) & x.poly_mid_j.notna()
print("flags: kalshi_scalar prints", int(x.kalshi_scalar.sum()), "print_bad", int(x.print_bad.sum()),
      "no grid mid", int(x.poly_mid_j.isna().sum()), flush=True)
x.to_parquet("data/arb_prints.parquet")
x = x[x.clean]

res = {}
E = x[x.edge.notna()]
res["prints_with_quote"] = int(len(E)); res["games_with_quote"] = int(E.unique_id.nunique())
res["median_abs_gap_mid_vs_print"] = float(E.gap.abs().median())
rows = []
for th in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
    for name, z in (("all", E), ("persist", E[E.edge_next > 0])):
        y = z[z.edge > th]
        rows.append(dict(th=th, filt=name, prints=len(y), games=y.unique_id.nunique(),
                         share_games=y.unique_id.nunique() / E.unique_id.nunique(), usd=y.usd.sum(),
                         profit=y.pnl.sum(),   # $ profit if we mirror the print
                         med_edge=y.edge.median(), med_min_to_kick=(-y.rel_min).median()))
tab = pd.DataFrame(rows)
print(tab.to_string())
tab.to_csv("out/arb_thresholds.csv", index=False)

sp = []
for s, z in E.groupby("split"):
    for th in (0.0, 0.01, 0.02):
        y = z[(z.edge > th) & (z.edge_next > 0)]
        sp.append(dict(split=s, th=th, games_quoted=z.unique_id.nunique(), months=z.kick.dt.to_period("M").nunique(),
                       arb_games=y.unique_id.nunique(), prints=len(y), usd=y.usd.sum(),
                       profit=y.pnl.sum(), med_edge=y.edge.median()))
sp = pd.DataFrame(sp); print(sp.to_string()); sp.to_csv("out/arb_by_split.csv", index=False)
lg = E[E.edge_next > 0].assign(arb=lambda d: d.edge > 0.01).groupby("series").agg(
    games=("unique_id", "nunique"), arb_prints=("arb", "sum")).sort_values("arb_prints", ascending=False)
lg.to_csv("out/arb_by_league.csv")
# Kalshi internal basket
kb = wide[(wide.kbasket < 1)]
res["kalshi_basket_lt1_minutes"] = int(len(kb)); res["kalshi_basket_lt1_games"] = int(kb.unique_id.nunique())
res["kalshi_basket_min"] = float(wide.kbasket.min())
json.dump(res, open("out/arb_summary.json", "w"), indent=1, default=float)
print(res)
