"""Step 6: structural arbitrage across the 3 outcomes of a 1X2 event.
 Buy all three Yes for less than $1 (after taker fees), or all three No for less than $2 (neg-risk: exactly one Yes wins,
 so three No shares pay exactly $2).
 (a) live order books (data/book_snapshots.parquet from s00, real best bid/ask): exact check.
 (b) history, midpoints: minutes where sum(mid) + 3*half-spread + fees < 1 (Yes side) or sum(mid) - 3*hs - fees > 1
     (No side), coherent books only (|sum(mid) - 1| < 0.1 to drop empty books).
 (c) history, real prints: within any 10-minute window, taker BUY prints of the three Yes(-equivalent) shares whose
     prices sum (with fees) to < 1, i.e. someone actually bought each leg that cheap.
Output out/arb.json
"""
import os, json, glob
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import HERE, LL, load_meta

FEE = 0.05


def fee(p):
    return FEE * p * (1 - p)


def hs_of(v):
    return 0.02 if v < 1e4 else 0.01 if v < 5e4 else 0.0075 if v < 2.5e5 else 0.005


def mids(args):
    uid, vol = args
    z = np.load(os.path.join(LL, "data", "series", f"{uid}.npz"))
    P = z["P"].astype(float)
    P = P[~np.isnan(P).any(1)]
    if not len(P):
        return None
    hs = hs_of(vol)
    s = P.sum(1)
    coh = np.abs(s - 1) < 0.1
    ask = np.clip(P + hs, 0.001, 0.999); bid = np.clip(P - hs, 0.001, 0.999)
    yes = (ask + fee(ask)).sum(1) < 1
    no = ((1 - bid) + fee(1 - bid)).sum(1) < 2
    return dict(uid=uid, minutes=int(coh.sum()), yes_min=int((yes & coh).sum()), no_min=int((no & coh).sum()),
                yes_edge=float(np.max(1 - (ask + fee(ask)).sum(1)[coh])) if coh.any() else np.nan)


if __name__ == "__main__":
    out = {}
    b = pd.read_parquet(os.path.join(HERE, "data", "book_snapshots.parquet"))
    ev = b.groupby(["snap_utc", "event_id"]).filter(lambda d: len(d) == 3)
    agg = ev.groupby(["snap_utc", "event_id"]).apply(lambda d: pd.Series(dict(
        yes_cost=float((d.ask + fee(d.ask)).sum()), no_cost=float(((1 - d.bid) + fee(1 - d.bid)).sum()),
        sum_ask=float(d.ask.sum()), sum_bid=float(d.bid.sum()), min_top_ask_usd=float((d.ask_size_top * d.ask).min()))))
    out["live"] = dict(snapshots=int(agg.index.get_level_values(0).nunique()), event_books=len(agg),
                       sum_ask_lt_1=int((agg.sum_ask < 1).sum()), yes_arb_after_fee=int((agg.yes_cost < 1).sum()),
                       sum_bid_gt_1=int((agg.sum_bid > 1).sum()), no_arb_after_fee=int((agg.no_cost < 2).sum()),
                       median_sum_ask=float(agg.sum_ask.median()), p1_sum_ask=float(agg.sum_ask.quantile(0.01)),
                       best_yes_cost=float(agg.yes_cost.min()), best_no_cost=float(agg.no_cost.min()))
    print(out["live"])
    g = load_meta()
    with Pool(7) as pool:
        R = pd.DataFrame([r for r in pool.map(mids, list(zip(g.uid, g.poly_volume)), chunksize=50) if r])
    out["mid"] = dict(games=len(R), coherent_minutes=int(R.minutes.sum()), yes_arb_minutes=int(R.yes_min.sum()),
                      no_arb_minutes=int(R.no_min.sum()), games_with_yes_arb=int((R.yes_min > 0).sum()),
                      games_with_no_arb=int((R.no_min > 0).sum()))
    print(out["mid"])
    # (c) real prints
    T = pd.read_parquet(os.path.join(HERE, "data", "trades.parquet"))
    T = T[T.yes_buy].sort_values(["event_id", "ts"])
    hits, events = 0, 0
    examples = []
    for eid, d in T.groupby("event_id"):
        if d.k.nunique() < 3:
            continue
        events += 1
        ts = d.ts.to_numpy(); k = d.k.to_numpy(); px = d.yes_px.to_numpy(); sz = d["size"].to_numpy()
        last = {0: (-1e18, 1.0, 0), 1: (-1e18, 1.0, 0), 2: (-1e18, 1.0, 0)}
        found = False
        for i in range(len(ts)):
            last[k[i]] = (ts[i], px[i], sz[i])
            if all(ts[i] - last[q][0] <= 600 for q in range(3)):
                c = sum(last[q][1] + fee(last[q][1]) for q in range(3))
                if c < 1:
                    found = True
                    if len(examples) < 20:
                        examples.append(dict(event_id=eid, cost=round(c, 4), min_usd=round(min(last[q][1] * last[q][2] for q in range(3)), 2)))
                    break
        hits += found
    out["prints"] = dict(events_with_all3=events, events_with_cheap_triple=hits, examples=examples)
    print(out["prints"])
    json.dump(out, open(os.path.join(HERE, "out", "arb.json"), "w"), indent=1, default=float)
