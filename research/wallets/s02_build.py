"""s02: assemble the wallet-level trade table from the raw s01 pages.

For every row (one filled order side of one wallet):
  wid (int wallet id; data/wallets.parquet maps it back), event_id, unique_id, k (Winner orientation 0 home/1 draw/2 away),
  is_yes (token is Yes of k), side (wallet's own BUY/SELL), price (token price, rounded 4dp), size (shares), ts,
  taker (True = taker row = first row of its transaction), long (wallet becomes LONGER outcome k:
  BUY Yes or SELL No), ypx (Yes-equivalent price of k, 4dp), usd (size*price, cash of this row), rel_min (min to start),
  tx_makers / tx_levels (for taker rows: number of maker fills and distinct maker prices in the tx -> 'sweep').
Only pre-start rows go to data/trades_pre.parquet (deleted after s03 to save disk; re-run s02 to rebuild); data/wallet_life.parquet keeps every wallet's first/last ts and row
counts over ALL rows (incl. in-play) for 'fresh wallet' features.
"""
import os, json, glob
import pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); os.makedirs(D, exist_ok=True)
C = os.path.join(HERE, "cache", "trades")


def main():
    tokmap = json.load(open(os.path.join(HERE, "..", "scalp", "data", "tokmap.json")))
    g = pd.read_parquet(os.path.join(HERE, "..", "lead_lag", "data", "games_meta.parquet"))
    g = g[g.ok == True]
    ev2uid = dict(zip(g.event_id.astype(str), g.unique_id))
    parts = []
    wmap = {}
    tx_off = 0
    for fn in sorted(glob.glob(os.path.join(C, "*.parquet"))):
        d = pd.read_parquet(fn)
        if not len(d):
            continue
        # taker = first row of each tx in API order (seq ascending == API order inside a batch)
        d = d.sort_values("seq")
        d["taker"] = ~d.duplicated("tx")
        grp = d.groupby("tx")
        d["tx_makers"] = (grp.tx.transform("size") - 1).astype(np.int16)
        d["tx_levels"] = grp.price.transform("nunique").astype(np.int16)  # rough sweep proxy
        d = d.drop_duplicates(["tx", "wallet", "token", "side", "size", "price", "taker"])
        m = d.token.map(tokmap)
        d = d[m.notna()].copy(); m = m[m.notna()]
        if not len(d):
            continue
        d["event_id"] = [str(x[0]) for x in m]; d["k"] = np.array([x[1] for x in m], dtype=np.int8)
        d["is_yes"] = np.array([x[2] for x in m]); start = np.array([x[3] for x in m])
        d["rel_min"] = ((d.ts.values - start) / 60).astype(np.float32)
        codes, uniq = pd.factorize(d.tx)
        d["tx"] = codes.astype(np.int64) + tx_off; tx_off += len(uniq)
        for w in d.wallet.unique():
            if w not in wmap:
                wmap[w] = len(wmap)
        d["wid"] = d.wallet.map(wmap).astype(np.int32)
        d["buy"] = d.side == "BUY"
        parts.append(d.drop(columns=["wallet", "token", "side", "seq"]))
    t = pd.concat(parts, ignore_index=True); del parts
    print("rows", len(t), flush=True)
    t["side"] = np.where(t.buy, "BUY", "SELL"); t = t.drop(columns="buy")
    t["unique_id"] = t.event_id.map(ev2uid)
    t["price"] = t.price.round(4)
    t["long"] = np.where(t.is_yes, t.side == "BUY", t.side == "SELL")
    t["ypx"] = np.where(t.is_yes, t.price, 1 - t.price).round(4)
    t["usd"] = (t["size"] * t.price).astype(np.float32)
    pd.DataFrame(dict(wid=list(wmap.values()), wallet=list(wmap.keys()))).to_parquet(os.path.join(D, "wallets.parquet"))
    w = wmap
    life = t.groupby("wid").agg(first_ts=("ts", "min"), last_ts=("ts", "max"), n_rows=("ts", "size"),
                                n_events=("event_id", "nunique"), n_inplay=("rel_min", lambda x: int((x >= 0).sum())),
                                usd=("usd", "sum"))
    life.to_parquet(os.path.join(D, "wallet_life.parquet"))
    pre = t[t.rel_min < 0]
    cols = ["wid", "event_id", "unique_id", "k", "is_yes", "side", "price", "size", "ts", "taker", "long", "ypx", "usd",
            "rel_min", "tx_makers", "tx_levels", "tx"]
    pre = pre[cols].sort_values(["ts", "tx"]).reset_index(drop=True)
    pre.to_parquet(os.path.join(D, "trades_pre.parquet"))
    print("wallets", len(w), "pre rows", len(pre), "taker rows", int(pre.taker.sum()), "events", pre.event_id.nunique(),
          "in-play share", round(1 - len(pre) / len(t), 3), flush=True)
    # sanity: taker size should equal sum of maker sizes in the tx (same share units) in most txs
    tk = pre[pre.taker].set_index("tx")["size"]
    mk = pre[~pre.taker].groupby("tx")["size"].sum()
    j = pd.concat([tk, mk], axis=1, keys=["t", "m"]).dropna()
    print("taker==sum(makers) within 1%:", round(float((abs(j.t - j.m) <= 0.01 * j.t + 0.01).mean()), 3), flush=True)


if __name__ == "__main__":
    main()
