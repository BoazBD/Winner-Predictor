"""s02: for a sample of markets from data/markets.parquet, download (a) maker prints of the last 4 h before kickoff
(data-api /v2/trades, taker_only=false; newest first, so in-play pages are skipped through, max PAGES pages) and
(b) the 1-minute price history of token0 for the same 4 h (CLOB prices-history, fidelity=1).
Output: cache/<cond>.npz with tape columns ts, ypx (token0 price), long (maker got longer token0), size, tx-id,
and price series pt, pp. Markets whose pre-kick window was not reached within PAGES pages are flagged (ok=0).
Usage: python s02_download.py <sample.parquet>
"""
import os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cache"); os.makedirs(C, exist_ok=True)
TR = "https://data-api.polymarket.com/v2/trades"; PH = "https://clob.polymarket.com/prices-history"
RATE, PAGES, PRE = 7.0, 40, 4 * 3600
_lock = threading.Lock(); _next = [time.time()]; _loc = threading.local()


def get(url, params):
    if not hasattr(_loc, "s"):
        _loc.s = requests.Session()
    for i in range(8):
        with _lock:
            now = time.time(); t = max(now, _next[0]); _next[0] = t + 1 / RATE
        if t > now:
            time.sleep(t - now)
        try:
            r = _loc.s.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (429,) and r.status_code < 500:
                return None
        except requests.RequestException:
            pass
        time.sleep(min(30, 2 ** i))
    return None


def one(row):
    f = os.path.join(C, f"{row.cond}.npz")
    if os.path.exists(f):
        return 0
    k0 = int(row.kick_ts); a = k0 - PRE
    rows, cur, ok = [], None, 0
    for _ in range(PAGES):
        j = get(TR, dict(condition_id=row.cond, limit=1000, taker_only="false", **({"cursor": cur} if cur else {})))
        data = (j or {}).get("data") or []
        rows += [(int(x["timestamp"]), x["transaction_hash"], int(x["outcome_index"]), float(x["price"]), x["side"] == "BUY",
                  float(x["size"])) for x in data]
        cur = ((j or {}).get("pagination") or {}).get("next_cursor")
        if j is None:
            break
        if not data or not cur or min(int(x["timestamp"]) for x in data) < a:
            ok = 1; break
    d = pd.DataFrame(rows, columns=["ts", "tx", "oi", "px", "buy", "size"])
    if len(d):
        d["taker"] = ~d.duplicated("tx")
        d = d[~d.taker & (d.ts >= a) & (d.ts < k0)]
    yes = d.oi.to_numpy() == 0
    ypx = np.round(np.where(yes, d.px, 1 - d.px), 4)
    long = np.where(yes, d.buy, ~d.buy.astype(bool))
    txid = pd.factorize(d.tx)[0] if len(d) else np.zeros(0, int)
    h = (get(PH, dict(market=row.tok0, startTs=a - 600, endTs=k0, fidelity=1)) or {}).get("history") or []
    np.savez_compressed(f, ts=d.ts.to_numpy(np.int64), ypx=ypx.astype(np.float64), long=long.astype(bool),
                        size=d["size"].to_numpy(np.float32), tx=txid.astype(np.int32),
                        pt=np.array([x["t"] for x in h], np.int64), pp=np.array([x["p"] for x in h], np.float64), ok=ok)
    return 1


if __name__ == "__main__":
    s = pd.read_parquet(sys.argv[1]); t = time.time(); n = 0
    with ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(one, s.itertuples())):
            n += r
            if i % 200 == 0:
                print(i, len(s), round(time.time() - t), "s", flush=True)
    print("done", n, round(time.time() - t), "s")
