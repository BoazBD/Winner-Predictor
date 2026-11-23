"""s01: download every Polymarket trade WITH WALLETS (taker AND maker rows) for the matched soccer games.

Endpoint: GET https://data-api.polymarket.com/v2/trades?event_id=<10 ids comma-separated>&limit=1000&taker_only=false&cursor=..
  * taker_only=false returns one row per filled order side: the taker's row followed by the maker rows of the same
    transaction (verified on full events: the first row of every transaction_hash is exactly the taker row of the
    default taker-only query, 100% of 3,069 txs; one taker per tx). Every row is the wallet's OWN side/token/price.
  * newest-first paging, time filters ignored -> each event's whole history (incl. in-play) is paged.
  * limit max 1000; ~7 req/s worked without 429s. No identifying headers.
Sample: ALL games in games_meta with ok==True and kickoff >= 2025-09-01 (16.6k events; earlier months are tiny).
Output: cache/trades/<batch>.parquet (raw) ; run `s02_build.py` to assemble.
"""
import os, time, threading, sys
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cache", "trades"); os.makedirs(C, exist_ok=True)
URL = "https://data-api.polymarket.com/v2/trades"
RATE = 7.0
_lock = threading.Lock(); _next = [time.time()]
_local = threading.local()


def _sess():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def _wait():
    with _lock:
        now = time.time(); t = max(now, _next[0]); _next[0] = t + 1.0 / RATE
    if t > now:
        time.sleep(t - now)


def get(params):
    for i in range(10):
        _wait()
        try:
            r = _sess().get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, float(r.headers.get("Retry-After", 2 ** i))))
                continue
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        except requests.RequestException:
            time.sleep(min(60, 2 ** i))
    raise RuntimeError("trades failed")


def fetch_batch(args):
    bi, evs = args
    f = os.path.join(C, f"{bi:05d}.parquet")
    if os.path.exists(f):
        return 0
    rows, cur, seq = [], None, 0
    while True:
        p = dict(event_id=",".join(evs), limit=1000, taker_only="false")
        if cur:
            p["cursor"] = cur
        j = get(p)
        for x in j.get("data") or []:
            rows.append((x["transaction_hash"], x["proxy_wallet"], x["side"], x["token_id"], float(x["price"]),
                         float(x["size"]), int(x["timestamp"]), seq))
            seq += 1
        cur = (j.get("pagination") or {}).get("next_cursor")
        if not cur or not j.get("data"):
            break
    df = pd.DataFrame(rows, columns=["tx", "wallet", "side", "token", "price", "size", "ts", "seq"])
    df.to_parquet(f)
    return len(df)


def main():
    g = pd.read_parquet(os.path.join(HERE, "..", "lead_lag", "data", "games_meta.parquet"))
    g = g[(g.ok == True) & (pd.to_datetime(g.kick_utc) >= "2025-09-01")].sort_values("kick_utc")
    evs = g.event_id.astype(str).tolist()
    B = 10
    jobs = [(i // B, evs[i:i + B]) for i in range(0, len(evs), B)]
    pd.DataFrame([(b, e) for b, es in jobs for e in es], columns=["batch", "event_id"]).to_parquet(
        os.path.join(HERE, "cache", "batches.parquet"))
    print("events", len(evs), "batches", len(jobs), flush=True)
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(8) as ex:
        for i, res in enumerate(ex.map(fetch_batch, jobs)):
            n += res
            if i % 50 == 0:
                print(i, n, round(time.time() - t0), flush=True)
    print("done", n, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
