"""Step 4: minute-level Polymarket price history (CLOB prices-history, fidelity=1) for the 3 "Yes" tokens of every
matched event, from 7 days before kickoff to kickoff. Cached per token in cache/prices/<token>.json.
Polite: ~7 requests/s total, retries with backoff."""
import os, json, time, threading
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "cache", "prices"); os.makedirs(C, exist_ok=True)
URL = "https://clob.polymarket.com/prices-history"
RATE = 8.0
_lock = threading.Lock(); _next = [time.time()]
_local = threading.local()


def _sess():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def _wait():
    with _lock:
        now = time.time()
        t = max(now, _next[0]); _next[0] = t + 1.0 / RATE
    if t > now:
        time.sleep(t - now)


def fetch(tok, start_ts, end_ts):
    f = os.path.join(C, f"{tok[-40:]}.json")
    if os.path.exists(f):
        return "cached"
    for i in range(7):
        _wait()
        try:
            r = _sess().get(URL, params=dict(market=tok, startTs=start_ts, endTs=end_ts, fidelity=1), timeout=30)
            if r.status_code == 200:
                h = [x for x in (r.json().get("history") or []) if start_ts <= x["t"] <= end_ts]
                json.dump(dict(tok=tok, s=start_ts, e=end_ts, h=h), open(f, "w"))
                return len(h)
            if r.status_code in (400, 404):
                json.dump(dict(tok=tok, s=start_ts, e=end_ts, h=[], err=r.status_code), open(f, "w"))
                return 0
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** i))
    return "fail"


if __name__ == "__main__":
    m = pd.read_parquet(os.path.join(HERE, "data", "matches.parquet"))
    jobs = []
    for r in m.itertuples():
        end = int(pd.Timestamp(r.start).tz_localize("UTC").timestamp())
        for k in range(3):
            jobs.append((getattr(r, f"tok{k}"), end - 7 * 86400, end))
    print("tokens", len(jobs), flush=True)
    done = 0; t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(lambda j: fetch(*j), jobs):
            done += 1
            if done % 1000 == 0:
                print(done, res, round(time.time() - t0), flush=True)
    print("done", done)
