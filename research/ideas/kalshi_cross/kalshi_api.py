"""Tiny polite client for Kalshi's public (no-auth) trade API v2.

Rate limited to <= 3 req/s with exponential backoff. Plain requests, no identifying headers.
"""
import time
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
_S = requests.Session()
_last = [0.0]
MIN_GAP = 0.34


def get(path, params=None, tries=6):
    for i in range(tries):
        gap = time.time() - _last[0]
        if gap < MIN_GAP:
            time.sleep(MIN_GAP - gap)
        _last[0] = time.time()
        try:
            r = _S.get(BASE + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** i)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** i)
            continue
        raise RuntimeError(f"{r.status_code} {path} {params} {r.text[:300]}")
    raise RuntimeError(f"gave up {path} {params}")


def paged(path, key, params=None, limit=200, max_pages=10_000):
    params = dict(params or {})
    params["limit"] = limit
    out = []
    for _ in range(max_pages):
        d = get(path, params)
        out.extend(d.get(key, []))
        cur = d.get("cursor")
        if not cur:
            break
        params["cursor"] = cur
    return out
