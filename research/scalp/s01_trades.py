"""Step 1: REAL Polymarket trades (data-api v2 /trades, taker side only) for every matched game in the lead/lag set.
The condition/event shapes of /v2/trades ignore time bounds, so each event's full history is paged (newest first);
only trades before the Polymarket start time are kept. 10 events per request, <= 7.5 requests/s, no identifying headers.
Events fetched: all events in the first 339 kickoff-ordered batches, then every other event with final volume
<= $1M plus a random 120 bigger ones (list in data/trades_events.parquet; games not in it are NOT "no trades").
Output: cache/trades/<batch>.parquet  ->  data/trades.parquet with
  event_id, k (0 home,1 draw,2 away in WINNER orientation), yes (token is the Yes share), side (taker BUY/SELL),
  price, size, ts (unix s), yes_px (Yes-equivalent price), yes_buy (taker bought Yes-equivalent)
"""
import os, json, time, glob, threading
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.join(HERE, "..", "lead_lag")
C = os.path.join(HERE, "cache", "trades"); os.makedirs(C, exist_ok=True)
URL = "https://data-api.polymarket.com/v2/trades"
RATE = 7.5
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
    for i in range(8):
        _wait()
        try:
            r = _sess().get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", 2 ** i)))
                continue
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        except requests.RequestException:
            time.sleep(min(60, 2 ** i))
    raise RuntimeError("trades failed")


def fetch_batch(args):
    bi, evs, starts = args
    f = os.path.join(C, f"{bi:05d}.parquet" if isinstance(bi, int) else f"{bi}.parquet")
    if os.path.exists(f):
        return 0
    rows, cur = [], None
    while True:
        p = dict(event_id=",".join(evs), limit=1000)
        if cur:
            p["cursor"] = cur
        j = get(p)
        for x in j.get("data") or []:
            rows.append((x["condition_id"], x["token_id"], x["side"], float(x["price"]), float(x["size"]),
                         int(x["timestamp"]), x.get("event_slug"), int(x.get("outcome_index", -1))))
        cur = (j.get("pagination") or {}).get("next_cursor")
        if not cur:
            break
    df = pd.DataFrame(rows, columns=["cond", "token", "side", "price", "size", "ts", "event_slug", "oi"])
    df.to_parquet(f)
    return len(df)


def download():
    g = pd.read_parquet(os.path.join(LL, "data", "games_meta.parquet"))
    g = g[g.ok == True].sort_values("kick_utc")
    evs = g.event_id.tolist(); starts = g.start.tolist()
    B = 10
    # phase 1 (first run): all events in kickoff order, 10 per request -> cache/trades/<batch>.parquet.
    # The API pages newest-first and ignores time bounds for event queries, so events with huge in-play volume
    # cost dozens of pages each; phase 1 was stopped after the batches already on disk.
    done_b = sorted(int(os.path.basename(f)[:5]) for f in glob.glob(os.path.join(C, "[0-9]*.parquet")))
    covered = set()
    for b in done_b:
        covered |= set(evs[b * B:(b + 1) * B])
    # phase 2: remaining events with final volume <= $1M, plus a random 120 of the bigger ones
    rest = g[~g.event_id.isin(covered)]
    big = rest[rest.poly_volume > 1e6].sample(120, random_state=0)
    todo = pd.concat([rest[rest.poly_volume <= 1e6], big])
    ev2 = todo.event_id.tolist()
    jobs = [(f"v2_{i // B:05d}", ev2[i:i + B], None) for i in range(0, len(ev2), B)]
    print("covered", len(covered), "phase-2 events", len(ev2), "batches", len(jobs), flush=True)
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(6) as ex:
        for i, res in enumerate(ex.map(fetch_batch, jobs)):
            n += res
            if i % 100 == 0:
                print(i, n, round(time.time() - t0), flush=True)
    fetched = sorted(covered | set(ev2))
    pd.DataFrame(dict(event_id=fetched)).to_parquet(os.path.join(HERE, "data", "trades_events.parquet"))


def assemble():
    # ---- assemble: map tokens to (event, outcome k in Winner orientation, yes/no), keep pre-start trades
    tokmap = {}
    g = pd.read_parquet(os.path.join(LL, "data", "games_meta.parquet"))
    g = g[g.ok == True]
    fmap = os.path.join(HERE, "data", "tokmap.json")
    for r in g.itertuples():
        toks = [r.tok0, r.tok1, r.tok2]
        for k, t in enumerate(toks):
            kk = (2 - k) if r.swapped else k
            tokmap[t] = (r.event_id, kk, True, pd.Timestamp(r.start).timestamp())
    # No tokens: from the gamma cache (clobTokenIds[1]); cached in data/tokmap.json
    need = set(g.event_id)
    if os.path.exists(fmap):
        tokmap = {k: tuple(v) for k, v in json.load(open(fmap)).items()}
    for fn in ([] if os.path.exists(fmap) else glob.glob(os.path.join(LL, "cache", "gamma", "*.json"))):
        for e in json.load(open(fn)):
            if str(e["id"]) not in need:
                continue
            for m in e.get("markets") or []:
                ids = json.loads(m.get("clobTokenIds") or "[]")
                if len(ids) == 2 and ids[0] in tokmap:
                    ev, kk, _, st = tokmap[ids[0]]
                    tokmap[ids[1]] = (ev, kk, False, st)
    json.dump(tokmap, open(fmap, "w"))
    parts = []
    for fn in sorted(glob.glob(os.path.join(C, "*.parquet"))):
        d = pd.read_parquet(fn)
        if len(d):
            parts.append(d)
    t = pd.concat(parts, ignore_index=True).drop_duplicates()
    m = t.token.map(tokmap)
    t = t[m.notna()].copy(); m = m[m.notna()]
    t["event_id"] = [x[0] for x in m]; t["k"] = [x[1] for x in m]; t["yes"] = [x[2] for x in m]
    t["start_ts"] = [x[3] for x in m]
    print("all trades", len(t), "in-play share", round(float((t.ts >= t.start_ts).mean()), 3))
    t = t[t.ts < t.start_ts]
    t["yes_px"] = np.where(t.yes, t.price, 1 - t.price)
    t["yes_buy"] = np.where(t.yes, t.side == "BUY", t.side == "SELL")
    t = t[["event_id", "k", "yes", "side", "price", "size", "ts", "yes_px", "yes_buy"]].sort_values(["event_id", "k", "ts"])
    t.to_parquet(os.path.join(HERE, "data", "trades.parquet"))
    print("pre-kick trades", len(t), "events", t.event_id.nunique())


if __name__ == "__main__":
    import sys
    if "assemble" not in sys.argv:
        download()
    assemble()
