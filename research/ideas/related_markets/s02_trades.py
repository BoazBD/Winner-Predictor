"""Step 2: REAL pre-kickoff trades of every -more-markets side event (totals/BTTS/spreads/team totals/half markets) found by s01.
Events are randomly sampled (see main); the run is capped with max_batches.

data-api.polymarket.com/v2/trades?event_id=<10 ids>&limit=1000, cursor-paged (newest first; time bounds are ignored by
this endpoint, so the full history is paged). <= 3 requests/s, plain requests. Only trades with ts < kickoff are kept.
Output: data/trades_cache/<batch>.parquet -> data/side_trades.parquet with
  cond, token, side (taker BUY/SELL), price, size, ts, oi (outcome index 0/1).
Usage: python s02_trades.py [max_batches]
"""
import os, sys, time, glob
import requests, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "data", "trades_cache"); os.makedirs(C, exist_ok=True)
URL = "https://data-api.polymarket.com/v2/trades"
S = requests.Session()
_last = [0.0]


def get(params):
    for i in range(8):
        dt = time.time() - _last[0]
        if dt < 0.34:
            time.sleep(0.34 - dt)
        _last[0] = time.time()
        try:
            r = S.get(URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", 2 ** i))); continue
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        except requests.RequestException:
            time.sleep(min(60, 2 ** i))
    raise RuntimeError("trades failed")


def fetch(bi, evs, kicks):
    f = os.path.join(C, f"s{bi:05d}.parquet")
    if os.path.exists(f):
        return
    rows, cur = [], None
    while True:
        p = dict(event_id=",".join(evs), limit=1000)
        if cur:
            p["cursor"] = cur
        j = get(p)
        for x in j.get("data") or []:
            rows.append((x["condition_id"], x["token_id"], x["side"], float(x["price"]), float(x["size"]), int(x["timestamp"]),
                         int(x.get("outcome_index", -1)), x.get("event_slug")))
        cur = (j.get("pagination") or {}).get("next_cursor")
        if not cur:
            break
    df = pd.DataFrame(rows, columns=["cond", "token", "side", "price", "size", "ts", "oi", "event_slug"])
    df = df[df.ts < df.event_slug.map(kicks).fillna(4e9)].drop(columns="event_slug")
    df.to_parquet(f)


def main():
    mx = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    d = pd.read_parquet(os.path.join(HERE, "data", "side_markets.parquet"))
    ev = d.groupby("side_event_id").agg(slug=("side_slug", "first"), kick=("kick_utc", "first"), vol=("vol", "sum")).reset_index()
    ev = ev[(ev.vol > 0) & ev.slug.str.endswith("-more-markets")]
    # order: validation+test (Jul 1 - Nov 18 2026) first, then train newest-first, so a time-capped run still covers all periods
    k = pd.to_datetime(ev.kick)
    # SAMPLE: full history is ~45 s per 10 events (rate-bound), so events are randomly sampled (seed 0), interleaving
    # 10-event batches from train (< Jul 1 2026) and from Jul 1 - Nov 18 2026 so that any prefix of the run is balanced.
    a = ev[(k >= "2026-07-01") & (k < "2026-11-19")].sample(frac=1, random_state=0)
    b = ev[k < "2026-07-01"].sample(frac=1, random_state=0)
    parts = []
    for i in range(0, max(len(a), len(b)), 10):
        parts += [a.iloc[i:i + 10], b.iloc[i:i + 10]]
    ev = pd.concat(parts).reset_index(drop=True)
    kicks = dict(zip(ev.slug, (pd.to_datetime(ev.kick).astype("datetime64[s]").astype("int64"))))
    ids = ev.side_event_id.astype(str).tolist()
    nb = 0
    for bi, i in enumerate(range(0, len(ids), 10)):
        if nb >= mx:
            break
        fetch(bi, ids[i:i + 10], kicks); nb += 1
        if bi % 100 == 0:
            print(bi, len(ids) // 10, time.strftime("%H:%M:%S"), flush=True)
    t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(C, "*.parquet")))], ignore_index=True).drop_duplicates()
    t.to_parquet(os.path.join(HERE, "data", "side_trades.parquet"))
    print("trades", len(t), "conds", t.cond.nunique())


if __name__ == "__main__":
    main()
