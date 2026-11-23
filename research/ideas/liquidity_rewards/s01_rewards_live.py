"""Step 1 (live, read-only): which soccer 1X2 markets pay Polymarket Liquidity Rewards, how big are the pools,
and how much competing *qualifying* liquidity sits in each book -> what share of the pool would our quote earn?

Sources (public, no auth, no identifying headers, <=3 req/s):
  - CLOB GET /rewards/markets/current  (paginated): condition_id -> rate_per_day, rewards_max_spread (c), rewards_min_size
  - gamma /events tag 100350 (soccer), next 7 days, 3-market moneyline events (1X2)
  - CLOB POST /books for the Yes and No token of every market

Scoring (docs.polymarket.com/market-makers/liquidity-rewards):
  S(v,s) = ((v-s)/v)^2 , v = max spread (c), s = distance (c) from the size-cutoff-adjusted midpoint
  Q_one = sum S*size over Yes bids + No asks ; Q_two = Yes asks + No bids
  Q_min = max(min(Q1,Q2), max(Q1,Q2)/3) if 0.10<=mid<=0.90 else min(Q1,Q2)   (c = 3)
  pool share per sample = our Q_min / sum over makers Q_min ; book sampled each minute; paid daily, $1 minimum.
The CLOB /book of the No token is an exact mirror of the Yes book (same orders, 1-p), so only the Yes book is scored.
Competitors' per-wallet Q_min is unknown (book is anonymous) -> we store aggregate Q1, Q2 and use
  central = (Q1+Q2)/2, low-competition bound = max(Q1,Q2)/3, high bound = Q1+Q2.

Appends one snapshot to data/reward_snapshots.parquet (one row per market) and saves raw reward config.
"""
import os, json, time, datetime as dt
import requests, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = requests.Session()
S.headers.update({"User-Agent": "python-requests"})
LAST = [0.0]


def get(url, method="GET", **kw):
    for a in range(6):
        w = 0.34 - (time.time() - LAST[0])
        if w > 0: time.sleep(w)
        LAST[0] = time.time()
        try:
            r = S.request(method, url, timeout=60, **kw)
        except Exception:
            time.sleep(2 ** a); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** a + 1); continue
        return r
    return None


def reward_markets():
    out, cur = {}, ""
    while True:
        r = get("https://clob.polymarket.com/rewards/markets/current", params={"next_cursor": cur})
        j = r.json()
        for d in j["data"]:
            rate = sum(float(c.get("rate_per_day") or 0) for c in d.get("rewards_config") or [])
            start = min([c.get("start_date") for c in d.get("rewards_config") or []] or [None])
            out[d["condition_id"]] = dict(rate=rate, max_spread=float(d.get("rewards_max_spread") or 0),
                                          min_size=float(d.get("rewards_min_size") or 0), rew_start=start)
        cur = j.get("next_cursor")
        if not cur or cur == "LTE=" or not j["data"]:
            break
    return out


def soccer_events(now):
    evs = []
    for d in range(7):
        a, b = now + dt.timedelta(days=d), now + dt.timedelta(days=d + 1)
        off = 0
        while off <= 2000:
            r = get("https://gamma-api.polymarket.com/events", params=dict(
                tag_id=100350, closed="false", limit=100, offset=off, order="startTime", ascending="true",
                start_time_min=f"{a:%Y-%m-%dT%H:%M:%SZ}", start_time_max=f"{b:%Y-%m-%dT%H:%M:%SZ}"))
            page = r.json()
            for e in page:
                ms = e.get("markets") or []
                if len(ms) == 3 and all(m.get("sportsMarketType") == "moneyline" for m in ms) and \
                        not any(m.get("closed") or not m.get("acceptingOrders", True) for m in ms):
                    evs.append(e)
            if len(page) < 100: break
            off += 100
    return evs


def levels(x):
    return [(round(float(l["price"]), 4), float(l["size"])) for l in x]


def adj_mid(bids, asks, min_size):
    b = [p for p, s in bids if s >= min_size]; a = [p for p, s in asks if s >= min_size]
    if not b or not a: return np.nan
    return (max(b) + min(a)) / 2


def qscore(lv, mid, v, min_size, sign):
    """sum of S*size for orders within v cents of mid on one side (sign=+1 bids below mid, -1 asks above)."""
    q = 0.0
    for p, s in lv:
        dist = round((mid - p) * sign * 100, 4)
        if s >= min_size and 0 <= dist < v:   # docs: orders strictly inside max spread
            q += ((v - dist) / v) ** 2 * s
    return q


def main():
    now = dt.datetime.now(dt.timezone.utc)
    rw = reward_markets()
    json.dump(rw, open(os.path.join(HERE, "data", f"reward_config_{now:%Y%m%d_%H%M}.json"), "w"))
    evs = soccer_events(now)
    rows = []
    for e in evs:
        ev_vol = sum(float(m.get("volumeNum") or 0) for m in e["markets"])
        for m in e["markets"]:
            ids = json.loads(m.get("clobTokenIds") or "[]")
            r = rw.get(m["conditionId"], {})
            rows.append(dict(event_id=str(e["id"]), title=e.get("title"), slug=e.get("slug"),
                             series=(e.get("series") or [{}])[0].get("slug") if e.get("series") else None,
                             start=e.get("startTime"), cond=m["conditionId"], tok=ids[0], tok_no=ids[1],
                             draw=(m.get("groupItemTitle") or "").lower().startswith("draw"),
                             vol=float(m.get("volumeNum") or 0), ev_vol=ev_vol, liq=float(m.get("liquidityNum") or 0),
                             rate=r.get("rate", 0.0), rew_start=r.get("rew_start"),
                             max_spread=r.get("max_spread", float(m.get("rewardsMaxSpread") or 0)),
                             min_size=r.get("min_size", float(m.get("rewardsMinSize") or 0)),
                             fee_sched=json.dumps(m.get("feeSchedule"))))
    df = pd.DataFrame(rows).drop_duplicates("tok")
    toks = df.tok.tolist() + df.tok_no.tolist()
    books = {}
    for i in range(0, len(toks), 100):
        r = get("https://clob.polymarket.com/books", method="POST", json=[{"token_id": t} for t in toks[i:i + 100]])
        if r is not None and r.status_code == 200:
            for b in r.json(): books[b["asset_id"]] = b
    out = []
    for r in df.itertuples(index=False):
        by, bn = books.get(r.tok), books.get(r.tok_no)
        if not by: continue
        yb, ya = levels(by.get("bids") or []), levels(by.get("asks") or [])
        nb, na = (levels(bn.get("bids") or []), levels(bn.get("asks") or [])) if bn else ([], [])
        d = r._asdict()
        d.update(bid=max([p for p, _ in yb], default=np.nan), ask=min([p for p, _ in ya], default=np.nan),
                 no_bid=max([p for p, _ in nb], default=np.nan), no_ask=min([p for p, _ in na], default=np.nan),
                 n_lv=len(yb) + len(ya))
        # are the Yes and No books mirrors (same orders) ? -> if yes, count only Yes book
        mirror = bool(nb) and abs((1 - d["no_ask"]) - d["bid"]) < 5e-5 and abs((1 - d["no_bid"]) - d["ask"]) < 5e-5 and \
            abs(sum(s for _, s in yb) - sum(s for _, s in na)) < 1e-6
        d["mirror"] = mirror
        v, ms = r.max_spread, r.min_size
        mid = adj_mid(yb, ya, ms)
        d["mid_adj"] = mid
        if np.isfinite(mid) and v > 0:
            q1 = qscore(yb, mid, v, ms, +1); q2 = qscore(ya, mid, v, ms, -1)
            if False:  # verified 2026-11-21: the No book is an exact mirror of the Yes book -> never add it
                q1 += qscore([(round(1 - p, 4), s) for p, s in na], mid, v, ms, +1)
                q2 += qscore([(round(1 - p, 4), s) for p, s in nb], mid, v, ms, -1)
            d.update(Q1=q1, Q2=q2)
            # USD depth inside the reward band, for capacity
            d["usd_in_band"] = sum(p * s for p, s in yb if (mid - p) * 100 < v and s >= ms) + \
                sum((1 - p) * s for p, s in ya if (p - mid) * 100 < v and s >= ms)
        out.append(d)
    o = pd.DataFrame(out)
    o["snap_utc"] = pd.Timestamp(now).tz_convert(None)
    o["start"] = pd.to_datetime(o.start, utc=True).dt.tz_convert(None)
    o["hours_to_kick"] = (o.start - o.snap_utc).dt.total_seconds() / 3600
    f = os.path.join(HERE, "data", "reward_snapshots.parquet")
    if os.path.exists(f):
        o = pd.concat([pd.read_parquet(f), o], ignore_index=True)
    o.to_parquet(f)
    cur = o[o.snap_utc == o.snap_utc.max()]
    print(now, "reward markets total", len(rw), "soccer 1X2 markets", len(cur), "with rate>0", int((cur.rate > 0).sum()),
          "mirror books", round(cur.mirror.mean(), 3))
    print(cur.groupby(cur.rate > 0).agg(n=("tok", "size"), rate_sum=("rate", "sum"), rate_med=("rate", "median"),
                                         Q1_med=("Q1", "median"), vol_med=("ev_vol", "median")))


if __name__ == "__main__":
    main()
