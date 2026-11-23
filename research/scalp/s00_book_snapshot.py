"""Step 0: live, read-only snapshot of order books on open Polymarket soccer 1X2 events (next 7 days).
For each Yes token: best bid/ask, spread, depth within 1c/3c of the touch (CLOB POST /books, public).
Used for (a) the spread/depth model and (b) a direct structural-arbitrage check: sum of the three Yes asks < 1 ?
Appends to data/book_snapshots.parquet (one call = one snapshot). No orders, no auth, no identifying headers."""
import os, json, time, datetime as dt
import requests, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = requests.Session()
now = dt.datetime.now(dt.timezone.utc)
evs = []
for d in range(7):
    a, b = now + dt.timedelta(days=d), now + dt.timedelta(days=d + 1)
    off = 0
    while off <= 2000:
        r = S.get("https://gamma-api.polymarket.com/events", params=dict(
            tag_id=100350, closed="false", limit=100, offset=off, order="startTime", ascending="true",
            start_time_min=f"{a:%Y-%m-%dT%H:%M:%SZ}", start_time_max=f"{b:%Y-%m-%dT%H:%M:%SZ}"), timeout=30)
        page = r.json(); time.sleep(0.2)
        for e in page:
            ms = e.get("markets") or []
            if len(ms) != 3 or not all(m.get("sportsMarketType") == "moneyline" for m in ms):
                continue
            if any(m.get("closed") or not m.get("acceptingOrders", True) for m in ms):
                continue
            evs.append(e)
        if len(page) < 100:
            break
        off += 100
rows = []
for e in evs:
    ev_vol = sum(float(m.get("volumeNum") or 0) for m in e["markets"])
    for m in e["markets"]:
        ids = json.loads(m.get("clobTokenIds") or "[]")
        rows.append(dict(event_id=str(e["id"]), start=e.get("startTime"), tok=ids[0], tok_no=ids[1] if len(ids) > 1 else None,
                         draw=(m.get("groupItemTitle") or "").lower().startswith("draw"), vol=float(m.get("volumeNum") or 0),
                         ev_vol=ev_vol, tick=float(m.get("orderPriceMinTickSize") or 0.01),
                         rew_min=m.get("rewardsMinSize"), rew_spread=m.get("rewardsMaxSpread"),
                         fee_sched=json.dumps(m.get("feeSchedule"))))
df = pd.DataFrame(rows).drop_duplicates("tok")
books = {}
toks = df.tok.tolist()
for i in range(0, len(toks), 100):
    r = S.post("https://clob.polymarket.com/books", json=[{"token_id": t} for t in toks[i:i + 100]], timeout=60)
    if r.status_code == 200:
        for b in r.json():
            books[b["asset_id"]] = b
    time.sleep(0.3)


def side(levels, best, sign, width):
    v = [(float(x["price"]), float(x["size"])) for x in levels]
    return sum(s * p for p, s in v if sign * (p - best) >= -width - 1e-9)


out = []
for r in df.itertuples():
    b = books.get(r.tok)
    if not b:
        continue
    bids = b.get("bids") or []; asks = b.get("asks") or []
    if not bids or not asks:
        continue
    bb = max(float(x["price"]) for x in bids); ba = min(float(x["price"]) for x in asks)
    out.append(dict(r._asdict(), bid=bb, ask=ba,
                    bid_usd_1c=side(bids, bb, 1, 0.01), ask_usd_1c=side(asks, ba, -1, 0.01),
                    bid_usd_3c=side(bids, bb, 1, 0.03), ask_usd_3c=side(asks, ba, -1, 0.03),
                    ask_size_top=sum(float(x["size"]) for x in asks if float(x["price"]) == ba)))
o = pd.DataFrame(out)
o["snap_utc"] = pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo else pd.Timestamp(now)
o["start"] = pd.to_datetime(o.start)
o["hours_to_kick"] = (o.start - pd.Timestamp(now)).dt.total_seconds() / 3600
o = o.drop(columns=["Index"], errors="ignore")
f = os.path.join(HERE, "data", "book_snapshots.parquet")
if os.path.exists(f):
    o = pd.concat([pd.read_parquet(f), o], ignore_index=True)
o.to_parquet(f)
cur = o[o.snap_utc == o.snap_utc.max()]
ev = cur.groupby("event_id").agg(n=("tok", "size"), sum_ask=("ask", "sum"), sum_bid=("bid", "sum"), ev_vol=("ev_vol", "first"))
ev = ev[ev.n == 3]
print("markets", len(cur), "events", len(ev))
print("sum of asks quantiles", ev.sum_ask.quantile([0, .01, .05, .5]).round(3).to_dict())
print("events with sum ask < 1:", int((ev.sum_ask < 1).sum()), " sum bid > 1:", int((ev.sum_bid > 1).sum()))
print(cur.assign(spread=cur.ask - cur.bid).groupby(pd.cut(cur.ev_vol, [-1, 1e4, 5e4, 2.5e5, 1e12]), observed=True)
      .spread.describe(percentiles=[.25, .5, .75]).round(3))
