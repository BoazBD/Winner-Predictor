"""Snapshot of current bid/ask spreads on open Polymarket soccer 1X2 markets (price history has no spread, so the
cost model uses these live spreads by volume bucket). Output data/spreads_snapshot.parquet."""
import os, json, time, datetime as dt
import requests, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = requests.Session()
now = dt.datetime.now(dt.timezone.utc)
rows = []
for d in range(4):
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
            ev_vol = sum(float(m.get("volumeNum") or 0) for m in ms)
            for m in ms:
                bb, ba = m.get("bestBid"), m.get("bestAsk")
                if bb is None or ba is None:
                    continue
                rows.append(dict(event_id=e["id"], start=e.get("startTime"), bid=float(bb), ask=float(ba),
                                 vol=float(m.get("volumeNum") or 0), ev_vol=ev_vol,
                                 liq=float(m.get("liquidityNum") or 0),
                                 draw=(m.get("groupItemTitle") or "").lower().startswith("draw")))
        if len(page) < 100:
            break
        off += 100
df = pd.DataFrame(rows)
df["start"] = pd.to_datetime(df.start)
df["hours_to_kick"] = (df.start - pd.Timestamp(now)).dt.total_seconds() / 3600
df["spread"] = df.ask - df.bid
df["mid"] = (df.ask + df.bid) / 2
df.to_parquet(os.path.join(HERE, "data", "spreads_snapshot.parquet"))
df["vol_bucket"] = pd.cut(df.ev_vol, [-1, 5e3, 2e4, 1e5, 5e5, 1e12], labels=["<5k", "5-20k", "20-100k", "100-500k", ">500k"])
print(len(df), df.event_id.nunique())
print(df.groupby("vol_bucket", observed=True).spread.describe(percentiles=[.25, .5, .75]))
