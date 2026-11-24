"""s01: catalogue every closed Polymarket SPORTS match-winner ("moneyline") market by game day.

Gamma /markets?closed=true&sports_market_types=moneyline&end_date_min/max (a game market's endDate = scheduled start),
100 per page. One row per market (soccer games have 3: home / draw / away; other sports 1 with two team tokens).
Output: data/markets.parquet
Usage: python s01_catalogue.py 2026-06-01 2026-11-18
"""
import sys, os, json, time
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://gamma-api.polymarket.com/markets"


def row(m):
    e = (m.get("events") or [{}])[0]
    try:
        tok = json.loads(m.get("clobTokenIds") or "[]"); op = json.loads(m.get("outcomePrices") or "[]")
        oc = json.loads(m.get("outcomes") or "[]")
    except ValueError:
        return None
    if len(tok) != 2 or len(op) != 2:
        return None
    return dict(cond=m["conditionId"], event_id=str(e.get("id")), slug=e.get("slug"), title=e.get("title"),
                series=e.get("seriesSlug"), question=m.get("question"), group=m.get("groupItemTitle"),
                out0=oc[0] if oc else None, out1=oc[1] if len(oc) > 1 else None, tok0=tok[0], tok1=tok[1],
                res0=float(op[0]), res1=float(op[1]), volume=float(m.get("volumeNum") or 0),
                start=m.get("gameStartTime") or e.get("startTime") or m.get("endDate"),
                fees=bool(m.get("feesEnabled")), fee_type=m.get("feeType"), ev_volume=float(e.get("volume") or 0))


def window(a, b, s):
    """Markets with endDate in [a, b] (pd.Timestamp). The API refuses offsets above ~2000: such a window is halved."""
    out, off = [], 0
    if a > b:
        return out
    while off <= 1900:
        for i in range(6):
            try:
                r = s.get(URL, params=dict(closed="true", sports_market_types="moneyline", limit=100, offset=off, order="endDate",
                                           ascending="true", end_date_min=a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                           end_date_max=b.strftime("%Y-%m-%dT%H:%M:%SZ")), timeout=60)
                if r.status_code == 200:
                    page = r.json(); break
                if r.status_code == 422:
                    page = []; break
            except requests.RequestException:
                pass
            time.sleep(2 ** i)
        else:
            raise RuntimeError(f"gamma failed {a} {off}")
        out += [x for x in map(row, page) if x]
        if len(page) < 100:
            return out
        off += 100
    if b - a <= pd.Timedelta(seconds=1):      # thousands of markets ending in the same second (golf match-ups): keep 2,000
        return out
    mid = a + (b - a) / 2
    mid = mid.floor("s")
    return window(a, mid, s) + window(mid + pd.Timedelta(seconds=1), b, s)


def day(d):
    return window(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(hours=24) - pd.Timedelta(seconds=1), requests.Session())


if __name__ == "__main__":
    days = [str(x.date()) for x in pd.date_range(sys.argv[1], sys.argv[2])]
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(day, days)):
            rows += r
            if i % 20 == 0:
                print(days[i], len(rows), flush=True)
    d = pd.DataFrame(rows).drop_duplicates("cond")
    d["start"] = pd.to_datetime(d.start, utc=True, errors="coerce")
    d = d.dropna(subset=["start"])
    d["kick_ts"] = (d.start - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    d.to_parquet(os.path.join(HERE, "data", "markets.parquet"))
    print(len(d), d.event_id.nunique(), "events")
