"""Step 1: catalogue every closed Polymarket soccer 1X2 event, day by day (Gamma API caps offset ~2000).
Raw pages cached in cache/gamma/<day>.json; parsed table written to data/poly_events.parquet.
Usage: python s01_poly_events.py [start_day] [end_day]
"""
import sys, json, time, os, re, datetime as dt
from difflib import SequenceMatcher
import requests, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "gamma"); os.makedirs(CACHE, exist_ok=True)
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
GAMMA = "https://gamma-api.polymarket.com/events"
S = requests.Session()


def get(params, tries=6):
    for i in range(tries):
        try:
            r = S.get(GAMMA, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"gamma failed {params}")


def fetch_window(a, b, f):
    if os.path.exists(f):
        return json.load(open(f))
    out, off = [], 0
    while off <= 2000:
        page = get(dict(tag_id=100350, closed="true", limit=100, offset=off, order="startTime", ascending="true",
                        start_time_min=a, start_time_max=b))
        out += page
        time.sleep(0.15)
        if len(page) < 100:
            break
        off += 100
    json.dump(out, open(f, "w"))
    return out


def fetch_day(day):
    out = fetch_window(f"{day}T00:00:00Z", f"{day}T23:59:59Z", os.path.join(CACHE, f"{day}.json"))
    if len(out) >= 1900:          # offset cap reached: re-query the day in 3-hour windows
        out = []
        for h in range(0, 24, 3):
            out += fetch_window(f"{day}T{h:02d}:00:00Z", f"{day}T{h + 2:02d}:59:59Z", os.path.join(CACHE, f"{day}_{h:02d}.json"))
    return out


def _r(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def parse(e):
    ms = e.get("markets") or []
    if len(ms) != 3 or not e.get("startTime"):
        return None
    if any(m.get("sportsMarketType") not in (None, "moneyline") for m in ms):
        return None
    title = e.get("title", "")
    if " vs. " not in title and " vs " not in title:
        return None
    t = re.sub(r"^[^:]{2,40}:\s*", "", title.replace("\xa0", " ")).strip()   # old style "EPL: A vs. B"
    parts = [x.strip() for x in re.split(r"\s+vs\.?\s+", t, maxsplit=1)]
    if len(parts) != 2:
        return None
    home, away = parts
    draws = [m for m in ms if (m.get("groupItemTitle") or m.get("question") or "").lower().strip().startswith("draw")
             or "draw" in (m.get("question") or "").lower()]
    if len(draws) != 1:
        return None
    rest = [m for m in ms if m is not draws[0]]
    a, b = (rest[0].get("groupItemTitle") or rest[0].get("question") or ""), (rest[1].get("groupItemTitle") or rest[1].get("question") or "")
    if _r(a, home) + _r(b, away) >= _r(a, away) + _r(b, home):
        mh, ma = rest
    else:
        ma, mh = rest
    g = dict(event_id=str(e["id"]), slug=e.get("slug"), title=title, poly_home=home, poly_away=away,
             series=e.get("seriesSlug") or ((e.get("series") or [{}])[0] or {}).get("slug"),
             series_title=((e.get("series") or [{}])[0] or {}).get("title"),
             start=pd.Timestamp(e["startTime"]).tz_convert("UTC").tz_localize(None),
             created=pd.Timestamp(e.get("createdAt") or e.get("startDate")).tz_convert("UTC").tz_localize(None),
             ev_volume=float(e.get("volume") or 0))
    for k, m in ((0, mh), (1, draws[0]), (2, ma)):
        toks = json.loads(m.get("clobTokenIds") or "[]")
        if not toks:
            return None
        g[f"tok{k}"] = toks[0]
        g[f"vol{k}"] = float(m.get("volumeNum") or 0)
        g[f"outcome_price{k}"] = (json.loads(m.get("outcomePrices") or "[null]") or [None])[0]
        g[f"fees{k}"] = bool(m.get("feesEnabled", False))
    return g


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-09-01"
    end = sys.argv[2] if len(sys.argv) > 2 else (dt.date.today() - dt.timedelta(days=1)).isoformat()
    rows = []
    for d in pd.date_range(start, end, freq="D"):
        day = d.strftime("%Y-%m-%d")
        evs = fetch_day(day)
        p = [g for g in map(parse, evs) if g]
        rows += p
        print(day, len(evs), len(p), flush=True)
    df = pd.DataFrame(rows).drop_duplicates("event_id")
    df.to_parquet(os.path.join(HERE, "data", "poly_events.parquet"))
    print("total", len(df))
