"""s01 - enumerate every Kalshi soccer *match* event (1X2 'GAME' series) and its markets.

Source: public GET /historical/markets?series_ticker= (settled before the ~2-month cutoff) and /series?category=Sports (tag 'Soccer'), then GET /events?series_ticker=...&with_nested_markets=true
for every series whose ticker ends in GAME (plus a few one-off match series). No auth.
Output: data/kalshi_markets.parquet (one row per binary market), out/kalshi_series_coverage.csv
"""
import json, sys
import pandas as pd
import kalshi_api as k

series = [x for x in k.get("/series", {"category": "Sports"})["series"] if "Soccer" in (x.get("tags") or [])]
json.dump(series, open("data/soccer_series.json", "w"))
skip_words = ("1H", "2H", "WINNER", "TEAMSIN", "GOALEVERY")
game = [s for s in series if (s["ticker"].endswith("GAME") or s["title"].endswith("Game") or s["ticker"] in
        ("KXQSTARSGAME", "KXUAEPLGAME", "KXCANPLGAME", "KXCLUBFGAME", "KXFINALISSIMAGAME"))
        and not any(w in s["ticker"] for w in ("TEAMSIN", "GOALEVERY", "PULL", "SPREAD", "TOTAL")) and s["title"] != "delete"]
print(len(game), "match series", flush=True)
rows = []
for i, s in enumerate(game):
    evs = k.paged("/events", "events", {"series_ticker": s["ticker"], "with_nested_markets": "true"}, limit=200)
    for e in evs:
        for m in e.get("markets", []):
            rows.append(dict(series=s["ticker"], series_title=s["title"], fee_type=s.get("fee_type"),
                             fee_mult=s.get("fee_multiplier"), event_ticker=e["event_ticker"], ev_title=e.get("title"),
                             ev_sub=e.get("sub_title"), ticker=m["ticker"], yes_sub=m.get("yes_sub_title"),
                             status=m.get("status"), result=m.get("result"), open_time=m.get("open_time"),
                             close_time=m.get("close_time"), occ=m.get("occurrence_datetime"),
                             exp_exp=m.get("expected_expiration_time"), settle_ts=m.get("settlement_ts"),
                             volume=float(m.get("volume_fp") or 0), oi=float(m.get("open_interest_fp") or 0),
                             rules=m.get("rules_primary"), rules2=m.get("rules_secondary")))
    hist = k.paged("/historical/markets", "markets", {"series_ticker": s["ticker"]}, limit=1000)
    evt = {e["event_ticker"]: e for e in evs}
    for m in hist:
        e = evt.get(m["event_ticker"], {})
        rows.append(dict(series=s["ticker"], series_title=s["title"], fee_type=s.get("fee_type"),
                         fee_mult=s.get("fee_multiplier"), event_ticker=m["event_ticker"], ev_title=e.get("title"),
                         ev_sub=e.get("sub_title"), ticker=m["ticker"], yes_sub=m.get("yes_sub_title"),
                         status=m.get("status"), result=m.get("result"), open_time=m.get("open_time"),
                         close_time=m.get("close_time"), occ=m.get("occurrence_datetime"),
                         exp_exp=m.get("expected_expiration_time"), settle_ts=m.get("settlement_ts"),
                         volume=float(m.get("volume_fp") or 0), oi=float(m.get("open_interest_fp") or 0),
                         rules=m.get("rules_primary"), rules2=m.get("rules_secondary"), hist=True))
    print(i, s["ticker"], len(evs), "events", len(hist), "hist markets", flush=True)
df = pd.DataFrame(rows).drop_duplicates("ticker")
df["hist"] = df["hist"].fillna(False).astype(bool)
for c in ("open_time", "close_time", "occ", "exp_exp", "settle_ts"):
    df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601").dt.tz_localize(None)
df.to_parquet("data/kalshi_markets.parquet")
cov = df.groupby(["series", "series_title"]).agg(events=("event_ticker", "nunique"), first=("open_time", "min"),
                                                 last=("close_time", "max"), volume=("volume", "sum")).reset_index()
cov.sort_values("volume", ascending=False).to_csv("out/kalshi_series_coverage.csv", index=False)
print(cov.sort_values("volume", ascending=False).head(60).to_string())
print("markets", len(df), "events", df.event_ticker.nunique())
