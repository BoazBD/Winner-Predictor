"""Step 0: which kinds of sibling events/markets Polymarket lists per soccer game, on 6 sample days (all soccer events,
gamma tag 100350). Output: out/suffix_survey.csv (day, event suffix, count) and out/suffix_market_types.csv."""
import requests, collections, re, time, os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
c = collections.Counter(); smt = collections.Counter()
for day in ["2025-10-18", "2026-01-10", "2026-04-12", "2026-08-16", "2026-10-04", "2026-11-08"]:
    off = 0
    while True:
        d = requests.get("https://gamma-api.polymarket.com/events", params=dict(tag_id=100350, limit=100, offset=off,
                         start_time_min=day + "T00:00:00Z", start_time_max=day + "T23:59:59Z"), timeout=60).json()
        for e in d:
            m = re.search(r"\d{4}-\d{2}-\d{2}(-.*)?$", e["slug"])
            suf = (m.group(1) or "(main)") if m else "(other)"
            c[(day, suf)] += 1
            for mm in e.get("markets", []):
                smt[(suf, mm.get("sportsMarketType"))] += 1
        off += 100; time.sleep(0.4)
        if len(d) < 100:
            break
pd.DataFrame([(a, b, v) for (a, b), v in c.items()], columns=["day", "suffix", "events"]).pivot(index="suffix", columns="day", values="events").fillna(0).astype(int).to_csv(os.path.join(HERE, "out", "suffix_survey.csv"))
pd.DataFrame([(a, b, v) for (a, b), v in smt.items()], columns=["suffix", "sportsMarketType", "markets"]).to_csv(os.path.join(HERE, "out", "suffix_market_types.csv"), index=False)
print(open(os.path.join(HERE, "out", "suffix_survey.csv")).read())
