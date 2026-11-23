"""s02 - Fetch GDELT DOC 2.0 article lists (title, url, seendate) for each sample game, 48h before kickoff.

One query per game: both team search names AND a team-news keyword group, English sources only.
GDELT returns at most 250 records/query; seendate has 15-min granularity (time GDELT's crawler saw the URL).
Polite: >= 8 s between calls, backoff on 429 (the API is shared with other users of this IP).
Cached per game in data/gdelt/<unique_id>.json (re-runnable / resumable).
Usage: python s02_gdelt_fetch.py [limit]
"""
import os, sys, json, time, datetime as dt
import requests, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data/gdelt"); os.makedirs(OUT, exist_ok=True)
U = "https://api.gdeltproject.org/api/v2/doc/doc"
KW = '(injury OR injured OR lineup OR suspended OR "team news" OR "ruled out" OR doubt)'
S = requests.Session()
S.headers["User-Agent"] = "research-script"


def q_for(h, a):
    h, a = [x.replace("MH ", "").replace("'", "") for x in (h, a)]
    return f'("{h}" OR "{a}") {KW} sourcelang:english'


def get(params):
    wait = 10
    for _ in range(12):
        try:
            r = S.get(U, params=params, timeout=90)
        except Exception as e:
            print("  err", e, flush=True); time.sleep(wait); continue
        if r.status_code == 200 and r.text.strip().startswith("{"):
            return r.json()
        if r.status_code == 200 and not r.text.strip():
            return {"articles": []}
        if r.status_code == 200:
            print("  non-json:", r.text[:150].replace("\n", " "), flush=True)
            return {"articles": [], "error": r.text[:300]}
        time.sleep(wait); wait = min(wait * 1.6, 120)
    return None


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet"))
    s = s.sort_values("poly_volume", ascending=False)  # most-traded (and most-covered) games first
    if lim:
        s = s.head(lim)
    for i, g in s.iterrows():
        fn = os.path.join(OUT, f"{g.unique_id}.json")
        if os.path.exists(fn):
            continue
        k = g.kick_utc.to_pydatetime()
        p = dict(query=q_for(g.home_q, g.away_q), mode="artlist", format="json", maxrecords=250, sort="datedesc",
                 startdatetime=(k - dt.timedelta(hours=48)).strftime("%Y%m%d%H%M%S"), enddatetime=k.strftime("%Y%m%d%H%M%S"))
        t0 = time.time()
        js = get(p)
        if js is None:
            print(i, g.unique_id, "FAILED", flush=True); time.sleep(30); continue
        js["query"] = p
        json.dump(js, open(fn, "w"))
        print(i, g.kick_utc, g.home_q, "-", g.away_q, len(js.get("articles", [])), f"{time.time()-t0:.0f}s", flush=True)
        time.sleep(8)
