"""s02b - Collect timestamped football team-news headlines from GDELT 2.0 GKG raw 15-minute files.

Why: the GDELT DOC API is rate-limited to 1 call / 5 s per IP and (shared with other users of this IP)
returned 429s for minutes at a time -> ~1 game per several minutes. The raw GKG files are static downloads
(https://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip, ~2.5 MB each, one per 15 min) that
contain, per English article: DATE (crawler 15-min slot), source, URL and Extras with <PAGE_TITLE> and
<PAGE_PRECISEPUBTIMESTAMP> (publisher's own timestamp, when present).
We stream every 15-min file covering the union of [kick-48h, kick] of the sample games, keep only rows
whose TITLE mentions a sample team (name or alias) - nothing else is stored.
Output: data/gkg/<slot>.parquet per file (resumable), merged by s02c into data/gkg_headlines.parquet.
"""
import os, io, re, sys, zipfile, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OD = os.path.join(HERE, "data/gkg"); os.makedirs(OD, exist_ok=True)
ALIAS = {"Manchester City": ["Man City"], "Manchester United": ["Man Utd", "Man United"], "Tottenham": ["Spurs"],
         "Inter Milan": ["Inter"], "Atletico Madrid": ["Atletico", "Atlético"], "Bayern Munich": ["Bayern"],
         "Barcelona": ["Barca", "Barça"], "Nottingham Forest": ["Forest"], "Aston Villa": ["Villa"],
         "PSG": ["Paris Saint-Germain", "Paris St-Germain"], "Dortmund": ["BVB"], "Leipzig": ["RB Leipzig"],
         "Gladbach": ["Monchengladbach", "Mönchengladbach"], "Cologne": ["Koln", "Köln"], "Wolves": ["Wolverhampton"],
         "Newcastle": ["Newcastle United"], "Leeds": ["Leeds United"], "Brighton": ["Brighton & Hove Albion"],
         "Athletic Bilbao": ["Athletic Club"], "Real Betis": ["Betis"], "Celta Vigo": ["Celta"],
         "Leverkusen": ["Bayer Leverkusen"], "Eintracht Frankfurt": ["Frankfurt"], "Bodo Glimt": ["Bodø/Glimt", "Bodo/Glimt"],
         "Coventry City": ["Coventry"], "Ipswich Town": ["Ipswich"], "Hull City": ["Hull"], "Sporting": ["Sporting CP", "Sporting Lisbon"],
         "Red Star Belgrade": ["Crvena Zvezda"], "Fenerbahce": ["Fenerbahçe"], "Besiktas": ["Beşiktaş"],
         "Alaves": ["Alavés"], "Malaga": ["Málaga"], "Deportivo La Coruna": ["Deportivo"], "Union Saint-Gilloise": ["Union SG", "Saint-Gilloise"]}


def names():
    s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet"))
    n = set(s.home_q) | set(s.away_q)
    out = set()
    for x in n:
        x = x.replace("MH ", "")
        out.add(x); out.update(ALIAS.get(x, []))
    return s, sorted(out, key=len, reverse=True)


S = requests.Session()
TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>")
PUB = re.compile(r"<PAGE_PRECISEPUBTIMESTAMP>(\d{14})</PAGE_PRECISEPUBTIMESTAMP>")


def fetch(slot, rx):
    fn = os.path.join(OD, f"{slot}.parquet")
    if os.path.exists(fn):
        return 0
    url = f"https://data.gdeltproject.org/gdeltv2/{slot}.gkg.csv.zip"
    for i in range(4):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                pd.DataFrame(columns=["slot", "pub", "src", "url", "title"]).to_parquet(fn); return -1
            r.raise_for_status(); break
        except Exception:
            if i == 3: return -2
    rows = []
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for line in z.read(z.namelist()[0]).decode("utf8", "replace").split("\n"):
        c = line.split("\t")
        if len(c) < 27:
            continue
        m = TITLE.search(c[26])
        if not m:
            continue
        t = m.group(1)
        if not rx.search(t):
            continue
        p = PUB.search(c[26])
        rows.append(dict(slot=c[1], pub=p.group(1) if p else None, src=c[3], url=c[4], title=t))
    pd.DataFrame(rows, columns=["slot", "pub", "src", "url", "title"]).to_parquet(fn)
    return len(rows)


if __name__ == "__main__":
    s, nm = names()
    rx = re.compile(r"\b(" + "|".join(re.escape(x) for x in nm) + r")\b")
    slots = set()
    for k in s.kick_utc:
        t = (k - pd.Timedelta(hours=48)).floor("15min")
        while t <= k:
            slots.add(t.strftime("%Y%m%d%H%M%S")); t += pd.Timedelta(minutes=15)
    slots = sorted(slots)
    if len(sys.argv) > 1:
        slots = slots[: int(sys.argv[1])]
    print(len(slots), "slots", flush=True)
    done = 0
    with ThreadPoolExecutor(4) as ex:
        for slot, n in zip(slots, ex.map(lambda x: fetch(x, rx), slots)):
            done += 1
            if done % 100 == 0 or n < 0:
                print(done, slot, n, flush=True)
    print("done")
