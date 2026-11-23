"""s07 - LIVE snapshot (forward evidence, no orders): full order books on both venues for upcoming matched games.

1. Polymarket: gamma /events?closed=false&tag_slug=soccer (moneyline 3-market events, kickoff in the next 72h).
2. Kalshi: open soccer match events (GET /events?status=open&series_ticker=..., GAME series with volume).
3. Match by team names (s02 sim) and kickoff date.
4. Books: Polymarket CLOB GET /book?token_id= (YES token of each outcome; NO token too), Kalshi
   GET /markets/{ticker}/orderbook (yes and no bids; YES ask = 1 - best NO bid).
5. For each outcome j and direction, walk both books and compute the max hedged size with positive edge after fees
   (Poly 0.05p(1-p), Kalshi taker 0.07p(1-p)), plus the best-level edge.
Run repeatedly (e.g. every hour) to build a forward record: appends to out/live_snapshots.csv.
"""
import json, time, datetime as dt
import numpy as np, pandas as pd, requests
import kalshi_api as k
from s02_match_lib import sim

fP = lambda p: 0.05 * p * (1 - p)
fK = lambda p: 0.07 * p * (1 - p) + 0.0001
S = requests.Session()
_last = [0.0]


def pget(url, params=None):
    for i in range(5):
        g = time.time() - _last[0]
        if g < 0.34:
            time.sleep(0.34 - g)
        _last[0] = time.time()
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return None


now = pd.Timestamp.now("UTC").tz_localize(None)
# --- Polymarket upcoming moneyline events
pe = []
off = 0
emin, emax = now.strftime("%Y-%m-%dT%H:%M:%SZ"), (now + pd.Timedelta("80h")).strftime("%Y-%m-%dT%H:%M:%SZ")
while off < 5000:      # gamma caps page size; page until empty. endDate window keeps it small.
    d = pget("https://gamma-api.polymarket.com/events", {"closed": "false", "tag_slug": "soccer", "limit": 100,
                                                         "offset": off, "end_date_min": emin, "end_date_max": emax})
    if not d:
        break
    pe += d
    off += len(d)
print("gamma events in window", len(pe), flush=True)
poly = []
for e in pe:
    ms = [m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"]
    if len(ms) != 3:
        continue
    gst = pd.to_datetime(ms[0].get("gameStartTime"), utc=True, errors="coerce")
    if pd.isna(gst):
        continue
    gst = gst.tz_localize(None)
    if not (now < gst < now + pd.Timedelta("72h")):
        continue
    draw = [m for m in ms if "draw" in (m.get("groupItemTitle") or "").lower()]
    teams = [m for m in ms if m not in draw]
    if len(draw) != 1 or len(teams) != 2:
        continue
    title = e.get("title", "")
    poly.append(dict(slug=e["slug"], kick=gst, t_a=teams[0]["groupItemTitle"], t_b=teams[1]["groupItemTitle"],
                     tok_a=json.loads(teams[0]["clobTokenIds"]), tok_d=json.loads(draw[0]["clobTokenIds"]),
                     tok_b=json.loads(teams[1]["clobTokenIds"]), title=title))
print("poly upcoming moneyline games", len(poly), flush=True)

# --- Kalshi open match events
series = json.load(open("data/soccer_series.json"))
cov = pd.read_csv("out/kalshi_series_coverage.csv")
active = set(cov[cov["last"] >= str(now.date())].series)
kev = []
for s in series:
    if s["ticker"] not in active:
        continue
    for e in k.paged("/events", "events", {"series_ticker": s["ticker"], "status": "open",
                                           "with_nested_markets": "true"}, limit=200):
        ms = e.get("markets", [])
        if len(ms) != 3:
            continue
        kev.append(dict(event_ticker=e["event_ticker"], title=e.get("title"), markets={m["yes_sub_title"]: m["ticker"]
                                                                                       for m in ms},
                        exp=pd.to_datetime(ms[0].get("expected_expiration_time"), utc=True).tz_localize(None)))
print("kalshi open events", len(kev), flush=True)

# --- match: poly team names vs kalshi yes_sub_titles, expiration within 6h after kickoff
pairs = []
for p in poly:
    best = None
    for e in kev:
        if not (pd.Timedelta("-2h") < e["exp"] - p["kick"] < pd.Timedelta("8h")):
            continue
        teams = [t for t in e["markets"] if t.lower() not in ("tie", "draw")]
        if len(teams) != 2:
            continue
        s1 = (sim(p["t_a"], teams[0]) + sim(p["t_b"], teams[1])) / 2
        s2 = (sim(p["t_a"], teams[1]) + sim(p["t_b"], teams[0])) / 2
        sc, order = (s1, teams) if s1 >= s2 else (s2, teams[::-1])
        if sc >= 0.75 and (best is None or sc > best[0]):
            tie = [t for t in e["markets"] if t.lower() in ("tie", "draw")][0]
            best = (sc, e, [e["markets"][order[0]], e["markets"][tie], e["markets"][order[1]]])
    if best:
        pairs.append((p, best[1], best[2], best[0]))
print("matched upcoming", len(pairs), flush=True)


def pbook(tok):
    d = pget("https://clob.polymarket.com/book", {"token_id": tok})
    if not d:
        return [], []
    asks = sorted([(float(a["price"]), float(a["size"])) for a in d.get("asks", [])])
    bids = sorted([(float(a["price"]), float(a["size"])) for a in d.get("bids", [])], reverse=True)
    return asks, bids


def kbook(tkr):
    d = k.get(f"/markets/{tkr}/orderbook").get("orderbook_fp", {})
    yb = sorted([(float(p), float(q)) for p, q in d.get("yes_dollars") or []], reverse=True)
    nb = sorted([(float(p), float(q)) for p, q in d.get("no_dollars") or []], reverse=True)
    yes_asks = [(round(1 - p, 4), q) for p, q in nb]          # buy YES = sell to NO bidders
    no_asks = [(round(1 - p, 4), q) for p, q in yb]
    return sorted(yes_asks), sorted(no_asks)


def walk(a1, a2, f1, f2):
    """buy along two ask ladders (complementary legs) while marginal cost < 1. returns size, pnl, best edge"""
    a1, a2 = [list(x) for x in a1], [list(x) for x in a2]
    i = j = 0; size = pnl = 0.0
    best = (1 - (a1[0][0] + f1(a1[0][0]) + a2[0][0] + f2(a2[0][0]))) if a1 and a2 else np.nan
    while i < len(a1) and j < len(a2):
        c = a1[i][0] + f1(a1[i][0]) + a2[j][0] + f2(a2[j][0])
        if c >= 1:
            break
        q = min(a1[i][1], a2[j][1]); size += q; pnl += q * (1 - c)
        a1[i][1] -= q; a2[j][1] -= q
        if a1[i][1] <= 1e-9: i += 1
        if a2[j][1] <= 1e-9: j += 1
    return size, pnl, best


rows = []
ts = now.isoformat()
for p, e, ktk, sc in pairs:
    toks = [p["tok_a"], p["tok_d"], p["tok_b"]]
    for jj in range(3):
        pya, _ = pbook(toks[jj][0]); pna, _ = pbook(toks[jj][1])
        kya, kna = kbook(ktk[jj])
        s1, p1, b1 = walk(pya, kna, fP, fK)      # Poly YES + Kalshi NO
        s2, p2, b2 = walk(pna, kya, fP, fK)      # Poly NO + Kalshi YES
        rows.append(dict(snap_utc=ts, slug=p["slug"], event_ticker=e["event_ticker"], j=jj, match_score=sc,
                         mins_to_kick=(p["kick"] - now).total_seconds() / 60,
                         poly_yes_ask=pya[0][0] if pya else np.nan, poly_no_ask=pna[0][0] if pna else np.nan,
                         k_yes_ask=kya[0][0] if kya else np.nan, k_no_ask=kna[0][0] if kna else np.nan,
                         best_edge_A=b1, size_A=s1, pnl_A=p1, best_edge_B=b2, size_B=s2, pnl_B=p2))
df = pd.DataFrame(rows)
import os
hdr = not os.path.exists("out/live_snapshots.csv")
df.to_csv("out/live_snapshots.csv", mode="a", header=hdr, index=False)
if len(df):
    df["best"] = df[["best_edge_A", "best_edge_B"]].max(axis=1)
    print(df.best.describe())
    print("outcome-rows with positive executable edge:", int(((df.size_A > 0) | (df.size_B > 0)).sum()), "of", len(df),
          "| total $ pnl available:", round(df.pnl_A.sum() + df.pnl_B.sum(), 2))
    print(df.sort_values("best", ascending=False).head(10)[["slug", "j", "mins_to_kick", "poly_yes_ask", "poly_no_ask",
                                                            "k_yes_ask", "k_no_ask", "best", "size_A", "size_B",
                                                            "pnl_A", "pnl_B"]].to_string())
