"""Polymarket soccer 1X2 prices + matching to Winner games (paper trading only; no orders are ever placed).

Polymarket lists a soccer match as an event with 3 binary "moneyline" markets (home / draw / away). Buying 1 share
of "Yes" at ask price p pays 1 USDC if it wins. Taker fee (sports): fee = rate * p * (1 - p) per share (rate 0.05),
so the effective decimal odds are 1 / (p + rate * p * (1 - p)).

Matching: a Winner game is matched to the Polymarket event with the same kickoff (Winner time is Israel local,
1 minute early) whose team names agree. Winner names are Hebrew, so each name is compared two ways and the better
counts: by sound (consonant skeletons, דורדרכט -> drdrkt, Dordrecht -> drdrkt) and by its English translation,
which an LLM on AWS Bedrock produces once per team name (stored in Firestore meta/team_names).
A wrong match shows up as a fake "edge", so the thresholds favour precision over recall.
"""
import re, json, unicodedata, datetime as dt
from difflib import SequenceMatcher
import requests
import numpy as np, pandas as pd

GAMMA = "https://gamma-api.polymarket.com/events"
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
TRADES = "https://data-api.polymarket.com/v2/trades"
SOCCER_TAG = 100350
DEFAULT_FEE_RATE = 0.05

# ---------------------------------------------------------------- fetching
def fetch_games(days=8):
    """Upcoming Polymarket soccer games with a 3-way moneyline. One row per game with best bid/ask per outcome."""
    s = requests.Session()
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for d in range(days):                       # day windows: the API caps offset at ~2000 per query
        a, b = now + dt.timedelta(days=d), now + dt.timedelta(days=d + 1)
        off = 0
        while off <= 2000:
            r = s.get(GAMMA, params=dict(tag_id=SOCCER_TAG, closed="false", limit=100, offset=off, order="startTime",
                                         ascending="true", start_time_min=f"{a:%Y-%m-%dT%H:%M:%SZ}",
                                         start_time_max=f"{b:%Y-%m-%dT%H:%M:%SZ}"), timeout=30)
            r.raise_for_status()
            page = r.json()
            for e in page:
                g = _parse(e)
                if g: rows.append(g)
            if len(page) < 100:
                break
            off += 100
    return pd.DataFrame(rows).drop_duplicates("event_id") if rows else pd.DataFrame()


def _parse(e):
    ms = e.get("markets") or []
    if len(ms) != 3 or not all(m.get("sportsMarketType") == "moneyline" for m in ms) or not e.get("startTime"):
        return None
    if " vs. " not in e.get("title", ""):
        return None
    home, away = e["title"].split(" vs. ", 1)
    by = {}
    for m in ms:
        t = (m.get("groupItemTitle") or "").strip()
        k = 1 if t.lower().startswith("draw") else 0 if t == home else 2 if t == away else None
        if k is None or k in by or not m.get("acceptingOrders", True) or m.get("closed"):
            return None
        by[k] = m
    if len(by) != 3:
        return None
    g = dict(event_id=str(e["id"]), slug=e.get("slug"), title=e["title"], series=e.get("seriesSlug"),
             series_title=((e.get("series") or [{}])[0] or {}).get("title"),
             start=pd.Timestamp(e["startTime"]).tz_convert("UTC").tz_localize(None), poly_home=home, poly_away=away)
    for k, m in by.items():
        fs = m.get("feeSchedule") or {}
        g[f"bid{k}"] = float(m.get("bestBid") or np.nan)
        g[f"ask{k}"] = float(m.get("bestAsk") or np.nan)
        g[f"liq{k}"] = float(m.get("liquidityNum") or 0)
        g[f"fee{k}"] = float(fs.get("rate", DEFAULT_FEE_RATE)) if m.get("feesEnabled", True) else 0.0
        g[f"mkt{k}"] = m.get("slug")
        g[f"cond{k}"] = m.get("conditionId")
        g[f"tok{k}"] = (json.loads(m.get("clobTokenIds") or "[]") or [None])[0]   # token of the "Yes" share
    return g


def price_at(token, ts, lookback_h=8):
    """Polymarket price of a "Yes" token at UTC time `ts` (last point at or before it), or None if it did not trade
    in the `lookback_h` hours before."""
    end = int(pd.Timestamp(ts).timestamp())
    r = requests.get(CLOB_HISTORY, params=dict(market=token, startTs=end - lookback_h * 3600, endTs=end, fidelity=10),
                     timeout=30)
    r.raise_for_status()
    h = [x for x in r.json().get("history") or [] if x["t"] <= end]   # the API can return points after endTs
    return float(h[-1]["p"]) if h else None


def max_yes_buy(cond, since, until, max_pages=15):
    """Highest price at which a taker ended up long a market's "Yes" share between UTC `since` and `until`
    (bought Yes at p, or sold No at 1-p): the test for whether a resting sell order on Yes would have been filled.
    The API returns a market's trades newest first and ignores time filters, so pages are read back to `since`.
    Returns (price or None, complete): complete=False when `max_pages` ran out before reaching `since`."""
    lo, hi = pd.Timestamp(since).timestamp(), pd.Timestamp(until).timestamp()
    best, cur = None, None
    for _ in range(max_pages):
        r = requests.get(TRADES, params=dict(condition_id=cond, limit=1000, **({"cursor": cur} if cur else {})), timeout=30)
        r.raise_for_status()
        j = r.json(); data = j.get("data") or []
        for x in data:
            yes = int(x.get("outcome_index", -1)) == 0
            if lo < int(x["timestamp"]) < hi and yes == (x["side"] == "BUY"):
                px = float(x["price"]) if yes else 1 - float(x["price"])
                best = px if best is None else max(best, px)
        cur = (j.get("pagination") or {}).get("next_cursor")
        if not data or not cur or min(int(x["timestamp"]) for x in data) <= lo:
            return best, True
    return best, False


def token_for_market(slug):
    """ "Yes" token of a market by slug (finished markets are only returned with closed=true)."""
    for closed in ("true", "false"):
        r = requests.get(GAMMA_MARKETS, params=dict(slug=slug, closed=closed), timeout=30)
        r.raise_for_status()
        if r.json():
            return json.loads(r.json()[0]["clobTokenIds"])[0]
    return None


def effective_odds(ask, fee_rate):
    """Decimal odds after the taker fee for buying at `ask`."""
    return 1.0 / (ask + fee_rate * ask * (1 - ask))

# ---------------------------------------------------------------- name skeletons
_HEB = {"ב": "b", "ג": "g", "ד": "d", "ז": "s", "ח": "k", "ט": "t", "כ": "k", "ך": "k", "ל": "l", "מ": "m", "ם": "m",
        "נ": "n", "ן": "n", "ס": "s", "פ": "p", "ף": "p", "צ": "s", "ץ": "s", "ק": "k", "ר": "r", "ש": "s", "ת": "t"}
_HEB_GERESH = {"ג": "g", "צ": "k", "ז": "s"}          # ג'=j  צ'=ch  ז'=zh
_STOP = {"fc", "sc", "afc", "cf", "cd", "fk", "sk", "ac", "ss", "sv", "as", "us", "club", "de", "del", "la", "le",
         "the", "of", "cp", "sd", "ud", "if", "bk", "ik", "nk", "hnk", "gnk", "rc", "ca", "ec", "se", "ks", "kv"}
_HEB_STOP = {"פ.צ", "אפ.סי", "פ.ק", "ס.ק", "מועדון"}


def skel_he(name):
    s = re.sub(r"[‎‏‪-‮]", "", str(name))
    s = " ".join(w for w in re.split(r"[\s\-]+", s) if w not in _HEB_STOP)
    out = []
    for i, c in enumerate(s):
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if nxt in "'׳" and c in _HEB_GERESH:
            out.append(_HEB_GERESH[c]); continue
        out.append(_HEB.get(c, " " if c.isspace() else ""))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def skel_en(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    s = " ".join(w for w in s.split() if w not in _STOP)
    for a, b in (("ch", "k"), ("ck", "k"), ("ph", "p"), ("sch", "s"), ("sh", "s"), ("th", "t"), ("ts", "s"),
                 ("tz", "s"), ("x", "ks"), ("q", "k"), ("w", "b"), ("v", "b"), ("f", "p"), ("j", "g"), ("c", "k"),
                 ("z", "s")):
        s = s.replace(a, b)
    s = re.sub(r"[aeiouyh]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def tags_he(name):
    """Age / women / reserve-team tags, which must agree between the two sites."""
    s = str(name)
    t = set(re.findall(r"עד (\d{2})", s))
    if "נשים" in s: t.add("w")
    if re.search(r"\sב'$|\sב׳$|\sII$", s): t.add("b")
    return t


def tags_en(name):
    s = str(name)
    t = set(re.findall(r"\bU-?(\d{2})\b", s, re.I))
    if re.search(r"\b(women|wfc|w)\b", s, re.I): t.add("w")
    if re.search(r"\s(B|II)$", s): t.add("b")
    return t


def _strip_he(name):
    return re.sub(r"עד \d{2}|נשים|\sב['׳]$|\sII$", " ", str(name))


def _strip_en(name):
    return re.sub(r"\bU-?\d{2}\b|\b(women|wfc)\b|\s(B|II|W)$", " ", str(name), flags=re.I)


def _ratio(a, b):
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not a or not b:
        return 0.0
    r = SequenceMatcher(None, a, b).ratio()
    # one name is often a shortened form of the other ("Hartberg" vs "TSV Hartberg")
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        r = max(r, 0.85)
    return r


def sim(he, en, alias=None):
    """Similarity of a Hebrew Winner name and an English Polymarket name. `alias` = the English name the LLM
    translator gave for `he` (team_names); the higher of the sound-alike and the translated comparison counts."""
    if tags_he(he) != tags_en(en):
        return 0.0
    b = skel_en(_strip_en(en))
    r = _ratio(re.sub(r"(.)\1+", r"\1", skel_he(_strip_he(he))), b)
    if alias:
        r = max(r, _ratio(skel_en(_strip_en(alias)), b))
    return r

# ---------------------------------------------------------------- LLM translation of team names (AWS Bedrock)
_TRANSLATE_PROMPT = (
    "You get soccer team names in Hebrew as listed by the Israeli bookmaker Winner, one per line as "
    "'<team> | <competition>'. Give each team's standard English name as international betting and sports sites "
    "list it (e.g. 'Bayern Munich', 'Paris Saint-Germain', 'Shrewsbury Town'); national teams by country name. "
    "Keep age, women and reserve markers as 'U21', 'Women', 'B'. Copy each Hebrew name exactly into `hebrew`. "
    "If you do not know the club, transliterate the name.")
_TRANSLATE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["teams"], "properties": {
    "teams": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                         "required": ["hebrew", "english"],
                                         "properties": {"hebrew": {"type": "string"}, "english": {"type": "string"}}}}}}


def translate(items, region, model, chunk=100):
    """items: [(hebrew team name, hebrew competition)]. Returns {hebrew name: English name} for the names the model
    answered; unanswered names are simply missing (asked again next run)."""
    import json
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region=region, timeout=60, max_retries=1)
    out = {}
    cfg = {"format": {"type": "json_schema", "schema": _TRANSLATE_SCHEMA}}
    if "haiku-4-5" not in model:   # Haiku 4.5 rejects the effort parameter (400)
        cfg["effort"] = "low"
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        r = client.messages.create(
            model=model, max_tokens=16000, system=_TRANSLATE_PROMPT, output_config=cfg,
            messages=[{"role": "user", "content": "\n".join(f"{n} | {lg}" for n, lg in part)}])
        if r.stop_reason != "end_turn":
            continue
        asked = {n for n, _ in part}
        for t in json.loads(next(b.text for b in r.content if b.type == "text"))["teams"]:
            if t["hebrew"] in asked and t["english"].strip():
                out[t["hebrew"]] = t["english"].strip()
    return out

# ---------------------------------------------------------------- matching
def match(games, poly, names=None, tol_min=12, min_score=0.65, min_margin=0.12, min_team=0.4, min_swapped=0.8):
    """games: DataFrame(unique_id, home, away, kick_utc). poly: fetch_games() output. names: {hebrew: English}.
    Returns DataFrame(unique_id, event_id, swapped, score) of confident matches. Each team must match on its own
    (min_team); a home/away-swapped match (neutral venue / different listing) needs a much higher score.
    tol_min stays below 14 so that a game in the neighbouring quarter-hour slot is never a candidate (the two sites
    list the same game exactly 1 minute apart), and a Polymarket event is given to one Winner game only."""
    if games.empty or poly.empty:
        return pd.DataFrame(columns=["unique_id", "event_id", "swapped", "score"])
    names = names or {}
    out = []
    pst = poly.start.values.astype("datetime64[m]")
    for g in games.itertuples():
        k = np.datetime64(pd.Timestamp(g.kick_utc), "m")
        cand = poly[np.abs((pst - k).astype(int)) <= tol_min]
        sc = []
        for p in cand.itertuples():
            h, w = names.get(g.home), names.get(g.away)
            a, b = sim(g.home, p.poly_home, h), sim(g.away, p.poly_away, w)
            c, d = sim(g.home, p.poly_away, h), sim(g.away, p.poly_home, w)
            straight = (a + b) / 2 if min(a, b) >= min_team else 0.0
            swapped = (c + d) / 2 if min(c, d) >= min_team and (c + d) / 2 >= min_swapped else 0.0
            sc.append((max(straight, swapped), swapped > straight, p.event_id))
        if not sc:
            continue
        sc.sort(reverse=True)
        best, second = sc[0][0], (sc[1][0] if len(sc) > 1 else 0.0)
        if best >= min_score and best - second >= min_margin:
            out.append(dict(unique_id=g.unique_id, event_id=sc[0][2], swapped=bool(sc[0][1]), score=round(best, 3)))
    m = pd.DataFrame(out, columns=["unique_id", "event_id", "swapped", "score"])
    return m.sort_values("score", ascending=False).drop_duplicates("event_id").reset_index(drop=True)
