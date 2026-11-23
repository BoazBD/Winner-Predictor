"""s02 - match Kalshi soccer match events to Polymarket games (games_meta, ok==True).

Kalshi side: home/away parsed from rules text "... the <HOME> vs <AWAY> professional ... scheduled for <Mon D, YYYY>";
outcome markets mapped by yes_sub_title (team name / 'Tie').  Polymarket side: poly_home / poly_away / kick_utc.
Candidates: same calendar date +-1 day.  Score = mean fuzzy similarity of (home,home),(away,away) using normalised
names (accents, FC/CF/SC... stripped, a small alias table, token containment, difflib ratio).  Greedy 1:1 assignment
by best score; accept >= 0.72 (inspected by hand, see out/match_borderline.csv).
Output: data/matches.parquet  (event_ticker, unique_id, score, kalshi tickers for poly k=0,1,2 in POLY orientation)
Polymarket k: grid/master 'k' is in Winner orientation; games_meta.swapped tells whether Winner home == poly away.
We store tickers per Polymarket token index (tok0=home win, tok1=draw, tok2=away win in POLY orientation) and
s03 converts to master's k via `swapped`.
"""
import re, unicodedata, difflib
import numpy as np, pandas as pd

ALIAS = {"man utd": "manchester united", "man united": "manchester united", "man city": "manchester city",
         "spurs": "tottenham", "wolves": "wolverhampton", "psg": "paris saint germain", "inter": "internazionale",
         "inter milan": "internazionale", "atletico": "atletico madrid", "atl madrid": "atletico madrid",
         "bayern": "bayern munich", "bayern munchen": "bayern munich", "gladbach": "monchengladbach",
         "nottm forest": "nottingham forest", "sheff utd": "sheffield united", "sheff wed": "sheffield wednesday",
         "usa": "united states", "lafc": "los angeles fc", "la galaxy": "los angeles galaxy",
         "ny red bulls": "new york red bulls", "nycfc": "new york city", "west brom": "west bromwich",
         "qpr": "queens park rangers", "leverkusen": "bayer leverkusen", "dortmund": "borussia dortmund"}
STOP = {"fc", "cf", "sc", "afc", "ac", "cd", "ca", "club", "de", "fk", "sk", "if", "bk", "the", "and", "sv", "us",
        "ud", "rc", "rcd", "ssc", "as", "calcio", "football", "futbol", "clube", "deportivo", "sporting"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("&", " and "))
    s = re.sub(r"\s+", " ", s).strip()
    s = ALIAS.get(s, s)
    return " ".join(w for w in s.split() if w not in STOP) or s


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    cont = len(ta & tb) / min(len(ta), len(tb))
    r = difflib.SequenceMatcher(None, a, b).ratio()
    pre = 1.0 if (a.startswith(b) or b.startswith(a)) and min(len(a), len(b)) >= 4 else 0.0
    return max(r, 0.9 * cont, 0.9 * pre)


km = pd.read_parquet("data/kalshi_markets.parquet")
rx = re.compile(r"the (.+?) vs\.? (.+?) (?:professional|soccer|women|men|game|match|\w+ soccer).*?scheduled for ([A-Z][a-z]{2}) (\d{1,2}), (\d{4})")
ev = []
for et, g in km.groupby("event_ticker"):
    if len(g) != 3:
        continue
    m = rx.search(g.rules.iloc[0] or "")
    if not m:
        continue
    home, away = m.group(1).strip(), m.group(2).strip()
    date = pd.Timestamp(f"{m.group(3)} {m.group(4)} {m.group(5)}")
    subs = dict(zip(g.yes_sub, g.ticker))
    tie = [t for s, t in subs.items() if str(s).lower() in ("tie", "draw")]
    teams = [(s, t) for s, t in subs.items() if str(s).lower() not in ("tie", "draw")]
    if len(tie) != 1 or len(teams) != 2:
        continue
    # map team markets to home/away by similarity
    (s1, t1), (s2, t2) = teams
    if sim(s1, home) + sim(s2, away) >= sim(s1, away) + sim(s2, home):
        th, ta = t1, t2
    else:
        th, ta = t2, t1
    ev.append(dict(event_ticker=et, series=g.series.iloc[0], fee_type=g.fee_type.iloc[0], k_home=home, k_away=away,
                   k_date=date, t_home=th, t_tie=tie[0], t_away=ta, k_volume=g.volume.sum(),
                   k_close=g.close_time.max(), k_open=g.open_time.min(), result_home=g.set_index("ticker").result[th],
                   result_tie=g.set_index("ticker").result[tie[0]], result_away=g.set_index("ticker").result[ta],
                   rules=g.rules.iloc[0], rules2=g.rules2.iloc[0]))
ev = pd.DataFrame(ev)
print("kalshi 3-way events parsed", len(ev), "of", km.event_ticker.nunique())

gm = pd.read_parquet("../../lead_lag/data/games_meta.parquet")
gm = gm[gm.ok].copy()
gm["kd"] = pd.to_datetime(gm.kick_utc).dt.normalize()
gm = gm[gm.kd >= ev.k_date.min() - pd.Timedelta("2D")]
by_day = {d: x for d, x in gm.groupby("kd")}
cand = []
for r in ev.itertuples():
    for dd in (-1, 0, 1):
        x = by_day.get(r.k_date + pd.Timedelta(days=dd))
        if x is None:
            continue
        for p in x.itertuples():
            sh, sa = sim(r.k_home, p.poly_home), sim(r.k_away, p.poly_away)
            sx = (sim(r.k_home, p.poly_away) + sim(r.k_away, p.poly_home)) / 2
            sc = (sh + sa) / 2
            if max(sc, sx) >= 0.6:
                cand.append(dict(event_ticker=r.event_ticker, unique_id=p.unique_id, score=sc, score_x=sx,
                                 sh=sh, sa=sa, dd=dd, k_home=r.k_home, k_away=r.k_away, poly_home=p.poly_home,
                                 poly_away=p.poly_away, kick_utc=p.kick_utc, league=p.series))
cand = pd.DataFrame(cand)
cand["best"] = cand[["score", "score_x"]].max(1) - 0.02 * cand.dd.abs()
cand = cand.sort_values("best", ascending=False)
used_e, used_p, keep = set(), set(), []
for r in cand.itertuples():
    if r.event_ticker in used_e or r.unique_id in used_p:
        continue
    used_e.add(r.event_ticker); used_p.add(r.unique_id); keep.append(r.Index)
mt = cand.loc[keep]
# require each side to be a plausible match, not just the average
ok = (mt.best >= 0.72) & (np.minimum(mt.sh, mt.sa).where(mt.score >= mt.score_x,
                                                          np.minimum(mt.sh, mt.sa) * 0 + mt.score_x) >= 0.55)
mt[(mt.best >= 0.6) & (mt.best < 0.85)].to_csv("out/match_borderline.csv", index=False)
mt = mt[ok].merge(ev, on=["event_ticker", "k_home", "k_away"])
mt["flipped"] = mt.score_x > mt.score          # kalshi home == poly away
# Kalshi ticker for Polymarket tok0 (poly home win), tok1 (draw), tok2 (poly away win)
mt["kt0"] = np.where(mt.flipped, mt.t_away, mt.t_home)
mt["kt1"] = mt.t_tie
mt["kt2"] = np.where(mt.flipped, mt.t_home, mt.t_away)
mt.to_parquet("data/matches.parquet")
print("matched", len(mt), "flipped", int(mt.flipped.sum()))
print(mt.league.value_counts().head(30).to_string())
mt.groupby(mt.kick_utc.dt.to_period("M")).size().to_csv("out/matches_by_month.csv")
print(mt.groupby(mt.kick_utc.dt.to_period("M")).size().to_string())
