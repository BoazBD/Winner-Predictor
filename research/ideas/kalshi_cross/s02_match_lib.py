"""Team-name normalisation + fuzzy similarity shared by s02 and s07."""
import re, unicodedata, difflib

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


