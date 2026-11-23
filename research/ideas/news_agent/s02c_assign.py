"""s02c - Assign GKG headlines to sample games.

A headline belongs to a game if its title mentions the home or away team (name or alias) and GDELT saw it
in [kick-48h, kick-5min). A broad team-news keyword filter keeps LLM cost down. Titles are HTML-unescaped
and de-duplicated per game (earliest sighting kept).
t_seen = GDELT 15-min slot (the time a GDELT-based agent could have the item: conservative).
t_pub  = publisher timestamp from <PAGE_PRECISEPUBTIMESTAMP> if it lies within 12h before t_seen (else NaN).
Output: data/headlines.parquet, out/source_stats.json
"""
import os, re, glob, html, json
import pandas as pd
from s02b_gkg_fetch import ALIAS

HERE = os.path.dirname(os.path.abspath(__file__))
KW = re.compile(r"injur|line-?up|starting (xi|eleven)|\bxi\b|team news|suspen|\bban\b|banned|ruled out|doubt|fitness|"
                r"\bfit\b|rotat|rested|\brest\b|return|absent|absence|sidelined|\bmiss|blow|boost|setback|knock|hamstring|"
                r"illness|\bill\b|unavailable|without|available|squad|bench|dropped|\bdrop|press conference|"
                r"confirmed|update|out for|out of|red card|sacked|interim|new manager|head coach", re.I)
s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet"))
fs = sorted(glob.glob(os.path.join(HERE, "data/gkg/*.parquet")))
d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
d["title"] = d.title.map(html.unescape).str.replace(r"\s+", " ", regex=True).str.strip()
d["t_seen"] = pd.to_datetime(d.slot, format="%Y%m%d%H%M%S")
d["t_pub"] = pd.to_datetime(d.pub, format="%Y%m%d%H%M%S", errors="coerce")
lag = (d.t_seen - d.t_pub).dt.total_seconds() / 60
stats = dict(files=len(fs), rows_team_title=len(d), share_with_pub=float(d.t_pub.notna().mean()),
             crawl_lag_min_quantiles={str(q): float(lag[(lag >= 0)].quantile(q)) for q in [.1, .25, .5, .75, .9]},
             share_pub_after_seen=float((lag < 0).mean()))
d.loc[(lag < 0) | (lag > 720), "t_pub"] = pd.NaT
d = d[d.title.str.contains(KW)]
out = []
for _, g in s.iterrows():
    k = g.kick_utc
    w = d[(d.t_seen >= k - pd.Timedelta(hours=48)) & (d.t_seen < k - pd.Timedelta(minutes=5))]
    if w.empty:
        continue
    nm = []
    for side, x in (("home", g.home_q), ("away", g.away_q)):
        x = x.replace("MH ", "")
        al = [x] + ALIAS.get(x, [])
        rx = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in al) + r")\b")
        nm.append(w.title.str.contains(rx))
    sel = w[nm[0] | nm[1]].copy()
    if sel.empty:
        continue
    sel["mentions"] = (nm[0][sel.index].map({True: "H", False: ""}) + nm[1][sel.index].map({True: "A", False: ""}))
    sel["key"] = sel.title.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str[:80]
    sel = sel.sort_values("t_seen").drop_duplicates("key")
    sel["unique_id"] = g.unique_id
    sel["mins_before"] = ((k - sel.t_seen).dt.total_seconds() / 60).round().astype(int)
    out.append(sel[["unique_id", "t_seen", "t_pub", "mins_before", "src", "url", "title", "mentions"]])
h = pd.concat(out, ignore_index=True)
h.to_parquet(os.path.join(HERE, "data/headlines.parquet"))
stats.update(games_with_headlines=int(h.unique_id.nunique()), headlines=len(h),
             per_game_median=float(h.groupby("unique_id").size().median()),
             headlines_last_90min=int((h.mins_before <= 90).sum()))
json.dump(stats, open(os.path.join(HERE, "out/source_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
