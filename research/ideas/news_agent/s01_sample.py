"""s01 - Build the game sample for the news-agent POC.

Top-tier Polymarket 1X2 games (big-5 leagues + UCL/UEL) with kickoff 2026-08-01 .. 2026-11-18,
poly_volume >= $250k, and 15-min grid coverage in the last 48h. Adds an English search name per team.
Output: data/sample.parquet
"""
import os, re, unicodedata
import pandas as pd, duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
OVR = {  # GDELT search names (English press usage)
    "BV Borussia 09 Dortmund": "Dortmund", "FC Bayern München": "Bayern Munich", "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Mönchengladbach": "Gladbach", "1. FC Köln": "Cologne", "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz", "TSG 1899 Hoffenheim": "Hoffenheim", "SV Werder Bremen": "Werder Bremen",
    "Hamburger SV": "Hamburg", "FC Internazionale Milano": "Inter Milan", "SSC Napoli": "Napoli", "SS Lazio": "Lazio",
    "AS Roma": "Roma", "AC Milan": "AC Milan", "Club Atlético de Madrid": "Atletico Madrid", "Real Madrid CF": "Real Madrid",
    "FC Barcelona": "Barcelona", "Real Betis Balompié": "Real Betis", "Real Sociedad de Fútbol": "Real Sociedad",
    "RC Celta de Vigo": "Celta Vigo", "RCD Espanyol de Barcelona": "Espanyol", "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Paris Saint-Germain FC": "PSG", "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
    "Racing Club de Lens": "Lens", "RC Strasbourg Alsace": "Strasbourg", "Stade Rennais FC 1901": "Rennes",
    "Stade Brestois 29": "Brest", "AS Monaco FC": "Monaco", "Lille OSC": "Lille", "Sport Lisboa e Benfica": "Benfica",
    "Brighton & Hove Albion FC": "Brighton", "Tottenham Hotspur FC": "Tottenham", "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United", "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nottingham Forest",
    "Wolverhampton Wanderers FC": "Wolves", "West Ham United FC": "West Ham", "AFC Bournemouth": "Bournemouth",
    "Leeds United FC": "Leeds", "AFC Ajax": "Ajax", "Feyenoord Rotterdam": "Feyenoord", "Galatasaray SK": "Galatasaray",
    "Fenerbahçe SK": "Fenerbahce", "Beşiktaş JK": "Besiktas", "FK Bodø/Glimt": "Bodo Glimt", "FC København": "Copenhagen",
    "Club Brugge KV": "Club Brugge", "PSV": "PSV", "Sporting CP": "Sporting", "FC Porto": "Porto",
    "Olympiakós SFP": "Olympiacos", "Panathinaikós AO": "Panathinaikos", "FK Crvena zvezda": "Red Star Belgrade",
    "GNK Dinamo Zagreb": "Dinamo Zagreb", "SK Slavia Praha": "Slavia Prague", "AC Sparta Praha": "Sparta Prague",
    "FC Red Bull Salzburg": "Salzburg", "Qarabağ Ağdam FK": "Qarabag", "Qairat FK": "Kairat", "FK Shakhtar Donetsk": "Shakhtar",
    "SK Puntigamer Sturm Graz": "Sturm Graz", "Atalanta BC": "Atalanta", "ACF Fiorentina": "Fiorentina",
    "Bologna FC 1909": "Bologna", "Juventus FC": "Juventus", "Torino FC": "Torino", "Genoa CFC": "Genoa",
    "US Sassuolo Calcio": "Sassuolo", "Udinese Calcio": "Udinese", "Cagliari Calcio": "Cagliari", "Parma Calcio 1913": "Parma",
    "Como 1907": "Como", "US Lecce": "Lecce", "Frosinone Calcio": "Frosinone", "AC Monza": "Monza", "Venezia FC": "Venezia",
    "VfB Stuttgart": "Stuttgart", "SC Freiburg": "Freiburg", "FC Augsburg": "Augsburg", "RB Leipzig": "Leipzig",
    "FC Schalke 04": "Schalke", "SC Paderborn 07": "Paderborn", "SV 07 Elversberg": "Elversberg", "Athletic Club": "Athletic Bilbao",
    "CA Osasuna": "Osasuna", "Deportivo Alavés": "Alaves", "Málaga CF": "Malaga", "RC Deportivo A Coruña": "Deportivo La Coruna",
    "Sevilla FC": "Sevilla", "Valencia CF": "Valencia", "Villarreal CF": "Villarreal", "Getafe CF": "Getafe", "Elche CF": "Elche",
    "Levante UD": "Levante", "Angers SCO": "Angers", "AJ Auxerre": "Auxerre", "Le Havre AC": "Le Havre", "FC Lorient": "Lorient",
    "Paris FC": "Paris FC", "Toulouse FC": "Toulouse", "ES Troyes AC": "Troyes", "Le Mans FC": "Le Mans", "SC Braga": "Braga",
    "Celtic FC": "Celtic", "Rangers FC": "Rangers", "RSC Anderlecht": "Anderlecht", "Eintracht Frankfurt": "Eintracht Frankfurt",
    "FC Midtjylland": "Midtjylland", "Ferencvárosi TC": "Ferencvaros", "FC Viktoria Plzeň": "Viktoria Plzen",
    "RU Saint-Gilloise": "Union Saint-Gilloise", "Páfos FC": "Pafos", "KAA Gent": "Gent", "FC Twente": "Twente",
}
BIG = {"פרמייר ליג", "ספרדית ראשונה", "איטלקית ראשונה", "גרמנית ראשונה", "צרפתית ראשונה", "ליגת האלופות", "הליגה האירופית"}


def sname(n):
    if n in OVR:
        return OVR[n]
    s = re.sub(r"\b(FC|AFC|CF|SC|AC|FK|SK|CA|RC|AS|US|SSC|SV|TSG|VfL|VfB|KV|PFK|NK|HNK|BK|IL|Calcio)\b", "", n)
    s = re.sub(r"\b\d{2,4}\b", "", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


m = pd.read_parquet(os.path.join(R, "lead_lag/data/games_meta.parquet"))
m = m[m.ok & (m.tier == "top") & m.league.isin(BIG) & (m.kick_utc >= "2026-08-01") & (m.kick_utc < "2026-11-19")
      & (m.poly_volume >= 250_000)].copy()
cov = duckdb.sql(f"""select unique_id, count(*) n48, count(p0) np from '{R}/lead_lag/data/grid.parquet'
                     where mins_to_kick<=2880 group by 1""").df()
m = m.merge(cov, on="unique_id")
m = m[m.np >= 150]
m["home_q"], m["away_q"] = m.poly_home.map(sname), m.poly_away.map(sname)
m = m.sort_values("kick_utc").reset_index(drop=True)
cols = ["unique_id", "event_id", "kick_utc", "league", "poly_home", "poly_away", "home_q", "away_q", "poly_volume", "swapped",
        "outcome_price0", "outcome_price1", "outcome_price2", "fees0", "np"]
m[cols].to_parquet(os.path.join(HERE, "data/sample.parquet"))
print(len(m), m.kick_utc.min(), m.kick_utc.max())
print(m.league.value_counts())
print(m.groupby(m.kick_utc.dt.month).size())
print(sorted(set(m.home_q) | set(m.away_q)))
