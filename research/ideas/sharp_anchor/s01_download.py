"""s01 - download football-data.co.uk CSVs (main leagues 2024/25-2026/27 and the 'extra' leagues files).

Plain HTTPS GETs, no identifying headers, 0.3 s between requests. Files land in raw/<season>/<div>.csv and raw/new/<CC>.csv.
Key facts found (see logs/s01.txt):
  * Pinnacle early (PSH/PSD/PSA) and closing (PSCH..) columns stop between Oct-2025 and Jan-2026 in every league and are
    absent from all 2026/27 files (Pinnacle shut its public odds API in 2025). Extra leagues only ever had closing odds.
  * Betfair Exchange early (BFEH..) and closing (BFECH..) are present throughout -> used as the sharp anchor.
  * notes.txt: "Betting odds for weekend games are collected Friday afternoons, and on Tuesday afternoons for midweek games."
"""
import os, time, requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
MAIN = "E0 E1 E2 E3 EC SC0 SC1 SC2 SC3 D1 D2 I1 I2 SP1 SP2 F1 F2 N1 B1 P1 T1 G1".split()
EXTRA = "ARG AUT BRA CHN DNK FIN IRL JPN MEX NOR POL ROU RUS SWE SWZ USA".split()


def get(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=60)
    if r.status_code == 200 and len(r.content) > 100:
        open(path, "wb").write(r.content)
        return True
    return False


if __name__ == "__main__":
    for s in ["2425", "2526", "2627"]:
        for d in MAIN:
            ok = get(f"https://www.football-data.co.uk/mmz4281/{s}/{d}.csv", os.path.join(RAW, s, f"{d}.csv"))
            print(s, d, ok); time.sleep(0.3)
    for c in EXTRA:
        ok = get(f"https://www.football-data.co.uk/new/{c}.csv", os.path.join(RAW, "new", f"{c}.csv"))
        print("new", c, ok); time.sleep(0.3)
    get("https://www.football-data.co.uk/notes.txt", os.path.join(RAW, "notes.txt"))
