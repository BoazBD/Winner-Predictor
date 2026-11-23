"""s02 - build one football-data table (kick >= 2025-07-01) and match it to Polymarket games (games_meta ok==True).

Matching: candidate Polymarket games whose kickoff is within 3 h of the football-data kickoff (FD 'Time' is UK local
time; converted Europe/London -> UTC; if no time, +-30 h). Score = name similarity of home+away (normalised, accents
stripped, club suffixes dropped, token-containment bonus). Pass 1 accepts confident pairs, learns an FD->Polymarket
team-name map; pass 2 re-scores with the map. One-to-one: each Polymarket game used once.
Orientation: grid k is Winner orientation (0 = Winner home). Polymarket tok0 = poly_home; swapped=True means Winner home
= poly_away. FD home is mapped to grid k via that.
Output data/fd_matched.parquet with FD odds converted to grid orientation (anchor columns *_0,*_1,*_2).
"""
import os, glob, re, unicodedata, difflib, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..")
RAW = os.path.join(HERE, "raw")

ODDS = {  # name -> (H, D, A) column triplets
    "ps": ("PSH", "PSD", "PSA"), "psc": ("PSCH", "PSCD", "PSCA"),
    "bfe": ("BFEH", "BFED", "BFEA"), "bfec": ("BFECH", "BFECD", "BFECA"),
    "avg": ("AvgH", "AvgD", "AvgA"), "avgc": ("AvgCH", "AvgCD", "AvgCA"),
    "b365": ("B365H", "B365D", "B365A"), "b365c": ("B365CH", "B365CD", "B365CA"),
}


def load_fd():
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "25*", "*.csv")) + glob.glob(os.path.join(RAW, "26*", "*.csv"))
                    + glob.glob(os.path.join(RAW, "new", "*.csv"))):
        d = pd.read_csv(f, encoding="latin1", on_bad_lines="skip", low_memory=False).copy()
        d.columns = [c.strip().lstrip("﻿").lstrip("ï»¿") for c in d.columns]
        if "HomeTeam" not in d:
            d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "Res": "FTR"})
        div = d["Div"] if "Div" in d else d["Country"]
        g = pd.DataFrame({"div": div, "file": os.path.relpath(f, RAW), "date": pd.to_datetime(d.Date, dayfirst=True, errors="coerce"),
                          "time": d.get("Time"), "fd_home": d.HomeTeam, "fd_away": d.AwayTeam, "ftr": d.get("FTR")})
        for k, cols in ODDS.items():
            for i, c in enumerate(cols):
                g[f"{k}{'HDA'[i]}"] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        out.append(g)
    fd = pd.concat(out, ignore_index=True)
    fd = fd[(fd.date >= "2025-07-01") & fd.fd_home.notna()].drop_duplicates(["date", "fd_home", "fd_away"])
    t = fd.time.fillna("").astype(str).str.strip()
    has_t = t.str.match(r"^\d{1,2}:\d{2}$")
    loc = pd.to_datetime(fd.date.dt.strftime("%Y-%m-%d") + " " + t.where(has_t, "12:00"), errors="coerce")
    utc = loc.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None)
    fd["kick_fd"] = utc
    fd["has_time"] = has_t
    return fd.reset_index(drop=True)


STOP = {"fc", "cf", "afc", "sc", "ac", "cd", "ca", "club", "de", "fk", "if", "ik", "bk", "sk", "the", "us", "as", "ss", "ssc",
        "rc", "rcd", "ud", "sd", "cp", "fotball", "football", "calcio", "1", "sv", "vfb", "vfl", "tsg", "fsv", "bc", "ec", "se",
        "esporte", "clube", "real", "atletico", "deportivo", "sporting", "city", "united", "town", "and", "hotspur", "albion", "athletic"}
ABBR = {"man": "manchester", "utd": "united", "nott'm": "nottingham", "ath": "athletic", "sp": "sporting", "st": "saint",
        "wolves": "wolverhampton", "spurs": "tottenham", "qpr": "queens park rangers", "mgladbach": "monchengladbach",
        "m'gladbach": "monchengladbach", "psg": "paris saint germain", "sociedad": "sociedad", "inter": "internazionale",
        "ein": "eintracht", "dortmund": "borussia dortmund", "leverkusen": "bayer leverkusen", "betis": "betis",
        "espanol": "espanyol", "vallecano": "rayo vallecano", "celta": "celta vigo", "fortuna": "fortuna", "sheffield": "sheffield"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    toks = []
    for w in s.split():
        toks += ABBR.get(w, w).split()
    return toks


def sim(a, b):
    ta, tb = norm(a), norm(b)
    ca, cb = [t for t in ta if t not in STOP] or ta, [t for t in tb if t not in STOP] or tb
    sa, sb = " ".join(ca), " ".join(cb)
    r = difflib.SequenceMatcher(None, sa, sb).ratio()
    # token containment / prefix bonus
    hit = sum(any(x == y or (len(x) >= 4 and (y.startswith(x) or x.startswith(y))) for y in cb) for x in ca)
    cont = hit / max(1, min(len(ca), len(cb)))
    return max(r, 0.5 * r + 0.5 * cont)


def main():
    fd = load_fd()
    gm = pd.read_parquet(os.path.join(R, "lead_lag", "data", "games_meta.parquet"))
    gm = gm[gm.ok].copy()
    gm["kick"] = gm[["kick_utc", "start"]].min(axis=1)
    gm = gm[gm.kick >= "2025-06-25"].reset_index(drop=True)
    print("FD rows", len(fd), "poly games", len(gm))
    gk = gm.kick.values.astype("datetime64[m]").astype(np.int64)
    fk = fd.kick_fd.values.astype("datetime64[m]").astype(np.int64)
    order = np.argsort(gk); gks = gk[order]
    team_map = {}

    def score_pass(use_map):
        cands = []
        for i, r in fd.iterrows():
            if pd.isna(r.kick_fd):
                continue
            win = 180 if r.has_time else 30 * 60
            lo, hi = np.searchsorted(gks, fk[i] - win), np.searchsorted(gks, fk[i] + win, side="right")
            for j in order[lo:hi]:
                g = gm.iloc[j]
                for flip in (False, True):
                    ph, pa = (g.poly_away, g.poly_home) if flip else (g.poly_home, g.poly_away)
                    sh = 1.0 if use_map and team_map.get(r.fd_home) == ph else sim(r.fd_home, ph)
                    sa = 1.0 if use_map and team_map.get(r.fd_away) == pa else sim(r.fd_away, pa)
                    s = sh + sa - (0.3 if flip else 0) - abs(fk[i] - gk[j]) / 600.0 * (1 if r.has_time else 0.1)
                    if min(sh, sa) >= 0.45 and sh + sa >= 1.25:
                        cands.append((s, i, j, flip, sh, sa))
        cands.sort(reverse=True)
        usedf, usedg, acc = set(), set(), []
        for s, i, j, flip, sh, sa in cands:
            if i in usedf or j in usedg:
                continue
            usedf.add(i); usedg.add(j); acc.append((i, j, flip, s, sh, sa))
        return acc

    acc = score_pass(False)
    # learn team map from confident pass-1 matches (both names >= .75)
    votes = {}
    for i, j, flip, s, sh, sa in acc:
        r, g = fd.iloc[i], gm.iloc[j]
        ph, pa = (g.poly_away, g.poly_home) if flip else (g.poly_home, g.poly_away)
        if sh >= 0.75 and sa >= 0.75 and not flip:
            votes.setdefault(r.fd_home, {}).setdefault(ph, 0); votes[r.fd_home][ph] += 1
            votes.setdefault(r.fd_away, {}).setdefault(pa, 0); votes[r.fd_away][pa] += 1
    team_map.update({k: max(v, key=v.get) for k, v in votes.items() if max(v.values()) >= 2})
    print("pass1", len(acc), "team map size", len(team_map))
    acc = score_pass(True)
    print("pass2", len(acc))
    rows = []
    for i, j, flip, s, sh, sa in acc:
        r, g = fd.iloc[i], gm.iloc[j]
        # FD home == poly_home unless flip. Winner home (k0) == poly_home unless swapped.
        fd_home_is_k0 = (not flip) != bool(g.swapped)
        d = dict(unique_id=g.unique_id, event_id=g.event_id, kick=g.kick, series=g.series, tier=g.tier, fees0=g.fees0,
                 ev_volume=g.ev_volume, title=g.title, fd_div=r["div"], fd_file=r.file, fd_home=r.fd_home, fd_away=r.fd_away,
                 kick_fd=r.kick_fd, has_time=r.has_time, flip=flip, fd_home_is_k0=fd_home_is_k0, score=s, sim_h=sh, sim_a=sa,
                 kick_diff_min=(fk[i] - gk[j]))
        idx = [0, 1, 2] if fd_home_is_k0 else [2, 1, 0]  # FD (H,D,A) position -> grid k
        res = {"H": 0, "D": 1, "A": 2}.get(r.ftr)
        d["result_k"] = np.nan if res is None else idx[res]
        for k in ODDS:
            for pos, L in enumerate("HDA"):
                d[f"{k}_{idx[pos]}"] = r[f"{k}{L}"]
        rows.append(d)
    out = pd.DataFrame(rows)
    # sanity: winner-orientation outcome from Polymarket resolution
    gmi = gm.set_index("unique_id")
    def poly_res(u):
        g = gmi.loc[u]
        op = [g.outcome_price0, g.outcome_price1, g.outcome_price2]
        try:
            v = [float(json.loads(x)[0]) if isinstance(x, str) and x.startswith("[") else float(x) for x in op]
        except Exception:
            return np.nan
        if max(v) < 0.99:
            return np.nan
        k = int(np.argmax(v))
        return (2 - k if k != 1 else 1) if g.swapped else k
    out["poly_res_k"] = [poly_res(u) for u in out.unique_id]
    agree = (out.poly_res_k == out.result_k)[out.poly_res_k.notna() & out.result_k.notna()]
    print("result agreement FD vs Polymarket resolution:", agree.mean(), agree.size)
    out.to_parquet(os.path.join(HERE, "data", "fd_matched.parquet"))
    json.dump(team_map, open(os.path.join(HERE, "data", "team_map.json"), "w"), indent=1, ensure_ascii=False)
    print(out.groupby(out.kick.dt.to_period("M")).size().to_string())
    print(out.fd_div.value_counts().to_string())
    print("FD rows matched share by div:\n", (out.fd_div.value_counts() / fd["div"].value_counts()).dropna().round(2).to_string())
    print(out.sort_values("score").head(25)[["fd_home", "fd_away", "title", "score", "kick_diff_min"]].to_string())


if __name__ == "__main__":
    main()
