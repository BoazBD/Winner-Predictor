"""Step 1: inventory of related (side) markets for every ok Polymarket soccer 1X2 game.

For each game slug S in research/lead_lag/data/games_meta.parquet (ok==True) we query Gamma for the
sibling events  S-more-markets (totals / BTTS / spreads),  S-exact-score,  S-halftime-result.
Gamma accepts repeated ?slug= params, so we batch 25 games (75 slugs) per request (<=3 req/s).
Raw JSON pages are cached gzipped in data/gamma_cache/, parsed table -> data/side_markets.parquet,
summary -> out/inventory_*.csv.
Usage: python s01_inventory.py
"""
import os, json, time, hashlib, gzip
import requests, pandas as pd, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "data", "gamma_cache"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
S = requests.Session()
SUFFIX = ["-more-markets", "-exact-score", "-halftime-result"]


def get(slugs, tries=6):
    key = hashlib.md5("|".join(slugs).encode()).hexdigest()
    f = os.path.join(CACHE, key + ".json.gz")
    if os.path.exists(f):
        return json.load(gzip.open(f, "rt"))
    for i in range(tries):
        try:
            r = S.get("https://gamma-api.polymarket.com/events", params=[("slug", s) for s in slugs] + [("limit", 500)], timeout=40)
            if r.status_code == 200:
                d = r.json(); json.dump(d, gzip.open(f, "wt")); time.sleep(0.35); return d
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError("gamma failed")


def jl(x):
    try:
        return json.loads(x) if isinstance(x, str) else (x or [])
    except Exception:
        return []


def main():
    g = pd.read_parquet(os.path.join(ROOT, "lead_lag", "data", "games_meta.parquet"))
    g = g[g.ok].copy()
    slugmap = {}
    for r in g.itertuples():
        for s in SUFFIX:
            slugmap[r.slug + s] = r.unique_id
    slugs = list(slugmap)
    rows = []
    B = 75
    for i in range(0, len(slugs), B):
        evs = get(slugs[i:i + B])
        for e in evs:
            uid = slugmap.get(e["slug"])
            if uid is None:
                continue
            for m in e.get("markets") or []:
                toks = jl(m.get("clobTokenIds")); outs = jl(m.get("outcomes")); px = jl(m.get("outcomePrices"))
                if len(toks) != 2:
                    continue
                rows.append(dict(unique_id=uid, side_event_id=e["id"], side_slug=e["slug"], kind=next(x for x in SUFFIX if e["slug"].endswith(x)),
                                 mslug=m.get("slug"), question=m.get("question"), smt=m.get("sportsMarketType"), line=m.get("line"),
                                 cond=m.get("conditionId"), tokA=toks[0], tokB=toks[1], outA=outs[0] if outs else None, outB=outs[1] if len(outs) > 1 else None,
                                 resA=float(px[0]) if px else np.nan, vol=float(m.get("volumeNum") or 0), created=m.get("createdAt"),
                                 fees=bool(m.get("feesEnabled")), neg=bool(m.get("negRisk")), closed=bool(m.get("closed"))))
        if (i // B) % 40 == 0:
            print(f"{i}/{len(slugs)} rows={len(rows)}", flush=True)
    d = pd.DataFrame(rows)
    d["created"] = pd.to_datetime(d.created, utc=True, format="mixed").dt.tz_localize(None)
    d = d.merge(g[["unique_id", "kick_utc", "slug", "tier", "series", "poly_volume", "event_id"]], on="unique_id")
    d.to_parquet(os.path.join(HERE, "data", "side_markets.parquet"))
    d["month"] = pd.to_datetime(d.kick_utc).dt.to_period("M").astype(str)
    g["month"] = pd.to_datetime(g.kick_utc).dt.to_period("M").astype(str)
    cov = g.groupby("month").size().rename("games").to_frame()
    for k in ["totals", "both_teams_to_score", "spreads", "soccer_exact_score", "soccer_halftime_result"]:
        cov[k] = d[d.smt == k].groupby("month").unique_id.nunique()
    cov.fillna(0).astype(int).to_csv(os.path.join(OUT, "inventory_coverage_by_month.csv"))
    print(cov.fillna(0).astype(int).to_string())
    lv = d.groupby(["smt", "line"], dropna=False).agg(n=("cond", "size"), games=("unique_id", "nunique"), vol_med=("vol", "median"),
                                                       vol_mean=("vol", "mean"), vol_p90=("vol", lambda x: x.quantile(.9))).reset_index()
    lv.to_csv(os.path.join(OUT, "inventory_by_type_line.csv"), index=False)
    print(lv.to_string())
    gv = g[["unique_id", "poly_volume"]].merge(d.groupby("unique_id").vol.sum().rename("side_vol"), on="unique_id", how="left").fillna(0)
    print("1X2 vol median", gv.poly_volume.median(), "side vol median (games with side)", gv[gv.side_vol > 0].side_vol.median(),
          "ratio of sums", gv.side_vol.sum() / gv.poly_volume.sum())


if __name__ == "__main__":
    main()
