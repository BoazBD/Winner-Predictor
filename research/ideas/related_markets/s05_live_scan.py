"""Step 5: LIVE order-book scan of upcoming soccer games (read-only, no orders) for hedged locks and model deviations.

For every game kicking off within the next H hours (default 48):
  * gamma events (tag 100350) -> main 1X2 event + siblings -more-markets, -exact-score, -halftime-result
  * CLOB POST /books for every token (best ask/bid + size)
  * full-time scoreline payoff matrix (0..10 x 0..10) for every token (1X2 yes/no, totals, BTTS, spreads, team totals,
    exact scores); LP: min sum_j x_j*(ask_j + fee_j)  s.t.  payoff(scoreline) >= 1 for every scoreline, x >= 0.
    Cost < 1 = guaranteed profit (true lock). Also halftime 3-way and exact-score negRisk sums checked separately.
  * DC model from 1X2 mids vs side-market mids (only when both sides of book exist and spread <= 0.05).
Output: out/live_scan_<UTC stamp>.csv (one row per game), out/live_tokens_<stamp>.parquet.
Usage: python s05_live_scan.py [hours]
Fee: taker fee per share 0.05*p*(1-p) on fee-enabled markets.
"""
import os, sys, re, json, time, datetime as dt
import requests, numpy as np, pandas as pd
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Table

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
S = requests.Session()
G = 11
I = np.arange(G); HG = np.repeat(I, G).reshape(G, G); AG = HG.T  # HG[h,a]=h
FEE = lambda p: 0.05 * p * (1 - p)
_last = [0.0]


def get(url, params=None, post=None):
    for i in range(6):
        dtm = time.time() - _last[0]
        if dtm < 0.34:
            time.sleep(0.34 - dtm)
        _last[0] = time.time()
        try:
            r = S.post(url, json=post, timeout=40) if post is not None else S.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(url)


def jl(x):
    return json.loads(x) if isinstance(x, str) else (x or [])


def payoff_A(slug, base, smt, line, home_code, away_code, listed_scores):
    """Payoff matrix (G,G) of token A (first outcome) for a full-time market, or None if not a FT goal market."""
    suf = slug[len(base):]
    if smt == "moneyline":
        if suf == "-draw":
            return (HG == AG).astype(float)
        if suf == "-" + home_code:
            return (HG > AG).astype(float)
        if suf == "-" + away_code:
            return (HG < AG).astype(float)
        return None
    if smt == "totals":
        return (HG + AG > abs(line)).astype(float)
    if smt == "both_teams_to_score":
        return ((HG > 0) & (AG > 0)).astype(float)
    if smt == "spreads":
        L = abs(line)
        return (HG - AG > L).astype(float) if "-spread-home-" in suf else (AG - HG > L).astype(float)
    if smt == "soccer_team_totals":
        L = abs(line)
        return (HG > L).astype(float) if "-team-total-home-" in suf else (AG > L).astype(float)
    if smt == "soccer_exact_score":
        m = re.search(r"-exact-score-(\d+)-(\d+)$", suf)
        if m:
            P = np.zeros((G, G)); P[int(m.group(1)), int(m.group(2))] = 1; return P
        if suf.endswith("any-other"):
            P = np.ones((G, G))
            for h, a in listed_scores:
                P[h, a] = 0
            return P
    return None


def main():
    H = float(sys.argv[1]) if len(sys.argv) > 1 else 48
    now = dt.datetime.now(dt.timezone.utc)
    evs, off = [], 0
    while True:
        d = get("https://gamma-api.polymarket.com/events", dict(tag_id=100350, closed="false", limit=100, offset=off,
                start_time_min=now.strftime("%Y-%m-%dT%H:%M:%SZ"), start_time_max=(now + dt.timedelta(hours=H)).strftime("%Y-%m-%dT%H:%M:%SZ")))
        evs += d; off += 100
        if len(d) < 100 or off > 3000:
            break
    games = {}
    for e in evs:
        m = re.match(r"^(.*\d{4}-\d{2}-\d{2})(-.*)?$", e["slug"])
        if not m:
            continue
        games.setdefault(m.group(1), {})[m.group(2) or "main"] = e
    toks = []
    for base, sib in games.items():
        if "main" not in sib or "-more-markets" not in sib:
            continue
        parts = base.split("-")
        home_code, away_code = parts[1], parts[2]
        listed = []
        for mk in (sib.get("-exact-score") or {}).get("markets", []):
            mm = re.search(r"-exact-score-(\d+)-(\d+)$", mk["slug"])
            if mm:
                listed.append((int(mm.group(1)), int(mm.group(2))))
        for key, e in sib.items():
            for mk in e.get("markets", []):
                tk = jl(mk.get("clobTokenIds"))
                if len(tk) != 2 or not mk.get("acceptingOrders", True):
                    continue
                smt = mk.get("sportsMarketType")
                P = payoff_A(mk["slug"], base, smt, mk.get("line") or 0, home_code, away_code, listed)
                ht = None
                if smt == "soccer_halftime_result":
                    ht = mk["slug"].rsplit("-", 1)[-1]
                if P is None and ht is None:
                    continue
                for j, t in enumerate(tk):
                    toks.append(dict(game=base, start=e.get("startTime"), slug=mk["slug"], smt=smt, side=j, token=t,
                                     fees=bool(mk.get("feesEnabled")), ht=ht, neg=bool(mk.get("negRisk")),
                                     P=None if P is None else (P if j == 0 else 1 - P)))
    T = pd.DataFrame(toks)
    print("games", T.game.nunique(), "tokens", len(T), flush=True)
    books = {}
    ids = T.token.tolist()
    for i in range(0, len(ids), 200):
        for b in get("https://clob.polymarket.com/books", post=[{"token_id": t} for t in ids[i:i + 200]]):
            asks = sorted(((float(x["price"]), float(x["size"])) for x in b.get("asks", [])))
            bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
            books[b["asset_id"]] = (asks[0] if asks else (np.nan, 0.0), bids[0] if bids else (np.nan, 0.0))
    T["ask"] = [books.get(t, ((np.nan, 0), (np.nan, 0)))[0][0] for t in T.token]
    T["ask_sz"] = [books.get(t, ((np.nan, 0), (np.nan, 0)))[0][1] for t in T.token]
    T["bid"] = [books.get(t, ((np.nan, 0), (np.nan, 0)))[1][0] for t in T.token]
    T["bid_sz"] = [books.get(t, ((np.nan, 0), (np.nan, 0)))[1][1] for t in T.token]
    T["cost"] = (T.ask + np.where(T.fees, FEE(T.ask), 0)).round(6)
    tab = Table()
    rows = []
    for g, d in T.groupby("game"):
        ft = d[d.P.notna() & d.ask.notna() & (d.ask < 1)]
        r = dict(game=g, start=d.start.iloc[0], n_tok=len(d), n_ft_ask=len(ft))
        if len(ft) >= 2:
            A = -np.stack([p.ravel() for p in ft.P]).T     # (states, tokens)
            res = linprog(ft.cost.to_numpy(), A_ub=A, b_ub=-np.ones(G * G), bounds=(0, None), method="highs")
            if res.status == 0:
                r["lp_cost"] = res.fun
                used = ft.iloc[np.where(res.x > 1e-6)[0]]
                r["lp_legs"] = ";".join(f"{s}[{sd}]@{a}x{x:.2f}(sz{z:.0f})" for s, sd, a, x, z in zip(used.slug, used.side, used.ask, res.x[res.x > 1e-6], used.ask_sz))
        ht = d[d.ht.notna() & (d.side == 0)]
        if len(ht) == 3 and ht.ask.notna().all():
            r["ht_yes_sum"] = ht.cost.sum()
        es = d[(d.smt == "soccer_exact_score") & (d.side == 0)]
        if len(es) and es.ask.notna().all():
            r["es_yes_sum"] = es.cost.sum(); r["es_n"] = len(es)
        # model deviation (two-sided, tight books only)
        mid = ((d.ask + d.bid) / 2).where((d.ask - d.bid) <= 0.05)
        ml = d[(d.smt == "moneyline") & (d.side == 0)]
        if len(ml) == 3 and mid[ml.index].notna().all():
            pm = [mid[ml.index[k]] for k in range(3)]
            Ps = list(ml.P)
            ph = sum(pm[k] for k in range(3) if Ps[k][1, 0] == 1); pa = sum(pm[k] for k in range(3) if Ps[k][0, 1] == 1)
            s = sum(pm); ix = tab.idx([ph / s], [pa / s])
            Mx = tab.M[ix][0]
            dev = []
            for k in d.index:
                if d.smt[k] in ("totals", "both_teams_to_score", "spreads", "soccer_team_totals") and pd.notna(mid[k]):
                    dev.append(mid[k] - float((Mx * d.P[k]).sum()))
            if dev:
                r["n_dev"] = len(dev); r["mean_abs_dev"] = float(np.mean(np.abs(dev)))
        rows.append(r)
    R = pd.DataFrame(rows)
    stamp = now.strftime("%Y%m%dT%H%M")
    R.to_csv(os.path.join(OUT, f"live_scan_{stamp}.csv"), index=False)
    T.drop(columns="P").to_parquet(os.path.join(OUT, f"live_tokens_{stamp}.parquet"))
    print(R.describe().to_string())
    if "lp_cost" in R:
        print("games with LP lock cost<1:", (R.lp_cost < 1 - 1e-9).sum(), "of", R.lp_cost.notna().sum())
        print(R.nsmallest(10, "lp_cost")[["game", "lp_cost", "lp_legs"]].to_string())


if __name__ == "__main__":
    main()
