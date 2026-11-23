"""sweep_v1 — "catching panic sales" (research/ideas/sweep_catch, tuned in s10/s11). PAPER ONLY: no order is ever sent.

Every 5 minutes (its own Lambda, same image as predict_winner_model, handler sweep.lambda_handler):
  1. Orders of the previous run are checked against REAL Polymarket trades: an order "fills" only when a resting
     order of someone else was hit strictly BELOW our price (a seller swept through our level), so by price
     priority ours would have been filled first. Fill size = shares that traded below our price in that sweep
     (capped at the order size).
  2. A filled position rests a sell order at the pre-sweep price floor(c). It sells only as far as real buyers
     traded strictly above that price (shares capped by their volume). Whatever is left 5 minutes before kickoff
     is sold 1c under the last real bid-hit print of the final 2 hours (taker fee paid); if there was none, it is
     held to the result and settled from Polymarket's resolution.
  3. New orders: for every matched game kicking off in 10 min – 3 h, for each outcome and BOTH sides (buy "Yes"
     catches dumps, buy "No" catches pumps), while the token's price c is 8–92c, two resting buys of SHARES each:
        level A  L = floor(min(c - 15c, 0.60 c))     level B  L = floor(min(c - 20c, 0.50 c))
     only if Winner's fair probability of that token is >= L + 6c (the sweep pushed the price AWAY from fair).
     After a fill the same (game, outcome, side, level) is not quoted for 60 minutes.
Backtest (16k games, real trades, Sep 2025 - Nov 2026, 1,000 shares per outcome-side): +72% / +69% / +100% per $
in train / validation / test, ~6 fills a month, ~$150-220 a month, ~$4-8k of resting orders.

Firestore: meta/sweep_universe (written by predict_winner_model every 3 h: matched games, tokens, Winner fair),
meta/sweep_state (orders of the last run + cooldowns), meta/sweep_status (last run summary, for the site),
sweep_positions/{id} (one document per fill).
"""
import os, json, time, logging, traceback
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np, pandas as pd

log = logging.getLogger("sweep"); log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

MODEL = "sweep_v1"
POS_COL = "sweep_positions"
WINDOW_MIN, STOP_MIN, LIQ_MIN = 180, 10, 5       # quote kick-3h .. kick-10min; liquidate leftovers at kick-5min
BAND = (0.08, 0.92)
LEVELS = {"A": (0.15, 0.40), "B": (0.20, 0.50)}  # L = floor(min(c - dm, c * (1 - f)))
GAP, LMIN, COOL_MIN = 0.06, 0.02, 60
SHARES = float(os.environ.get("SWEEP_SHARES", "500"))          # per order (level); 2 levels = 1,000 per outcome-side
MAX_ORDER_AGE_MIN = 20       # an order missed by a failed run is checked for at most this long
UNIVERSE_MAX_AGE_H = 6.5
TOL = 5e-5
TRADES = "https://data-api.polymarket.com/v2/trades"
MIDPOINTS = "https://clob.polymarket.com/midpoints"
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
_S = requests.Session()


def floor_c(v):
    return float(np.floor(v * 100 + 1e-6) / 100)


def fee(p, rate=0.05):
    return rate * p * (1 - p)


def _get(url, params, tries=3):
    for i in range(tries):
        try:
            r = _S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 * (i + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def prices_now(tokens):
    """Midpoint of each Yes token (CLOB /midpoints, one request per 100 tokens). {token: price}."""
    out = {}
    for i in range(0, len(tokens), 100):
        r = _S.post(MIDPOINTS, json=[{"token_id": t} for t in tokens[i:i + 100]], timeout=20)
        r.raise_for_status()
        out.update({k: float(v) for k, v in (r.json() or {}).items() if v is not None})
    return out


def maker_rows(conds, since_ts, max_pages=15):
    """MAKER fills of up to ~10 markets since `since_ts` (taker_only=false lists each transaction's taker row first,
    then its makers; the API is newest first and ignores time filters). Per market a DataFrame of: ts, tx,
    ypx = price of the market's "Yes" share, long = the maker ended up longer "Yes" (a resting bid on Yes or ask on
    No was hit), size."""
    rows, cur = [], None
    for _ in range(max_pages):
        j = _get(TRADES, dict(condition_id=",".join(conds), limit=1000, taker_only="false", **({"cursor": cur} if cur else {})))
        data = (j or {}).get("data") or []
        rows.extend(data)
        cur = ((j or {}).get("pagination") or {}).get("next_cursor")
        if not data or not cur or min(int(x["timestamp"]) for x in data) < since_ts:
            break
    else:
        log.warning("trades of %s: page limit reached before %s", conds, since_ts)
    empty = pd.DataFrame(dict(ts=pd.Series(dtype=int), tx=pd.Series(dtype=str), ypx=pd.Series(dtype=float),
                              long=pd.Series(dtype=bool), size=pd.Series(dtype=float)))
    out = {c: empty for c in conds}
    if not rows:
        return out
    d = pd.DataFrame(rows)
    d["taker"] = ~d.duplicated("transaction_hash")            # first row of each transaction = its taker
    d = d[~d.taker & (d.timestamp.astype(int) >= since_ts)]
    yes = d.outcome_index.astype(int) == 0
    px = d.price.astype(float)
    buy = d.side == "BUY"
    t = pd.DataFrame(dict(cond=d.condition_id.values, ts=d.timestamp.astype(int).values, tx=d.transaction_hash.values,
                          ypx=np.round(np.where(yes, px, 1 - px), 4), long=np.where(yes, buy, ~buy),
                          size=d["size"].astype(float).values))
    for c, g in t.groupby("cond"):
        if c in out:
            out[c] = g.drop(columns="cond").sort_values("ts", kind="stable").reset_index(drop=True)
    return out


def ours(m, s):
    """(price of OUR token, prints where a resting BID on our token was hit, prints where an ASK was hit)."""
    px = m.ypx if s > 0 else np.round(1 - m.ypx, 4)
    bid_hit = m.long if s > 0 else ~m.long
    return px, bid_hit, ~bid_hit


def _db():
    from google.cloud import firestore
    if not hasattr(_db, "c"):
        _db.c = firestore.Client()
    return _db.c


def run(now=None, dry=False, time_left=lambda: 1e9):
    now = pd.Timestamp.now("UTC") if now is None else pd.Timestamp(now)
    now_ts = int(now.timestamp())
    db = _db()
    meta = db.collection("meta")
    uni = meta.document("sweep_universe").get().to_dict() or {}
    st_ref = meta.document("sweep_state"); st = st_ref.get().to_dict() or {}
    games = {g["uid"]: g for g in uni.get("games", [])}
    cool = {k: v for k, v in (st.get("cool") or {}).items() if v > now_ts}
    res = dict(run=now.isoformat(), checked_orders=len(st.get("orders") or []), fills=0, exits=0, liquidated=0, held=0,
               settled=0, orders=0, games_quoted=0, errors=0)

    # ---------------- trades needed: last run's orders + open positions
    pos_docs = [(d.id, d.to_dict()) for d in db.collection(POS_COL).where("status", "in", ["open", "held"]).stream()]
    need = {}
    for o in st.get("orders") or []:
        need[o["cond"]] = min(need.get(o["cond"], 10 ** 12), o["placed_ts"])
    for pid, p in pos_docs:
        if p["status"] == "open":                     # liquidation (kick-5) also needs the bids of the final 2 hours
            since = p["checked_ts"] if now_ts < p["kick_ts"] - LIQ_MIN * 60 else \
                min(p["checked_ts"], max(p["fill_ts"], p["kick_ts"] - 120 * 60))
            need[p["cond"]] = min(need.get(p["cond"], 10 ** 12), since)
    tape = {}
    conds = sorted(need, key=lambda c: need[c])
    groups = [conds[i:i + 9] for i in range(0, len(conds), 9)]
    def fetch(grp):
        try:
            return grp, maker_rows(grp, min(need[c] for c in grp) - 1)
        except Exception as e:
            log.warning("trades %s failed: %s", grp, e); return grp, None
    with ThreadPoolExecutor(6) as ex:
        for grp, r in ex.map(fetch, groups):
            if r is None:
                res["errors"] += 1
            else:
                tape.update(r)

    # ---------------- 1. fills of last run's orders
    new_pos = []
    for o in st.get("orders") or []:
        m = tape.get(o["cond"])
        if m is None:
            continue
        until = min(now_ts, o["placed_ts"] + MAX_ORDER_AGE_MIN * 60, o["kick_ts"] - STOP_MIN * 60)
        px, bid_hit, _ = ours(m, o["s"])
        hit = bid_hit & (px < o["L"] - TOL) & (m.ts >= o["placed_ts"]) & (m.ts <= until)
        if not hit.any():
            continue
        i = int(np.flatnonzero(hit.values)[0])
        tx = m.tx.iloc[i]
        through = float(m["size"][(m.tx == tx) & hit].sum())
        q = min(SHARES, through)
        g = games.get(o["uid"], {})
        key = f'{o["uid"]}_{o["o"]}_{o["s"]}_{o["lvl"]}'
        cool[key] = int(m.ts.iloc[i]) + COOL_MIN * 60
        pid = f'{key}_{int(m.ts.iloc[i])}'
        doc = dict(model_name=MODEL, game_id=o["uid"], outcome=o["o"], side=o["s"], level=o["lvl"],
                   cond=o["cond"], fee_rate=o.get("fee_rate", 0.05),
                   home_team=g.get("home"), away_team=g.get("away"), english_home_team=g.get("en_home"),
                   english_away_team=g.get("en_away"), league=g.get("league"), english_league=g.get("en_league"),
                   poly_title=g.get("poly_title"), poly_slug=g.get("poly_slug"), match_time=g.get("match_time"),
                   kick_ts=o["kick_ts"], c=o["c"], L=o["L"], T=o["T"], wf=o["wf"], placed_ts=o["placed_ts"],
                   fill_ts=int(m.ts.iloc[i]), fill_tx=tx, sweep_low=float(px[(m.tx == tx)].min()),
                   through_shares=round(through, 2), q=round(q, 4), stake=round(q * o["L"], 4),
                   q_exit=0.0, checked_ts=int(m.ts.iloc[i]), status="open",
                   mins_to_kick=round((o["kick_ts"] - int(m.ts.iloc[i])) / 60, 1),
                   filled_at=pd.Timestamp(int(m.ts.iloc[i]), unit="s").strftime("%Y-%m-%d %H:%M"),
                   recorded_at=now.isoformat())
        new_pos.append((pid, doc)); res["fills"] += 1

    # ---------------- 2. exits (real buyers above our sell price), liquidation at kick-5, settlement of held
    work = sorted(pos_docs + new_pos, key=lambda x: x[1]["fill_ts"])
    used = {}                                                  # (cond, side, T) -> shares already given to positions
    updates = {}
    for pid, p in work:
        if p["status"] != "open":
            continue
        m = tape.get(p["cond"])
        if m is None:
            continue
        stop = p["kick_ts"] - LIQ_MIN * 60
        px, bid_hit, ask_hit = ours(m, p["side"])
        upto = min(now_ts, stop)
        new = ask_hit & (px > p["T"] + TOL) & (m.ts > p["checked_ts"]) & (m.ts <= upto)   # checked_ts starts at fill
        vol = float(m["size"][new].sum())
        k = (p["cond"], p["side"], p["T"])
        avail = max(vol - used.get(k, 0.0), 0.0)
        take = min(avail, p["q"] - p["q_exit"])
        used[k] = used.get(k, 0.0) + take
        p = dict(p, q_exit=round(p["q_exit"] + take, 4), checked_ts=max(p["checked_ts"], upto))
        if take > 0:
            p["exit_last_ts"] = upto
        if p["q_exit"] >= p["q"] - 1e-9:
            p.update(status="closed", how="target", pnl=round(p["q"] * (p["T"] - p["L"]), 4), closed_at=now.isoformat())
            res["exits"] += 1
        elif now_ts >= stop:
            rest = p["q"] - p["q_exit"]
            w = bid_hit & (m.ts > p["fill_ts"]) & (m.ts >= p["kick_ts"] - 120 * 60) & (m.ts <= stop)
            if w.any():
                vb = max(float(px[w].iloc[-1]) - 0.01, 0.001)
                pnl = p["q_exit"] * (p["T"] - p["L"]) + rest * (vb - fee(vb, p.get("fee_rate", 0.05)) - p["L"])
                p.update(status="closed", how="target+liquidated" if p["q_exit"] > 0 else "liquidated",
                         liq_price=round(vb, 4), pnl=round(pnl, 4), closed_at=now.isoformat())
                res["liquidated"] += 1
            else:
                p.update(status="held"); res["held"] += 1
        updates[pid] = p
    for pid, p in pos_docs:                                    # held to the result: settle from Polymarket
        if p["status"] != "held" or now_ts < p["kick_ts"] + 2 * 3600:
            continue
        try:
            j = _get(GAMMA_MARKETS, dict(condition_ids=p["cond"], closed="true"))
            if not j:
                continue
            op = [float(x) for x in json.loads(j[0].get("outcomePrices") or "[]")]
            if len(op) != 2 or sorted(op) != [0.0, 1.0]:
                continue
            won = op[0] if p["side"] > 0 else op[1]
            rest = p["q"] - p["q_exit"]
            p = dict(p, status="closed", how="held to result", won=bool(won),
                     pnl=round(p["q_exit"] * (p["T"] - p["L"]) + rest * (won - p["L"]), 4), closed_at=now.isoformat())
            updates[pid] = p; res["settled"] += 1
        except Exception as e:
            log.warning("settle %s failed: %s", pid, e)

    # ---------------- 3. new orders
    orders = []
    age_h = (now - pd.Timestamp(uni["updated"])).total_seconds() / 3600 if uni.get("updated") else 99
    todo = [g for g in games.values() if STOP_MIN * 60 < g["kick_ts"] - now_ts <= WINDOW_MIN * 60]
    if age_h <= UNIVERSE_MAX_AGE_H and todo:
        toks = [(g["uid"], oc) for g in todo for oc in g["outcomes"]]
        mids = prices_now([oc["token"] for _, oc in toks])
        for uid, oc in toks:
            c = mids.get(oc["token"])
            if c is None or oc.get("wf") is None:
                continue
            g = games[uid]
            for s in (1, -1):
                cs = round(c if s > 0 else 1 - c, 4)
                if not BAND[0] <= cs <= BAND[1]:
                    continue
                wf = oc["wf"] if s > 0 else 1 - oc["wf"]
                for lvl, (dm, f) in LEVELS.items():
                    L = floor_c(min(cs - dm, cs * (1 - f)))
                    key = f'{uid}_{oc["o"]}_{s}_{lvl}'
                    if L < LMIN or wf - L < GAP - 1e-9 or key in cool:
                        continue
                    orders.append(dict(uid=uid, o=oc["o"], s=s, lvl=lvl, cond=oc["cond"],
                                       fee_rate=oc.get("fee_rate", 0.05), kick_ts=g["kick_ts"], c=cs, L=L,
                                       T=max(floor_c(cs), round(L + 0.01, 2)), wf=round(wf, 4), placed_ts=now_ts))
        res["games_quoted"] = len({o["uid"] for o in orders})
    elif age_h > UNIVERSE_MAX_AGE_H:
        res["skipped"] = f"universe is {age_h:.1f} h old"
    res["orders"] = len(orders)

    if dry:
        res["new_positions"] = [d for _, d in new_pos]
        res["sample_orders"] = orders[:10]
        return res
    batch = db.batch()
    for pid, p in updates.items():
        batch.set(db.collection(POS_COL).document(pid), p)
    batch.commit()
    st_ref.set(dict(orders=orders, cool=cool, last_run=now.isoformat()))
    live = sorted(orders, key=lambda o: o["kick_ts"])
    meta.document("sweep_status").set(dict(
        last_run=now.isoformat(), universe_updated=uni.get("updated"), games_in_window=len(todo),
        games_quoted=res["games_quoted"], orders=len(orders), locked_usd=round(sum(o["L"] * SHARES for o in orders), 2),
        shares_per_order=SHARES, result=res,
        live=[dict(uid=o["uid"], o=o["o"], s=o["s"], lvl=o["lvl"], c=o["c"], L=o["L"], kick_ts=o["kick_ts"],
                   title=games[o["uid"]].get("poly_title"), home=games[o["uid"]].get("home"), away=games[o["uid"]].get("away"))
              for o in live[:300]]))
    return res


def lambda_handler(event, context):
    dry = bool((event or {}).get("dry_run")) or os.environ.get("DRY_RUN") == "1"
    try:
        out = run(dry=dry)
    except Exception as e:
        log.error("sweep failed: %s\n%s", e, traceback.format_exc())
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
    log.info("result %s", out)
    return {"statusCode": 200, "body": json.dumps(out, default=str)}
