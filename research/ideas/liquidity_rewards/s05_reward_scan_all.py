"""Step 5 (live, read-only, side question): where on ALL of Polymarket is the reward pool large relative to competing
qualifying liquidity? For the top-400 markets by daily reward rate (latest data/reward_config_*.json) fetch the CLOB market
(tokens, closed/accepting flags) and the Yes book, compute aggregate competing Q (same scoring as s01, Yes book only) and the
$/day a two-sided quote of $1,000 per side (at the touch-adjacent price, but at least min_size) would earn if the
competition stayed as observed. This is a pool-size/competition screen only: it says nothing about adverse selection.
Output out/reward_scan_all.csv.
"""
import os, glob, json, time, numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__))
S = requests.Session(); LAST = [0.0]


def get(url, method="GET", **kw):
    for a in range(6):
        w = 0.34 - (time.time() - LAST[0])
        if w > 0: time.sleep(w)
        LAST[0] = time.time()
        try: r = S.request(method, url, timeout=60, **kw)
        except Exception: time.sleep(2 ** a); continue
        if r.status_code == 429 or r.status_code >= 500: time.sleep(2 ** a + 1); continue
        return r
    return None


rw = json.load(open(sorted(glob.glob(os.path.join(HERE, "data", "reward_config_*.json")))[-1]))
cfg = pd.DataFrame(rw).T.reset_index().rename(columns={"index": "cond"})
cfg = cfg[cfg.rate > 0].sort_values("rate", ascending=False).head(400)
rows = []
for c in cfg.itertuples():
    r = get(f"https://clob.polymarket.com/markets/{c.cond}")
    if r is None or r.status_code != 200: continue
    m = r.json()
    if m.get("closed") or not m.get("accepting_orders"): continue
    toks = m.get("tokens") or []
    if not toks: continue
    rows.append(dict(cond=c.cond, q=m.get("question"), slug=m.get("market_slug"), tok=toks[0]["token_id"], rate=c.rate,
                     v=c.max_spread, ms=c.min_size, end=m.get("end_date_iso"), tags=",".join(m.get("tags") or [])))
df = pd.DataFrame(rows)
books = {}
for i in range(0, len(df), 100):
    r = get("https://clob.polymarket.com/books", method="POST", json=[{"token_id": t} for t in df.tok.iloc[i:i + 100]])
    if r is not None and r.status_code == 200:
        for b in r.json(): books[b["asset_id"]] = b
out = []
for r in df.itertuples(index=False):
    b = books.get(r.tok)
    if not b: continue
    bids = [(round(float(x["price"]), 4), float(x["size"])) for x in b.get("bids") or []]
    asks = [(round(float(x["price"]), 4), float(x["size"])) for x in b.get("asks") or []]
    bb = [p for p, s in bids if s >= r.ms]; aa = [p for p, s in asks if s >= r.ms]
    if not bb or not aa: continue
    mid = (max(bb) + min(aa)) / 2
    def q(lv, sign):
        t = 0.0
        for p, s in lv:
            d = round((mid - p) * sign * 100, 4)
            if s >= r.ms and 0 <= d < r.v: t += ((r.v - d) / r.v) ** 2 * s
        return t
    q1, q2 = q(bids, 1), q(asks, -1)
    # our quote: one tick inside nothing, i.e. at the best bid / best ask level (joining the touch)
    tb, ta = max(p for p, _ in bids), min(p for p, _ in asks)
    sb, sa = (mid - tb) * 100, (ta - mid) * 100
    Sb = ((r.v - sb) / r.v) ** 2 if sb < r.v else 0; Sa = ((r.v - sa) / r.v) ** 2 if sa < r.v else 0
    nb = max(1000 / max(tb, 0.01), r.ms); na = max(1000 / max(1 - ta, 0.01), r.ms)
    ours = min(nb * Sb, na * Sa)
    if 0.10 <= mid <= 0.90: ours = max(ours, max(nb * Sb, na * Sa) / 3)
    comp = (q1 + q2) / 2
    share = ours / (ours + comp) if ours > 0 else 0
    out.append(dict(r._asdict(), mid=mid, spread=ta - tb, Q1=q1, Q2=q2, share_1k=share, usd_day_1k=share * r.rate))
o = pd.DataFrame(out).sort_values("usd_day_1k", ascending=False)
o.to_csv(os.path.join(HERE, "out", "reward_scan_all.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(len(o), "markets scanned; $/day for $1k/side at the touch (central competition):")
print(o[["q", "rate", "v", "ms", "mid", "spread", "Q1", "Q2", "share_1k", "usd_day_1k"]].head(30).round(3).to_string())
print(o.usd_day_1k.describe(percentiles=[.5, .9, .99]).round(2))
