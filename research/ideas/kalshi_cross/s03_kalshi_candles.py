"""s03 - download Kalshi 1-minute candlesticks (yes_bid / yes_ask OHLC + volume) for every matched game,
window [kickoff - 72h, kickoff + 5 min], for all 3 outcome markets.

Live markets: GET /series/{series}/markets/{ticker}/candlesticks ; settled before the historical cutoff:
GET /historical/markets/{ticker}/candlesticks (older schema, field names without _dollars).  Candles are sparse:
a row exists only for minutes where something changed.  Resumable: one parquet per event in data/candles/.
Output: data/candles/<event_ticker>.parquet with columns ticker, ts (minute END, unix s), bid_o/h/l/c, ask_o/h/l/c, vol
"""
import os, sys, time
import pandas as pd
import kalshi_api as k

os.makedirs("data/candles", exist_ok=True)
mt = pd.read_parquet("data/matches.parquet").sample(frac=1, random_state=7)   # random order: partial runs are an unbiased sample
CUT = pd.Timestamp(k.get("/historical/cutoff")["market_settled_ts"]).tz_localize(None)


def f(x, key):
    if x is None:
        return None
    v = x.get(key + "_dollars", x.get(key))
    return float(v) if v is not None else None


def candles(series, ticker, s, e, hist):
    path = f"/historical/markets/{ticker}/candlesticks" if hist else f"/series/{series}/markets/{ticker}/candlesticks"
    d = k.get(path, {"start_ts": s, "end_ts": e, "period_interval": 1})
    out = []
    for c in d.get("candlesticks", []):
        b, a = c.get("yes_bid") or {}, c.get("yes_ask") or {}
        out.append(dict(ticker=ticker, ts=c["end_period_ts"], bid_o=f(b, "open"), bid_h=f(b, "high"),
                        bid_l=f(b, "low"), bid_c=f(b, "close"), ask_o=f(a, "open"), ask_h=f(a, "high"),
                        ask_l=f(a, "low"), ask_c=f(a, "close"),
                        vol=float(c.get("volume_fp", c.get("volume")) or 0)))
    return out


n = 0
for r in mt.itertuples():
    fn = f"data/candles/{r.event_ticker}.parquet"
    if os.path.exists(fn):
        continue
    kick = int(pd.Timestamp(r.kick_utc).tz_localize("UTC").timestamp())
    s, e = kick - 72 * 3600, kick + 300
    hist = pd.Timestamp(r.k_close) < CUT if pd.notna(r.k_close) else True
    rows = []
    try:
        for t in (r.kt0, r.kt1, r.kt2):
            try:
                rows += candles(r.series, t, s, e, hist)
            except RuntimeError as ex:      # wrong partition -> try the other one
                rows += candles(r.series, t, s, e, not hist)
    except Exception as ex:
        print("FAIL", r.event_ticker, ex, flush=True)
        continue
    pd.DataFrame(rows, columns=["ticker", "ts", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l",
                                "ask_c", "vol"]).to_parquet(fn)
    n += 1
    if n % 100 == 0:
        print(n, r.event_ticker, r.kick_utc, flush=True)
print("done", n)
