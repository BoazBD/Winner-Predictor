"""s01: build the compact print tape used to replay resting deep orders.

Source: research/wallets/data/trades_marked.parquet (every pre-start fill with taker AND maker rows, ~15k games,
kickoffs Sep 2025 - Nov 2026). A MAKER row is an exact price level that a taker hit; a sweep shows up as several maker
rows at successively worse prices inside one transaction (tx).

Kept: maker rows with rel_min in [-72h, 0) (the 1-minute mid series of research/lead_lag covers the last 72 h).
Columns: unique_id, k (Winner orientation), ts, ypx (Yes-equivalent price of outcome k, float64 rounded 4dp), size,
  mlong (the MAKER got longer k = the TAKER sold k: these prints fill resting BIDS on k),
  tx, twid (taker wallet id of that tx), tlev (distinct maker prices in the tx), tusd (taker usd of the tx), kick_ts.
Output: data/tape.parquet (sorted by unique_id, ts, tx)
"""
import os, time
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); os.makedirs(D, exist_ok=True)
SRC = os.path.join(HERE, "..", "..", "wallets", "data", "trades_marked.parquet")


def main():
    t0 = time.time()
    c = duckdb.connect(config=dict(threads=2, memory_limit="2GB"))
    c.sql(f"""
    copy (
      with base as (select unique_id, k, ts, round(ypx, 4) ypx, size, taker, long, tx, wid, tx_levels, usd, kick_ts
                    from '{SRC}' where rel_min >= -4320 and rel_min < 0),
      tk as (select tx, any_value(wid) twid, any_value(tx_levels) tlev, sum(usd) tusd from base where taker group by tx)
      select b.unique_id, b.k::TINYINT k, b.ts, b.ypx, b.size::FLOAT size, b.long mlong, b.tx, tk.twid, tk.tlev,
             tk.tusd::FLOAT tusd, b.kick_ts
      from base b join tk using (tx) where not b.taker
      order by b.unique_id, b.ts, b.tx
    ) to '{os.path.join(D, "tape.parquet")}' (format parquet)""")
    print(c.sql(f"select count(*) n, count(distinct unique_id) g, avg((tlev>1)::int) sweep_share "
                f"from '{os.path.join(D, 'tape.parquet')}'").df(), flush=True)
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
