"""Step 7: build REPORT.html (self-contained, charts as base64 PNG) and REPORT.md (same text; charts saved in out/)."""
import os, json, base64, html
import numpy as np, pandas as pd
from common import HERE
import charts as C

O = os.path.join(HERE, "out")
R = pd.read_csv(os.path.join(O, "final_summary.csv"))
TR = pd.read_parquet(os.path.join(O, "final_trades.parquet"))
MM = pd.read_csv(os.path.join(O, "final_monthly.csv"))
SENS = pd.read_csv(os.path.join(O, "final_sensitivity.csv"))
BD = pd.read_csv(os.path.join(O, "byd.csv"))
FR = pd.read_csv(os.path.join(O, "fill_rates.csv"))
ER = pd.read_csv(os.path.join(O, "entry_reality.csv"))
ARB = json.load(open(os.path.join(O, "arb.json")))
SL = pd.read_csv(os.path.join(O, "slices.csv"))
N_MID = int(open(os.path.join(O, "n_configs_mid.txt")).read())
N_TR = int(open(os.path.join(O, "n_configs_trades.txt")).read())
N_SL = SL.groupby(["fam", "ent", "d", "slice"]).ngroups
N_ALL = N_MID + N_TR + N_SL
N_TEST = R.name.nunique()

FAM = {"F1": "Winner moved first", "F2": "Polymarket below Winner fair value", "F3": "Fade a sharp Polymarket jump",
       "F4": "Stale outcome (3 prices don't add to 100%)", "F6": "No signal, fixed time (control)"}
ENT = {"taker": "buy at the ask (taker)", "taker_v": "buy at the ask, only if someone really bought there",
       "bid60": "limit buy at the bid, 1 h", "bid240": "limit buy at the bid, 4 h", "mid60": "limit buy at mid, 1 h"}
EXIT = {"pnl_kick": "sell at market 5 min before kickoff", "pnl_t180": "sell at market after 3 h",
        "pnl_t720": "sell at market after 12 h", "pnl_hold": "hold to the result"}


def pct(x, nd=1):
    return "–" if pd.isna(x) else f"{x * 100:+.{nd}f}%"


def cfg_text(r):
    d = "Winner's fair price" if r.d < 0 else f"+{r.d * 100:.0f}c"
    sub = "" if str(r["sub"]) in ("nan", "None") else f", outcome {['home', 'draw', 'away'][int(eval(r['sub'])[0])]} {'Yes' if eval(r['sub'])[1] > 0 else 'No'}"
    thr = f"{r.thr:.0f} h before kickoff" if r.fam == "F6" else f"signal ≥ {r.thr * 100:.0f} pts"
    filt = ", ".join(x for x in [r.time if r.time != "all" else "", f"price {r.price}", r.vol if r.vol != "all" else "",
                                 r.tier if r.tier != "all" else "", "coherent book" if r.book == "coherent" else ""] if x)
    return f"{thr}; {ENT[r.ent]}; sell limit {d}; if unfilled: {EXIT[r.exit]}; {filt}{sub}"


# ---------------------------------------------------------------- charts
names_A = sorted(n for n in R.name.unique() if n.startswith("A-"))
names_B = sorted(n for n in R.name.unique() if n.startswith("B-"))
img = {}
img["cumA"] = C.cum_pnl(TR, names_A, "Cumulative profit: the 5 closed-before-kickoff finalists (real-trade fills)")
img["cumZ"] = C.cum_pnl(TR, ["Z-F2"], "The midpoint mirage Z-F2 (entry prices not really available)")
img["cumB"] = C.cum_pnl(TR, names_B, "Cumulative profit: finalists allowed to hold to the result")
img["fills"] = C.fill_bars(FR)
img["byd"] = C.by_d(BD, names_A)
img["monthZ"] = C.monthly(MM, "Z-F2")
img["monthA"] = C.monthly(MM, "A-F6")
img["sens"] = C.mid_vs_trades(SENS, names_A + ["Z-F2"])
for k, v in img.items():
    open(os.path.join(O, f"fig_{k}.png"), "wb").write(base64.b64decode(v))

# ---------------------------------------------------------------- tables
piv = R.set_index(["name", "split"])


def row_of(n):
    r = R[(R.name == n) & (R.split == "train")].iloc[0]
    v = piv.loc[(n, "val")]; t = piv.loc[(n, "test")]
    return dict(name=n, family=FAM[r.fam], rules=cfg_text(r),
                train=f"{pct(piv.loc[(n, 'train')].ret)} (n={int(piv.loc[(n, 'train')].n)})",
                val=f"{pct(v.ret)} [{pct(v.ret_lo, 0)}, {pct(v.ret_hi, 0)}] n={int(v.n)}",
                test=f"{pct(t.ret)} [{pct(t.ret_lo, 0)}, {pct(t.ret_hi, 0)}] n={int(t.n)}",
                entry_fill=f"{v.entry_fill * 100:.0f}%", exit_fill=f"{v.exit_fill * 100:.0f}% / {t.exit_fill * 100:.0f}%",
                win=f"{t.win * 100:.0f}%", cents=f"{t.cents:+.1f}c / {t.cents_median:+.1f}c",
                worst=f"{t.worst_cents:.0f}c", worst_day=f"${t.worst_day_usd:.0f}",
                month=f"${t.usd_per_month_at_100:,.0f}", cap=f"${t.capital_usd_at_100:,.0f}",
                held="0%" if r.exit != "pnl_hold" else f"{t.held * 100:.0f}%")


rows = [row_of(n) for n in names_A + names_B + ["Z-F2"]]
COLS = [("name", "Strategy"), ("family", "Idea"), ("rules", "Exact rules (chosen on train)"), ("train", "Train return per $"),
        ("val", "Validation return per $ [95% CI]"), ("test", "Final test return per $ [95% CI]"),
        ("entry_fill", "Buy order filled (val)"), ("exit_fill", "Sell limit filled before kickoff (val / test)"),
        ("held", "Held to result (test)"), ("win", "Win rate (test)"), ("cents", "Avg / median profit per share (test)"),
        ("worst", "Worst trade per share (test)"), ("worst_day", "Worst day at $100/trade (test)"),
        ("month", "Profit per month at $100/trade (test)"), ("cap", "Capital needed at $100/trade (test peak)")]


def html_table(rows, cols, cls="t"):
    h = "".join(f"<th>{html.escape(c[1])}</th>" for c in cols)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(r[c[0]]))}</td>" for c in cols) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def md_table(rows, cols):
    s = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(r[c[0]]).replace("|", "/") for c in cols) + " |\n"
    return s


er = ER.copy()
er_rows = [dict(fam=FAM[r.fam], book=r.book, n=f"{int(r.signals):,}", pr=f"{r.with_real_buy_30m * 100:.0f}%",
                ex=f"{r.median_excess_cents:+.1f}c") for r in er.itertuples()]
ER_COLS = [("fam", "Signal"), ("book", "Sum of the 3 Yes prices"), ("n", "Signals"),
           ("pr", "Someone really bought within 30 min"), ("ex", "Real price paid minus our model's ask (median)")]
fr_rows = [dict(rule=r.rule, ent=ENT[r.ent], ef=f"{r.entry_fill * 100:.0f}%",
                xf="–" if pd.isna(r.exit_fill_2c) else f"{r.exit_fill_2c * 100:.0f}%") for r in FR.itertuples()]
FR_COLS = [("rule", "Fill rule"), ("ent", "Buy order"), ("ef", "Buy filled"), ("xf", "Then +2c sell filled before kickoff")]
sens_rows = []
for n in names_A + ["Z-F2"]:
    for v in SENS[SENS.name == n].variant.unique():
        x = SENS[(SENS.name == n) & (SENS.variant == v)].set_index("split")
        sens_rows.append(dict(name=n, v=v, tr=pct(x.ret.get("train")), va=pct(x.ret.get("val")), te=pct(x.ret.get("test"))))
SENS_COLS = [("name", "Strategy"), ("v", "Assumption"), ("tr", "Train"), ("va", "Validation"), ("te", "Test")]
sl = SL.pivot_table(index=["fam", "ent", "d", "slice"], columns="split", values="ret").reset_index()
sl_rows = [dict(fam=FAM[r.fam], ent=ENT[r.ent], d="Winner fair" if r.d < 0 else f"+{r.d * 100:.0f}c", s=r.slice,
                tr=pct(r.train), va=pct(r.val)) for r in sl.itertuples()]
SL_COLS = [("fam", "Signal"), ("ent", "Buy"), ("d", "Sell target"), ("s", "Slice"), ("tr", "Train"), ("va", "Validation")]
monthly_rows = []
for n in names_A + ["Z-F2"]:
    for r in MM[MM.name == n].sort_values("month").itertuples():
        monthly_rows.append(dict(name=n, m=r.month, sp=r.split, n=r.n, ret=pct(r.ret), usd=f"${r.usd:,.0f}"))
MON_COLS = [("name", "Strategy"), ("m", "Month"), ("sp", "Period"), ("n", "Trades"), ("ret", "Return per $"), ("usd", "Profit at $100/trade")]

live, mid, pr = ARB["live"], ARB["mid"], ARB["prints"]
ex = pd.DataFrame(pr["examples"]) if pr["examples"] else pd.DataFrame(columns=["min_usd", "cost"])
zt = piv.loc[("Z-F2", "test")]; zv = piv.loc[("Z-F2", "val")]; ztr = piv.loc[("Z-F2", "train")]
a6v, a6t = piv.loc[("A-F6", "val")], piv.loc[("A-F6", "test")]
best_A_val = max(names_A, key=lambda n: piv.loc[(n, "val")].ret)
bav, bat = piv.loc[(best_A_val, "val")], piv.loc[(best_A_val, "test")]
nA_pos_test = sum(piv.loc[(n, "test")].ret > 0 for n in names_A)
nB_pos_test = sum(piv.loc[(n, "test")].ret > 0 for n in names_B)

# ---------------------------------------------------------------- text (same for HTML and MD)
S = []   # list of (kind, content): kind in h2, p, ul, table, img, h3, note


def h2(t): S.append(("h2", t))
def h3(t): S.append(("h3", t))
def p(t): S.append(("p", t))
def ul(items): S.append(("ul", items))
def tab(rows, cols): S.append(("table", (rows, cols)))
def fig(k, cap): S.append(("img", (k, cap)))


h2("1. Verdict")
p(f"<b>No. We found no buy-then-limit-sell strategy that makes money after realistic costs, and none that is close to risk-free.</b> "
  f"We tested 5 families of ideas on 16,763 matched games (September 2025 to November 2026), checked fills against "
  f"5.2 million real Polymarket trades, and tried {N_ALL:,} rule combinations on the tuning and validation periods. "
  f"Every finalist that closes its trade before kickoff lost money on validation and again on the untouched final test "
  f"(October–November 2026): the best of them lost {pct(bav.ret)} per $ on validation and {pct(bat.ret)} on test. "
  f"The rules that looked very profitable on midpoint prices (up to +27% per $) are an illusion. They buy at prices that "
  f"only exist as the midpoint of an empty order book, where real buyers paid 4–21 cents more. The pattern is always the same: "
  f"the +1c to +5c target fills only 20–60% of the time, and the trades that don't fill lose more than the filled ones make. "
  f"A limit sell above your buy price changes the <i>shape</i> of the result (many small wins, fewer bigger losses). It does not "
  f"create an edge. True arbitrage (buying all three outcomes for under $1) does not exist after fees in the live books we checked; "
  f"in the history it appears only for a dollar or two at a time.")

h2("2. Recommended strategy")
p("<b>None. Do not trade any of these with real money.</b> No strategy passed validation (validation return > 0 with its 95% "
  "confidence interval above 0) using realistic fills, so there is nothing to recommend. The table below gives the exact rules "
  "of every finalist, so the claim can be checked.")
p(f"<b>What “least bad” looks like:</b> {best_A_val}, “{FAM[R[R.name == best_A_val].fam.iloc[0]]}”. Rules: "
  f"{cfg_text(R[(R.name == best_A_val) & (R.split == 'train')].iloc[0])}. "
  f"Validation: {pct(bav.ret)} per $ ({int(bav.n)} trades); final test: {pct(bat.ret)} ({int(bat.n)} trades). "
  f"The sell limit filled before kickoff in {bav.exit_fill * 100:.0f}% (validation) and {bat.exit_fill * 100:.0f}% (test) of trades. "
  f"Unfilled trades were sold at market 5 minutes before kickoff, for an average of {bat.avg_loss_unfilled_cents:+.1f}c per share. "
  f"It is the least bad only because it almost never trades (its buy order filled about {bav.entry_fill * 100:.0f}% of the time).")
p("<b>If you still want to experiment:</b> run it as paper trading only, with maker (limit) orders. Change nothing about the "
  "rules, and evaluate on kickoffs after 2026-11-20. The live models' CLV of −4% to −6.6% points the same way as this study: "
  "prices you can actually buy at tend to move against you.")

h2("3. All strategies compared")
p("Each finalist is the best rule set of its family on the <b>train</b> period (kickoff before 2026-07-01), ranked by a "
  "cautious score (average return minus 2 standard errors, games resampled). It was then checked on <b>validation</b> "
  "(Jul–Sep 2026) and evaluated <b>once</b> on the <b>final test</b> (kickoff from 2026-10-01). "
  "Track A (“A-”) must close before kickoff, as the owner asked. Track B (“B-”) may hold unfilled shares to the result. "
  "Z-F2 is the best rule set when we let the model buy at midpoint-based prices in empty books. It is shown only to expose "
  "that mirage. Returns are per $ staked at the entry price, after the taker fee where we pay it. Fills are checked against "
  "real trades (a resting order fills only if a real trade printed strictly through our price). “Capital” is the peak number "
  "of open positions × $100.")
tab(rows, COLS)
p(f"Of the 5 closed-before-kickoff finalists, {nA_pos_test} made money on the test period. Of the 5 hold-to-result finalists, "
  f"{nB_pos_test} did, and those are coin-flips with confidence intervals of about ±10–50%. "
  f"In total, {N_TEST} rule sets were evaluated on test, once each.")

h2("4. Charts")
fig("cumA", "Cumulative profit at $100 per trade, by kickoff date. Grey = tuning, blue = validation, orange = final test. "
            "All five drift down out of sample.")
fig("cumZ", "Z-F2, the ‘mirage’: buying Polymarket ≥8 pts below Winner at ‘midpoint + half-spread’. Most of the paper profit "
            "came in a few months of 2025, when many books were empty, and it nearly stops on the final test. Real buyers "
            "could not get these prices (section 6).")
fig("cumB", "Finalists allowed to hold unfilled shares to the result. Their curves are mostly settlement luck, so they swing a lot.")
fig("fills", "How often orders fill, over all F1–F4 signals in train and validation. ‘Through’ means a real trade printed "
             "strictly better than our limit; ‘touch’ allows a trade at exactly our price, which is optimistic. Midpoint "
             "rules are what you get without trade data.")
fig("byd", "Return by profit target d for the closed-before-kickoff finalists (train solid, validation dashed). No target "
           "makes them profitable out of sample.")
fig("monthZ", "The midpoint mirage by month: large in the tuning months, shrinking, close to zero in the final test.")
fig("monthA", "A-F6 (no-signal control, maker buy, +1c target) by month: small, steady losses.")
fig("sens", "The same finalists under different fill assumptions (train + validation average). Midpoint-based fills are "
            "far too kind for the F2 rules. With real trades every rule is negative or near zero.")

h2("5. What we tested")
ul([
    "<b>F1 Winner moved first.</b> Winner's margin-free probability moved ≥2/3/5 points in 6 h, and Polymarket moved less "
    "than half of that, so we follow Winner. This is the existing move_v1 idea.",
    "<b>F2 Model / fair-value gap.</b> Polymarket is ≥3/5/8 points below Winner's margin-free probability (or above it, so "
    "we buy No). The sell target can be Winner's fair price. Winner's probability stands in for the live models, whose "
    "historical scores we did not rebuild. The live CLV of −4% to −6.6% suggests they would not do better.",
    "<b>F3 Overreaction fade.</b> Polymarket moved ≥5/8/12 points in an hour while Winner moved less than 1 point that way, "
    "so we buy the other side.",
    "<b>F4 Stale outcome.</b> The three Yes prices add up to more or less than 100% by ≥3/5/8 points after one outcome moved. "
    "We buy (or sell via No) the outcome that hasn't updated.",
    "<b>F6 Control / time-to-kickoff.</b> No signal: every outcome and side at 24 h and 6 h before kickoff, also split by "
    "outcome (home/draw/away, Yes/No). It measures pure market-making and any kickoff drift.",
    "<b>Structural arbitrage:</b> buy all three Yes for under $1, or all three No for under $2 (live books, minute midpoints "
    "and real trades).",
    "<b>Extra slices:</b> the Israel-morning hours (06–10) and the first 6 hours after a market opens.",
    "<b>Grid for each family:</b> signal threshold × buy style (taker, verified taker, maker at the bid for 1 h or 4 h, maker "
    "at the mid) × target (+1, +2, +3, +5c, Winner fair) × if unfilled (sell at kickoff, after 3 h, after 12 h, hold) × filters: "
    "hours to kickoff (>12 h / ≤12 h), price (15–85c / 25–75c), series volume ($10k–$250k, from training months only, so no "
    "look-ahead), non-top leagues, and a coherent book (the 3 Yes prices sum to 0.98–1.05). Longshots under 15c are always "
    "excluded.",
])

h2("6. Why it doesn't work (in plain words)")
ul([
    "<b>Polymarket is usually right first.</b> When Winner and Polymarket disagree, Polymarket is more often the one already "
    "at the new price (see the lead/lag study). Buying Polymarket ‘cheap’ against Winner usually means buying just before "
    "Winner catches up to Polymarket, not the other way round.",
    "<b>A limit buy fills when you least want it to.</b> Your resting bid fills when sellers push the price down through "
    "it, often because of news. After a ‘through’ fill the price tends to keep going down. Only 4–9% of maker buys at the "
    "bid filled at all with real trades, and those were the worse ones.",
    "<b>A sell limit above your price is not free money.</b> With +2c targets the sell filled in about 30–56% of trades "
    "before kickoff. The rest are sold at kickoff at a loss, or held to the result. The losses are bigger than the small "
    "wins, so the average is negative.",
    "<b>Midpoints lie in thin books.</b> In 3% (F1) to 74% (F4) of signals the three Yes ‘prices’ add up to more than 105%. The book is "
    "nearly empty and the midpoint is meaningless. Real buyers in those moments paid 4–21c more than ‘midpoint + half-spread’, "
    "and usually nobody traded at all (table below). Every big paper profit we found came from these moments.",
])
tab(er_rows, ER_COLS)
tab(fr_rows, FR_COLS)

h2("7. Fees, spreads and arbitrage facts used")
ul([
    "<b>Fees (Polymarket docs, read 2026-11-20, and each market's feeSchedule):</b> sports taker fee = 0.05 × price × (1 − price) "
    "per share, which is about 1.25c at 50c. <b>Makers pay no fee.</b> Makers get a 15% rebate of the taker fees their fills "
    "generate, about 0.2c per share, too small to change any result. From 2026-11-20 23:59 UTC, maker rebates become tiered "
    "($200k+ monthly volume to qualify) and taker fees are unchanged. Liquidity rewards for resting quotes exist per market "
    "(e.g. min 50 shares, within 4.5c of mid), but past payouts aren't public, so they are not included.",
    "<b>Spreads:</b> we charge half-spreads of 2c (games under $10k final volume), 1c ($10k–50k), 0.75c ($50k–250k) and 0.5c "
    "(above). Live books (2 snapshots, 1,512 event books) give median spreads of 1c in active books and 3c+ in thin ones. "
    "Real trade prints show a median effective half-spread of 0.5c, but people trade when spreads are tight, so we kept the "
    "more cautious model. Doubling the spread on taker legs makes everything worse (sensitivity table).",
    f"<b>Arbitrage, live books:</b> {live['event_books']:,} event snapshots. The sum of the 3 Yes asks was below $1 in "
    f"{live['sum_ask_lt_1']} books, but after taker fees in 0 (cheapest cost {live['best_yes_cost']:.3f}). Three No for under "
    f"$2 after fees: 0 (cheapest {live['best_no_cost']:.3f}). Median sum of asks: {live['median_sum_ask']:.2f}.",
    f"<b>Arbitrage, history:</b> real taker buys of all three outcomes within 10 minutes at a total under $1 after fees "
    f"happened in {pr['events_with_cheap_triple']} of {pr['events_with_all3']:,} events (2%). The smallest leg was typically "
    f"about ${ex.min_usd.median():.0f}, so the profit was a few cents per event. Not a business.",
])

h2("8. Robustness and extra checks")
p("Sensitivity of the closed-before-kickoff finalists and the mirage to the fill assumption, required size and spread:")
tab(sens_rows, SENS_COLS)
p("Extra slices (train and validation only, never test): Israel-morning hours and newly opened markets. All are negative.")
tab(sl_rows, SL_COLS)
p("Monthly results of the closed-before-kickoff finalists and the mirage:")
tab(monthly_rows, MON_COLS)

h2("9. Caveats")
ul([
    "Polymarket price history is the minute midpoint, not the order book. Fills come from real taker trades (a trade must "
    "print strictly through our limit). Our own order would add liquidity and could attract fills that didn't happen in "
    "history, and queue position is unknown. The ‘touch’ rule (optimistic) is shown, and it does not rescue anything.",
    "Trade history was downloaded for 16,140 of 16,763 games (all games with final volume ≤ $1M, plus 56 random bigger ones). "
    "The API pages newest-first through all in-play trades, so very big games were too slow to fetch. The trade-based results "
    "therefore cover mostly small and mid-size markets, which are the ones where the edge was supposed to be.",
    "Taker entries and exits are priced at midpoint ± modelled half-spread. In empty books (sum of mids > 1.05) that price is "
    "fantasy (section 6). That is why finalists with a plain taker entry had to have a coherent book. This rule came from a "
    "train+validation diagnostic, before the test was run.",
    "Winner's margin-free probability stands in for the live ML models. Their historical probabilities were not rebuilt.",
    f"Selection bias: {N_ALL:,} rule combinations were scored on train/validation. That is why we trust only validated rules "
    f"and the one-time test. The rank correlation of train vs validation returns across configurations was 0.53 (real trades). "
    f"Only 37% of the configurations that looked significant on train were even positive on validation.",
    "The final test covers only ~7 weeks (October to 18 November 2026), with 24–826 trades per finalist. A small positive "
    "test result would not be proof. All closed-before-kickoff finalists were negative anyway.",
    "Small size assumed ($100 per trade). Requiring $100 of real volume through our price lowers fills and returns further. "
    "Larger size would be worse.",
    "Times: Winner snapshot times were converted to UTC in the lead/lag pipeline. Prices after kickoff and duplicate snapshots "
    "were removed there. Every decision uses data up to the 15-minute check time, and orders start one minute later.",
])

h2("10. How to reproduce")
ul([
    "Python: <code>winner_model/.venv/bin/python</code>; matplotlib from <code>research/lead_lag/.pylib</code> (added automatically by charts.py). "
    "Uses the lead/lag data in <code>research/lead_lag/data</code> (series, games_meta, gamma cache). Nothing in production is touched.",
    "<code>s00_book_snapshot.py</code>: live order-book snapshot (read-only); run twice.",
    "<code>s01_trades.py</code>: real trades from data-api v2 /trades (≤7.5 req/s, ~1.5 h); <code>s01_trades.py assemble</code> rebuilds data/trades.parquet.",
    "<code>s02_signals.py</code>: signals + midpoint execution simulation → data/signals.parquet, data/sims.parquet.",
    "<code>s03_grid.py mid</code>: midpoint grid. <code>s04_trade_fills.py</code>: spread calibration + real-trade fills → data/sims_trades.parquet. "
    "<code>s04b_entry_reality.py</code>: taker price reality check. <code>s03_grid.py trades</code>: real-trade grid.",
    "<code>s05_final.py</code>: finalists, validation and the one-time test. <code>s05b_slices.py</code>: extra slices (train/val only). "
    "<code>s06_arb.py</code>: arbitrage. <code>s07_report.py</code>: this report.",
])

# ---------------------------------------------------------------- render
CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff;--accent:#2a78d6;--bad:#b42318;--soft:#f4f3ef}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff;--soft:#f4f3ef}}
:root[data-theme="dark"]{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff;--soft:#f4f3ef}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color-scheme:light}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:8px 0 4px}h2{font-size:21px;margin:36px 0 10px;padding-top:12px;border-top:1px solid var(--line)}
.sub{color:var(--ink2);margin:0 0 18px}
.verdict{background:#fdf3f2;border:1px solid #f3c7c3;border-radius:10px;padding:14px 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.tile .v{font-size:24px;font-weight:700}.tile .l{color:var(--ink2);font-size:13px}
.scroll{overflow-x:auto;margin:10px 0 16px;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;font-size:13px;min-width:100%}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--soft);font-weight:600;position:sticky;top:0}tbody tr:last-child td{border-bottom:0}
figure{margin:16px 0}figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{color:var(--ink2);font-size:14px;margin-top:6px}code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:13px}
ul{padding-left:20px}li{margin:6px 0}
"""
tiles = [("No", "strategy passed validation with real fills"),
         (f"{pct(bat.ret)}", f"best closed-before-kickoff finalist on final test ({best_A_val})"),
         (f"{N_ALL:,}", "rule combinations tried (train/validation)"),
         (f"{N_TEST}", "rule sets evaluated once on the final test"),
         ("5.2M", "real Polymarket trades used to check fills"),
         ("0", "fee-adjusted arbitrages in live books")]
body = []
for kind, c in S:
    if kind == "h2":
        body.append(f"<h2>{c}</h2>")
    elif kind == "h3":
        body.append(f"<h3>{c}</h3>")
    elif kind == "p":
        body.append(f'<p class="verdict">{c}</p>' if c.startswith("<b>No. We found") else f"<p>{c}</p>")
    elif kind == "ul":
        body.append("<ul>" + "".join(f"<li>{x}</li>" for x in c) + "</ul>")
    elif kind == "table":
        body.append(html_table(*c))
    elif kind == "img":
        body.append(f'<figure><img alt="{html.escape(c[1])}" src="data:image/png;base64,{img[c[0]]}"><figcaption>{c[1]}</figcaption></figure>')
tile_html = '<div class="tiles">' + "".join(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in tiles) + "</div>"
H = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polymarket Scalp Backtest</title><style>{CSS}</style></head><body><main>
<h1>Can we buy cheap and sell a little higher before kickoff?</h1>
<p class="sub">Backtest of buy → limit-sell strategies on Polymarket soccer 1X2 markets, using Winner odds as a signal. 16,763 games, Sep 2025 – Nov 2026. Written 2026-11-20. Research only; no orders were placed.</p>
{tile_html}
{''.join(body)}
</main></body></html>"""
open(os.path.join(HERE, "REPORT.html"), "w").write(H)

# markdown
import re
def strip(t):
    t = re.sub(r"</?b>", "**", t); t = re.sub(r"</?i>", "*", t); t = re.sub(r"</?code>", "`", t)
    return re.sub(r"<[^>]+>", "", t)
md = ["# Can we buy cheap and sell a little higher before kickoff?", "",
      "*Backtest of buy → limit-sell strategies on Polymarket soccer 1X2 markets, using Winner odds as a signal. 16,763 games, Sep 2025 – Nov 2026. Written 2026-11-20. Research only; no orders were placed.*", ""]
for kind, c in S:
    if kind == "h2":
        md += [f"## {c}", ""]
    elif kind == "p":
        md += [strip(c), ""]
    elif kind == "ul":
        md += [f"- {strip(x)}" for x in c] + [""]
    elif kind == "table":
        md += [md_table(*c), ""]
    elif kind == "img":
        md += [f"![{c[1]}](out/fig_{c[0]}.png)", "", f"*{c[1]}*", ""]
open(os.path.join(HERE, "REPORT.md"), "w").write("\n".join(md))
print("written", os.path.getsize(os.path.join(HERE, "REPORT.html")))
