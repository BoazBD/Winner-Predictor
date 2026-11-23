import json, base64
R = json.load(open("out/results.json"))
img = lambda f: '<img src="data:image/png;base64,%s">' % base64.b64encode(open(f"out/{f}", "rb").read()).decode()
pct = lambda x, d=1: f"{100*x:+.{d}f}%"
H = {v["name"][0] + ("o" if "optimistic" in v["name"] else ""): v for v in R["house"]}
def hrow(v):
    c = lambda s: f'<td class="n">{pct(v[s]["roi"])}<span>{v[s]["fills"]} bets, {v[s]["lost"]} lost</span></td>'
    return f'<tr><td>{v["name"]}</td>{c("train")}{c("val")}{c("test")}<td class="n">{100*v["all"]["fill_rate"]:.0f}%</td></tr>'
B = H["B"]; ms = R["house_month_sim"]; A = R["arb"]; P = R["post"]; L = R["late"]; X = R["house_exit"]
mon = "".join(f'<tr><td>{m["m"]}</td><td class="n">{m["bets"]}</td><td class="n">{int(m["lost"])}</td><td class="n {"pos" if m["usd"]>0 else "neg"}">{m["usd"]:+.0f}</td></tr>' for m in R["house_monthly"] if m["m"] >= "2025-09")
css = """:root{--bg:#f9f9f7;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--good:#006300;--bad:#c42f2f;--accent:#2a78d6;--soft:#e6f0fc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:28px 16px 60px}h1{font-size:1.9rem;line-height:1.2;margin:0 0 .3em}h2{margin:2.2em 0 .6em;font-size:1.3rem;border-bottom:1px solid var(--line);padding-bottom:.3em}
h3{margin:1.4em 0 .4em;font-size:1.05rem}.sub{color:var(--ink2)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
.lead{border-color:var(--accent);background:var(--soft)}.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem;min-width:520px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}td span{display:block;color:var(--ink2);font-size:.78rem}
.pos{color:var(--good);font-weight:600}.neg{color:var(--bad);font-weight:600}img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-size:1.35rem}.kpi span{color:var(--ink2);font-size:.8rem}@media(max-width:640px){.kpis{grid-template-columns:repeat(2,1fr)}h1{font-size:1.5rem}}
.tag{display:inline-block;border-radius:999px;padding:1px 9px;font-size:.75rem;font-weight:600}.g{background:#e3f3e3;color:var(--good)}.y{background:#fdf3dc;color:#7a5300}.r{background:#fbe9e9;color:var(--bad)}"""
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Polymarket edge study</title><style>{css}</style></head><body><main>
<h1>Where is the money on Polymarket soccer?</h1>
<p class="sub">Tested on 13 million real Polymarket trades from 16,136 soccer games (Sep 2024 – Nov 2026), including trades during and after the games. Written 20 Nov 2026. No real money was used.</p>

<div class="card lead"><b>Short answer.</b> Predicting how the price will move, then selling before kickoff, does not work — prices before the game don't trend, and the fee eats any small move. But the trades show one clear, repeated pattern: <b>people who buy longshots lose heavily, and whoever sells to them wins</b>. The best strategy found is to be that seller: place limit orders against extreme longshots (3–8¢) before the game. It won {100*B["all"]["win_rate"]:.0f}% of its bets and made about <b>{pct(B["all"]["roi"])} per bet</b> in every test period — but it does depend on game results. Two truly result-free trades also exist (arbitrage with Winner, and buying winning shares after the final whistle), but both are tiny.</div>

<div class="kpis">
<div class="kpi"><b>{pct(B["all"]["roi"])}</b><span>profit per $ at risk, per bet (strategy B, all periods)</span></div>
<div class="kpi"><b>{100*B["all"]["win_rate"]:.1f}%</b><span>of bets won ({B["all"]["fills"]} bets, {B["all"]["lost"]} lost)</span></div>
<div class="kpi"><b>{pct(B["test"]["roi"])}</b><span>in the final test (Oct–Nov 2026)</span></div>
<div class="kpi"><b>{100*ms["p_loss"]:.0f}%</b><span>estimated chance of a losing month at ~70 bets</span></div></div>

<h2>1. The pattern: longshot buyers lose</h2>
<p>For every share, I took the first real purchase in the last 24 hours before kickoff and checked how it ended. Fee included.</p>
{img("fig_buyers.png")}
<p>Buyers of 3–8¢ shares lost about half their money in each of the three periods (−45%, −59%, −68%). Buyers of 8–15¢ shares lost 12–22%. In the middle (30–85¢), buyers lose roughly the fee, so the price is fair. Longshots are overpriced because people like cheap lottery tickets.</p>

<h2>2. Best strategy: be the seller of longshots</h2>
<div class="card"><b>Exact rules (strategy B)</b>
<ol><li><b>When:</b> 6 hours before kickoff, look at every soccer game on Polymarket.</li>
<li><b>What:</b> any outcome priced between 3¢ and 8¢ (a big underdog, or a draw in a very one-sided game).</li>
<li><b>Order:</b> place a <b>limit order to buy the "No" share</b> of that outcome at (100¢ − its current price). Example: outcome at 6¢ → bid 94¢ for "No". This is a resting limit order, so there is no fee.</li>
<li><b>If nobody fills it by kickoff:</b> cancel. No trade, no cost.</li>
<li><b>If filled:</b> hold to the end of the game. "No" pays 100¢ unless the longshot actually wins.</li>
<li><b>Size:</b> the same small amount on every bet (e.g. 100 shares ≈ $94). Never more on one game — a loss costs about 15 wins.</li></ol></div>
<p>A fill is counted only when a real trade went <i>through</i> our price (strict). Rules were chosen on games before July 2026, then checked on July–September, then run once on October–November.</p>
<div class="tw"><table><tr><th>Variant</th><th class="n">Before Jul 2026</th><th class="n">Jul–Sep check</th><th class="n">Oct–Nov final test</th><th class="n">Orders filled</th></tr>
{hrow(H["B"])}{hrow(H["A"])}{hrow(H["Ao"])}{hrow(H["C"])}{hrow(H["D"])}{hrow(H["E"])}</table></div>
<p class="sub">Numbers are profit per $ at risk. Variant A's strict result (1 loss in 242) is partly luck; its optimistic-fill row and variant B are the realistic guide. The edge is only in the extreme longshots: at 8–15¢ it is about +1% and unreliable, and above 15¢ there is none.</p>
{img("fig_house.png")}
<h3>Month by month (strategy B, 100 shares per bet)</h3>
<div class="tw"><table><tr><th>Month</th><th class="n">Bets</th><th class="n">Lost</th><th class="n">Profit $</th></tr>{mon}</table></div>
<p>13 of the last 15 months were profitable. A typical recent month has 40–65 bets and earns about $100–290 at 100 shares per bet, using roughly $1,000–2,000 of capital at a time (money is tied up for less than a day per bet). Half of the fills had 900+ shares of real buying behind them, so bets of a few hundred dollars are realistic; much bigger is not.</p>
<h3>Can we close before kickoff and avoid the game?</h3>
<p><b>No.</b> Buying the position back 5 minutes before kickoff turns the profit into a loss: {pct(X["train"]["roi"])}, {pct(X["val"]["roi"])} and {pct(X["test"]["roi"])} in the three periods. The profit comes from the longshot not winning, not from a price move.</p>
<h3>Risks</h3>
<ul><li><b>It depends on results.</b> About 1 bet in 35 loses, and a loss wipes out about 15 wins. Several losses close together are possible: the worst months (Oct 2025, Apr 2026) had 3 losses and ended slightly negative.</li>
<li><b>Our orders weren't in the real book.</b> I assume we get filled when a real trade crosses our price. If others are ahead of us in the queue we fill less often — fewer bets, same quality.</li>
<li><b>Fill rates are falling</b> (76% → 56% → 51%), which suggests more sellers are competing. The edge can shrink.</li>
<li><b>Sample size.</b> 587 bets with 16 losses. The 95% range for the true return is {pct(B["all"]["ci"][0])} to {pct(B["all"]["ci"][1])} per bet.</li>
<li><b>The Oct–Nov period</b> was already looked at in earlier studies (for other strategies). A fully clean test is a live paper-trade from today.</li></ul>

<h2>3. Truly result-free trades (real, but small)</h2>
<h3>a) Arbitrage: Winner vs Polymarket <span class="tag y">risk-free, rare</span></h3>
<p>Winner is slow. Sometimes Polymarket moves and, for a few minutes, betting an outcome at Winner <i>and</i> buying "No" on the same outcome at Polymarket locks a profit whatever the result. Checked against real Polymarket trades and Winner odds that were live before and after.</p>
{img("fig_arb.png")}
<ul><li>Found in {A["train"]["games"]+A["val"]["games"]+A["test"]["games"]} of about 16,700 games (0.4%). Recently about {A["per_month_recent"]:.0f} games a month.</li>
<li>Typical locked profit: {100*A["test"]["median_edge"]:.1f}% of the money put in.</li>
<li>Usually 45–75 minutes before kickoff (line-up news), and the window lasts only a few minutes.</li>
<li>With up to $1,000 per game: about <b>${A["profit_per_month_recent"]:.0f} a month</b>.</li>
<li>Needs a single bet at Winner placed within minutes. Check Winner's rules on single bets, limits and tax first.</li></ul>
<h3>b) Buy the winner after the final whistle <span class="tag y">near risk-free, crowded</span></h3>
<p>After a game ends, holders who don't want to wait for the payout sell winning shares at 99.9¢. A resting bid at 99.9¢ earns 0.1¢ per share, with no fee.</p>
<ul><li>From 150 minutes after kickoff: ${(P["train"]["usd"]+P["val"]["usd"]+P["test"]["usd"])/1e6:.0f}M was sold this way, and only ${P["test"]["lost_usd"]:.0f} of it went to a share that lost.</li>
<li>Profit: 0.1% per round. Recently about ${P["usd_per_day_recent"]/1000:.0f}k a day is sold this way in these games, so the whole pie is about ${P["usd_per_day_recent"]*0.001:.0f} a day, shared between all bidders (mostly bots, first in line gets filled).</li>
<li>Safe, but realistically a few dollars a day.</li></ul>
<h3>c) Late-game lottery tickets <span class="tag y">small, needs speed</span></h3>
<p>In the last minutes, sellers of 1–2¢ shares earned {pct(L["train"]["roi"])}, {pct(L["val"]["roi"])} and {pct(L["test"]["roi"])} on their capital in the three periods. Positive every time, but a late goal turns a 1¢ share into a full loss, and faster bots pull their orders first.</p>

<h2>4. What does not work</h2>
<div class="tw"><table><tr><th>Idea</th><th>Result</th></tr>
<tr><td>Predict the price move, sell before kickoff</td><td><span class="tag r">no</span> After a move of 3¢+ in the last 1–3 hours, the next move averages 0.0–0.5¢. A round trip costs about 3¢ in fees and spread.</td></tr>
<tr><td>Buy heavy favourites (85–97¢)</td><td><span class="tag r">no</span> As a fixed rule: −0.3%, +0.3%, +2.0% in the three periods — not reliable.</td></tr>
<tr><td>Buy near-certain shares during the game</td><td><span class="tag r">no</span> In-play prices are accurate; buyers lose about the fee.</td></tr>
<tr><td>Sell 8–30¢ outcomes</td><td><span class="tag r">no</span> About 0%.</td></tr>
<tr><td>Follow Winner moves; buy low and limit-sell</td><td><span class="tag r">no</span> Earlier studies: −2% to −8%.</td></tr></table></div>
<p>The common thread: on Polymarket, people who <b>take</b> prices lose and people who <b>post</b> limit orders win. Every profitable result here is on the posting side.</p>

<h2>5. Recommendation</h2>
<ol><li><b>Paper-trade strategy B live</b> for 4–6 weeks in our existing system (it already reads Polymarket every 3 hours). This gives a clean test and real fill data.</li>
<li>If it holds, start with small real size (about $50–100 per bet) and fixed stakes.</li>
<li>Treat the arbitrage as an optional alert ("Winner is stale vs Polymarket now"), not as income.</li></ol>

<h2>How this was tested</h2>
<ul><li>13.0M real trades (pre-match, in-play, post-game) for 16,136 games with clean results; minute-by-minute prices; Winner odds every 15 minutes.</li>
<li>Three periods: before 1 Jul 2026 (choose rules), Jul–Sep 2026 (check), Oct–18 Nov 2026 (final test).</li>
<li>Ranges are 95% bootstrap intervals, resampling whole games.</li>
<li>Files: <code>research/edge/</code> — <code>s01_master.py</code> (trades + results), <code>s02_house.py</code> (longshot orders), <code>s03_arb.py</code> (arbitrage), <code>s04_final.py</code> (numbers and charts), <code>x02–x06</code> (checks of the ideas that failed), <code>out/results.json</code>.</li></ul>
</main></body></html>"""
open("REPORT.html", "w").write(html); print(len(html))
