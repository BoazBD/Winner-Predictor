"""s04 - Sonnet final judgement on material news, event study on the 15-min Polymarket grid, and trading P&L.

News event = (game, GDELT 15-min slot) with >= 1 headline that Haiku tagged material, not post-match,
|home_effect| >= 1. For each event Sonnet 4.6 (prompts/judge_sonnet.txt) sees ONLY the new headlines, the
earlier material headlines of that game, and the market midpoints at the slot time; it returns point moves
d_home/d_draw/d_away. Signal = outcome with the largest |d| if |d| >= 1.5 points (buy YES if d>0, NO if d<0).
Also a Haiku-only signal (sign of the summed home_effect -> home YES or NO) for comparison.

Event study: signed midpoint change of the signalled side from t_seen-180 .. t_seen+180 min and to kickoff.
Trading: decide at t_seen + delay (delay 0 = 'act on the GDELT slot', 60 = hourly cron worst case is +60; we
use next full hour), buy at grid midpoint + 0.5c half-spread + taker fee, sell at the last grid midpoint
before kickoff - 0.5c - fee. Return per $ staked; 95% CI by bootstrapping games.
Placebos: (a) random-direction: the same entry times with a random outcome/side (1000 permutations);
(b) label swap: Haiku re-run with HOME/AWAY swapped (s03 swap) must flip the sign of home_effect.
Outputs: out/events.csv, out/event_study.csv, out/trading.csv, out/placebo.json, out/fig_event_study.png
"""
import os, re, json, glob, time, datetime as dt
import numpy as np, pandas as pd, duckdb
from anthropic import AnthropicBedrock

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
SON = "global.anthropic.claude-sonnet-4-6"
TPL = open(os.path.join(HERE, "prompts/judge_sonnet.txt")).read()
HS = 0.005
fee = lambda p: 0.05 * p * (1 - p)
rng = np.random.default_rng(7)
from s03_llm_screen import COMP

s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet")).set_index("unique_id")
# --- screened headlines
rows = []
for f in glob.glob(os.path.join(HERE, "data/screen/*.json")):
    u = os.path.basename(f)[:-5]
    for o in json.load(open(f)):
        rows.append(dict(unique_id=u, **{k: o.get(k) for k in ["i", "material", "team", "type", "home_effect", "post_match", "ts", "mins_before", "title", "domain"]}))
sc = pd.DataFrame(rows)
sc["ts"] = pd.to_datetime(sc.ts)
sc["home_effect"] = pd.to_numeric(sc.home_effect, errors="coerce").fillna(0)
mat = sc[(sc.material == True) & (sc.post_match != True) & (sc.home_effect.abs() >= 1)].copy()
HL = pd.read_parquet(os.path.join(HERE, "data/headlines.parquet"))[["unique_id", "title", "t_pub"]].drop_duplicates(["unique_id", "title"])
mat = mat.merge(HL, on=["unique_id", "title"], how="left")
print("screened", len(sc), "games", sc.unique_id.nunique(), "material", len(mat), "games w/ material", mat.unique_id.nunique())

# --- grid
ids = ",".join(f"'{u}'" for u in s.index)
G = duckdb.sql(f"select unique_id, t_min, mins_to_kick, p0, p1, p2 from '{R}/lead_lag/data/grid.parquet' where unique_id in ({ids})").df()
for c in ["p0", "p1", "p2"]:
    G[c] = G[c].astype("float64").round(4)
G = G.sort_values(["unique_id", "t_min"])
GG = {u: g.reset_index(drop=True) for u, g in G.groupby("unique_id")}


def price_at(u, tmin):
    """latest grid row at or before unix minute tmin -> (p0,p1,p2) or None"""
    g = GG.get(u)
    if g is None:
        return None
    i = np.searchsorted(g.t_min.values, tmin, side="right") - 1
    if i < 0 or tmin - g.t_min.values[i] > 20:
        return None
    r = g.iloc[i]
    return np.array([r.p0, r.p1, r.p2])


def close_px(u):
    g = GG[u]; g = g[(g.mins_to_kick >= 3) & g.p0.notna()]
    return np.array(g.iloc[-1][["p0", "p1", "p2"]], dtype=float), int(g.iloc[-1].t_min)


tm = lambda ts: int(pd.Timestamp(ts).timestamp() // 60)

# --- Sonnet judge per event (cached)
cache_f = os.path.join(HERE, "data/judge_cache.json")
cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
cli = AnthropicBedrock(aws_region="il-central-1")


def judge(u, slot, new, earlier, p):
    key = f"{u}|{slot}"
    if key in cache:
        return cache[key]
    g = s.loc[u]
    items = "\n".join(f"- [{r.mins_before} min before kickoff] {r.title}" for r in new.itertuples())
    ear = "\n".join(f"- [{r.mins_before} min before kickoff] {r.title}" for r in earlier.tail(8).itertuples()) or "(none)"
    prompt = TPL.format(home=g.home_q, away=g.away_q, competition=COMP.get(g.league, g.league), p0=p[0], p1=p[1], p2=p[2], items=items, earlier=ear)
    for i in range(4):
        try:
            r = cli.messages.create(model=SON, max_tokens=300, messages=[{"role": "user", "content": prompt}])
            cost = r.usage.input_tokens * 3e-6 + r.usage.output_tokens * 15e-6
            with open(os.path.join(HERE, "logs/llm_usage.csv"), "a") as f:
                f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S},judge,{SON},{r.usage.input_tokens},{r.usage.output_tokens},{cost:.6f}\n")
            j = json.loads(re.search(r"\{.*\}", r.content[0].text, re.S).group(0))
            cache[key] = j
            return j
        except Exception as e:
            print("retry", e); time.sleep(5)
    return None


ev = []
for (u, slot), new in mat.groupby(["unique_id", "ts"]):
    p = price_at(u, tm(slot))
    if p is None or np.isnan(p).any():
        continue
    earlier = mat[(mat.unique_id == u) & (mat.ts < slot)]
    j = judge(u, str(slot), new, earlier, p)
    if not j:
        continue
    d = np.array([float(j.get("d_home", 0) or 0), float(j.get("d_draw", 0) or 0), float(j.get("d_away", 0) or 0)])
    ev.append(dict(unique_id=u, t_seen=slot, mins_before=int(new.mins_before.max()), n_items=len(new),
                   types=",".join(sorted(set(new.type.astype(str)))), titles=" || ".join(new.title)[:400],
                   haiku_home=float(new.home_effect.sum()), t_pub=new.t_pub.min(), d_home=d[0], d_draw=d[1], d_away=d[2],
                   conf=j.get("confidence"), reason=j.get("reason"), p0=p[0], p1=p[1], p2=p[2]))
json.dump(cache, open(cache_f, "w"))
E = pd.DataFrame(ev)
E["k_sig"] = E[["d_home", "d_draw", "d_away"]].abs().values.argmax(1)
E["d_sig"] = E[["d_home", "d_draw", "d_away"]].values[np.arange(len(E)), E.k_sig]
E["sonnet_sig"] = E.d_sig.abs() >= 1.5
E.to_csv(os.path.join(HERE, "out/events.csv"), index=False)
print("events", len(E), "sonnet signals", int(E.sonnet_sig.sum()))

# --- event study (signed midpoint path of the signalled side)
OFF = [-180, -120, -60, -30, -15, 0, 15, 30, 45, 60, 90, 120, 180]


def path(u, t0, k, sgn):
    """signed path (cents) of side k around unix-minute t0"""
    base = price_at(u, t0)
    if base is None or np.isnan(base[k]):
        return None
    out = {}
    cp, ct = close_px(u)
    for o in OFF:
        if t0 + o > ct:
            out[o] = np.nan; continue
        q = price_at(u, t0 + o)
        out[o] = np.nan if q is None else sgn * (q[k] - base[k]) * 100
    out["kick"] = sgn * (cp[k] - base[k]) * 100
    return out


def study(sub, kcol, scol, name, anchor="t_seen"):
    P = []
    for r in sub.itertuples():
        a = getattr(r, anchor)
        if pd.isna(a):
            continue
        x = path(r.unique_id, tm(a), getattr(r, kcol), getattr(r, scol))
        if x:
            P.append(dict(set=name, unique_id=r.unique_id, **{str(k): v for k, v in x.items()}))
    return pd.DataFrame(P)


S1 = E[E.sonnet_sig].copy(); S1["sgn"] = np.sign(S1.d_sig)
H1 = E[E.haiku_home != 0].copy(); H1["k0"] = 0; H1["sgn"] = np.sign(H1.haiku_home)
big = S1[S1.d_sig.abs() >= 3]
es = pd.concat([study(S1, "k_sig", "sgn", "sonnet |d|>=1.5"), study(big, "k_sig", "sgn", "sonnet |d|>=3"),
                study(H1, "k0", "sgn", "haiku home_effect"),
                study(S1, "k_sig", "sgn", "sonnet |d|>=1.5, anchored at publisher time", anchor="t_pub")])
es.to_csv(os.path.join(HERE, "out/event_study_rows.csv"), index=False)
summ = es.drop(columns="unique_id").groupby("set").mean().T
sem = (es.drop(columns="unique_id").groupby("set").std() / np.sqrt(es.groupby("set").size().values[:, None])).T
sem.to_csv(os.path.join(HERE, "out/event_study_se.csv"))
summ.loc["n_events"] = es.groupby("set").size().reindex(summ.columns).values
summ.to_csv(os.path.join(HERE, "out/event_study.csv"))
print(summ.round(2).to_string())

# --- trading
def trade(u, t_dec, k, sgn):
    cp, ct = close_px(u)
    if t_dec > ct - 10:
        return None
    p = price_at(u, t_dec)
    if p is None or np.isnan(p[k]):
        return None
    pe, px = (p[k], cp[k]) if sgn > 0 else (1 - p[k], 1 - cp[k])
    if pe <= 0.02 or pe >= 0.98:
        return None
    c = pe + HS + fee(pe + HS); pr = px - HS - fee(px - HS)
    return (pr - c) / c


def run(sub, kcol, delay):
    out = []
    for r in sub.itertuples():
        t0 = tm(r.t_seen)
        td = t0 + delay if delay >= 0 else (t0 // 60 + 1) * 60   # -1 = next full hour (hourly cron)
        x = trade(r.unique_id, td, getattr(r, kcol), r.sgn)
        if x is not None:
            out.append((r.unique_id, x))
    return pd.DataFrame(out, columns=["u", "ret"])


def boot(df):
    gm = df.groupby("u").ret.agg(["sum", "count"])
    b = [(lambda x: x["sum"].sum() / x["count"].sum())(gm.sample(len(gm), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
    return np.percentile(b, [2.5, 97.5])


TR = []
for name, sub, kcol in [("sonnet |d|>=1.5", S1, "k_sig"), ("sonnet |d|>=3", big, "k_sig"), ("haiku home_effect", H1, "k0")]:
    for delay, lab in [(0, "at GDELT slot (~10-15 min after publish)"), (-1, "hourly cron (next full hour)"), (60, "+60 min"), (180, "+180 min")]:
        df = run(sub, kcol, delay)
        if len(df) < 3:
            continue
        lo, hi = boot(df)
        TR.append(dict(signal=name, decision=lab, trades=len(df), games=df.u.nunique(), ret_per_dollar=df.ret.mean(),
                       ci_lo=lo, ci_hi=hi, win_share=(df.ret > 0).mean()))
TR = pd.DataFrame(TR); TR.to_csv(os.path.join(HERE, "out/trading.csv"), index=False)
print(TR.round(4).to_string())

# --- placebo (a): random outcome/side at the same entry times as the Sonnet signal, slot-time decision
base = run(S1, "k_sig", 0).ret.mean()
perm = []
for i in range(300):
    X = S1.copy(); X["k_r"] = rng.integers(0, 3, len(X)); X["sgn"] = rng.choice([-1, 1], len(X))
    perm.append(run(X, "k_r", 0).ret.mean())
plc = dict(sonnet_ret=base, random_mean=float(np.mean(perm)), random_p95=float(np.percentile(perm, 95)),
           p_value=float((np.array(perm) >= base).mean()))
# --- placebo (b): label swap agreement
sw = []
for f in glob.glob(os.path.join(HERE, "data/screen_swap/*.json")):
    u = os.path.basename(f)[:-5]
    a = {o["title"]: o for o in json.load(open(f))}
    b = sc[sc.unique_id == u].set_index("title")
    for t, o in a.items():
        if t in b.index and (o.get("material") or b.loc[t].material is True):
            he_n = float(b.loc[t].home_effect) if np.isscalar(b.loc[t].home_effect) else 0.0
            sw.append((he_n, float(o.get("home_effect") or 0)))
if sw:
    sw = np.array(sw); nz = (sw[:, 0] != 0) & (sw[:, 1] != 0)
    plc.update(swap_pairs=int(len(sw)), swap_both_nonzero=int(nz.sum()),
               swap_sign_flipped_share=float((np.sign(sw[nz, 0]) == -np.sign(sw[nz, 1])).mean()) if nz.any() else None)
json.dump(plc, open(os.path.join(HERE, "out/placebo.json"), "w"), indent=1)
print(plc)

# --- chart
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 3.8))
cols = {"sonnet |d|>=1.5": "#2a6fdb", "sonnet |d|>=3": "#0b3a8c", "haiku home_effect": "#d9822b", "sonnet |d|>=1.5, anchored at publisher time": "#3a9e5c"}
for st in summ.columns:
    y = summ.loc[[str(o) for o in OFF], st].values
    ax.plot(OFF, y, "o-", ms=3, color=cols.get(st), label=f"{st} (n={int((es.set == st).sum())})")
ax.axvline(0, color="grey", lw=.6, ls="--"); ax.axhline(0, color="grey", lw=.6)
ax.set_xlabel("minutes from news time (GDELT sighting; green = publisher timestamp)"); ax.set_ylabel("price move in predicted direction, cents")
ax.set_title("Polymarket midpoint around material team news"); ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
plt.savefig(os.path.join(HERE, "out/fig_event_study.png"), dpi=130)
