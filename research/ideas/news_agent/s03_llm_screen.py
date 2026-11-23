"""s03 - LLM screening of GDELT headlines with Haiku 4.5 on Bedrock (bulk, cheap).

For each sample game: the GKG headlines from s02c (seen in [kick-48h, kick-5min), de-duplicated), ask Haiku (prompts/screen_haiku.txt) to tag each one: material?, team,
type, home_effect (-2..2), post_match. Each headline is judged only from its own text, so there is no
cross-time look-ahead: a decision at time t only uses headlines with seendate <= t.
PLACEBO (mode 'swap'): the same headlines but with the HOME/AWAY labels in the prompt swapped; a model that
reads the headline (not its memory of the season) must flip the sign of home_effect.
Cached per game in data/screen[_swap]/<uid>.json. Token usage appended to logs/llm_usage.csv.
Usage: python s03_llm_screen.py [normal|swap] [max_games]   (4 games in parallel)
"""
import os, sys, json, glob, re, time, datetime as dt
import pandas as pd
from anthropic import AnthropicBedrock
from concurrent.futures import ThreadPoolExecutor
import threading
LOCK = threading.Lock()

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
PRICE = (1.0, 5.0)   # $/M input, output tokens (Haiku 4.5)
TPL = open(os.path.join(HERE, "prompts/screen_haiku.txt")).read()
COMP = {"פרמייר ליג": "English Premier League", "ספרדית ראשונה": "La Liga", "איטלקית ראשונה": "Serie A",
        "גרמנית ראשונה": "Bundesliga", "צרפתית ראשונה": "Ligue 1", "ליגת האלופות": "UEFA Champions League",
        "הליגה האירופית": "UEFA Europa League"}
cli = AnthropicBedrock(aws_region="il-central-1")


H = None


def headlines(g):
    """Headlines assigned to this game by s02c (GKG titles, seen in [kick-48h, kick-5min))."""
    global H
    if H is None:
        H = pd.read_parquet(os.path.join(HERE, "data/headlines.parquet"))
    df = H[H.unique_id == g.unique_id].sort_values("t_seen").reset_index(drop=True)
    return df.rename(columns={"t_seen": "ts", "src": "domain"})


def log_usage(stage, model, u):
    p = os.path.join(HERE, "logs/llm_usage.csv"); new = not os.path.exists(p)
    cost = u.input_tokens / 1e6 * PRICE[0] + u.output_tokens / 1e6 * PRICE[1]
    with LOCK, open(p, "a") as f:
        if new: f.write("time,stage,model,in_tok,out_tok,usd\n")
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S},{stage},{model},{u.input_tokens},{u.output_tokens},{cost:.6f}\n")
    return cost


def call(prompt, stage, max_tokens=4000):
    for i in range(5):
        try:
            r = cli.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
            log_usage(stage, MODEL, r.usage)
            txt = r.content[0].text
            m = re.search(r"\[.*\]", txt, re.S)
            return json.loads(m.group(0))
        except Exception as e:
            print("  retry", type(e).__name__, str(e)[:120], flush=True); time.sleep(5 * (i + 1))
    return None


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    od = os.path.join(HERE, "data/screen" + ("_swap" if mode == "swap" else "")); os.makedirs(od, exist_ok=True)
    s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet")).sort_values("poly_volume", ascending=False)
    headlines(s.iloc[0])
    todo = [g for _, g in s.iterrows() if not os.path.exists(os.path.join(od, f"{g.unique_id}.json"))][:mx]

    def work(g):
        fn = os.path.join(od, f"{g.unique_id}.json")
        df = headlines(g)
        if df.empty:
            json.dump([], open(fn, "w")); return
        home, away = (g.home_q, g.away_q) if mode == "normal" else (g.away_q, g.home_q)
        out = []
        for st in range(0, len(df), 60):
            ch = df.iloc[st:st + 60]
            items = "\n".join(f"{i}. [{r.mins_before} min before kickoff] {r.title}" for i, r in ch.iterrows())
            res = call(TPL.format(home=home, away=away, competition=COMP.get(g.league, g.league), items=items), "screen_" + mode)
            for o in res or []:
                if isinstance(o, dict) and o.get("i") in ch.index:
                    r = ch.loc[o["i"]]
                    out.append({**o, "ts": str(r.ts), "mins_before": int(r.mins_before), "title": r.title, "domain": r.domain})
        json.dump(out, open(fn, "w"))
        print(g.unique_id, g.home_q, "-", g.away_q, len(df), "material:", sum(bool(o.get("material")) for o in out), flush=True)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, todo))
