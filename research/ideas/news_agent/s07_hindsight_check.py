"""s07 - Does the LLM know the results of the sample games (hindsight leakage)?

Ask Haiku 4.5 and Sonnet 4.6 for the result of 30 random sample games (Aug-Nov 2026), and also their
training cutoff. If they cannot beat the closing-favourite baseline, memory of results is not driving the
headline judgements. Output: out/hindsight_check.csv, .json
"""
import os, json, re
import numpy as np, pandas as pd, duckdb
from anthropic import AnthropicBedrock

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))
cli = AnthropicBedrock(aws_region="il-central-1")
MODELS = {"haiku": ("global.anthropic.claude-haiku-4-5-20251001-v1:0", 1, 5), "sonnet": ("global.anthropic.claude-sonnet-4-6", 3, 15)}
s = pd.read_parquet(os.path.join(HERE, "data/sample.parquet")).sample(30, random_state=1)
rows, cost = [], 0
for name, (mid, pi, po) in MODELS.items():
    r = cli.messages.create(model=mid, max_tokens=60, messages=[{"role": "user", "content": "In one line: what is your training data cutoff month/year?"}])
    cut = r.content[0].text.strip()
    for _, g in s.iterrows():
        q = (f"Football match played on {g.kick_utc:%d %B %Y}: {g.home_q} (home) vs {g.away_q} (away). "
             "What was the result? If you do not know, give your best guess. Answer ONLY JSON: "
             '{"known": true|false, "outcome": "home"|"draw"|"away"}')
        r = cli.messages.create(model=mid, max_tokens=60, messages=[{"role": "user", "content": q}])
        cost += r.usage.input_tokens / 1e6 * pi + r.usage.output_tokens / 1e6 * po
        try:
            j = json.loads(re.search(r"\{.*\}", r.content[0].text, re.S).group(0))
        except Exception:
            j = {}
        rows.append(dict(model=name, cutoff=cut, uid=g.unique_id, claims_known=j.get("known"), guess=j.get("outcome"),
                         actual=["home", "draw", "away"][[float(g.outcome_price0), float(g.outcome_price1), float(g.outcome_price2)].index(1.0)]))
df = pd.DataFrame(rows); df["hit"] = df.guess == df.actual
df.to_csv(os.path.join(HERE, "out/hindsight_check.csv"), index=False)
res = {m: dict(cutoff=g.cutoff.iloc[0], claims_known=float((g.claims_known == True).mean()), hit_rate=float(g.hit.mean()),
               n=len(g), home_guess_share=float((g.guess == "home").mean())) for m, g in df.groupby("model")}
act = df[df.model == "haiku"].actual
res["baseline_always_home_hit"] = float((act == "home").mean())
res["usd"] = cost
json.dump(res, open(os.path.join(HERE, "out/hindsight_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
