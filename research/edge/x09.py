import numpy as np, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 300)
exec(open("x07.py").read().split("S = sys.argv")[0])
rows = []
for band in ((.02, .05), (.05, .08), (.08, .10), (.10, .15), (.02, .08), (.02, .10), (.03, .10)):
    for T in ((1,), (3,), (6,), (9, 12), (24,), (48, 72)):
        for fill in ("strict", "touch"):
            for s in ("train", "val", "test"):
                x = d[(d.mid >= band[0]) & (d.mid <= band[1]) & d.T0.isin(T) & (d.split == s)]
                e = ev(x, 0, fill); rows.append(dict(band=f"{band[0]:.2f}-{band[1]:.2f}", T=str(T), fill=fill, split=s, **e))
r = pd.DataFrame(rows)
p = r.pivot_table(index=["band", "T", "fill"], columns="split", values=["roi", "fills"])[[("roi", "train"), ("roi", "val"), ("roi", "test"), ("fills", "train"), ("fills", "val"), ("fills", "test")]]
print(p.round(3).to_string())
r.to_csv("out/grid_results.csv", index=False)
