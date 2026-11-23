"""Step 5b: extra slices suggested by the lead/lag study (train + validation ONLY, never test):
Israel-morning checks (06-10 h Israel time) and the market's first 6 hours ('opening price mispricing'),
for F1/F2 (+F6 control) with realistic entries (maker bid 4 h or verified taker), +2c target / Winner fair, sell at kickoff.
Output out/slices.csv"""
import os
import numpy as np, pandas as pd
from common import load_all, HERE

S, M = load_all()
S = S[(S.split != "test") & S.x0.between(0.15, 0.85)]
T = pd.read_parquet(os.path.join(HERE, "data", "sims_trades.parquet"))
T = T[(T["mode"] == "through") & T.filled.astype(bool) & T.ent.isin(["bid240", "taker_v"])]
X = T.merge(S[["sid", "uid", "fam", "thr", "split", "hour_il", "age", "psum"]], on="sid")
X = X[X.fam.isin(["F1", "F2", "F6"]) & ((X.fam != "F6") | (X.thr == 24.0)) & ((X.fam == "F1") & (X.thr == 0.03) | (X.fam == "F2") & (X.thr == 0.05) | (X.fam == "F6"))]
X["ret"] = X.pnl_kick.astype(float) / X.pe
X["slice"] = np.where(X.hour_il.between(6, 10), "Israel 06-10h", "other hours")
X2 = X.copy(); X2["slice"] = np.where(X2.age < 360, "market < 6 h old", "market >= 6 h old")
Y = pd.concat([X, X2])
out = Y[np.isclose(Y.d, 0.02) | np.isclose(Y.d, -1.0)].assign(d=lambda z: z.d.round(2)).groupby(["fam", "ent", "d", "slice", "split"]).agg(n=("ret", "size"), ret=("ret", "mean"), hit=("hit", "mean")).reset_index()
out.to_csv(os.path.join(HERE, "out", "slices.csv"), index=False)
pd.set_option("display.width", 200)
print(out.pivot_table(index=["fam", "ent", "d", "slice"], columns="split", values=["n", "ret"]).round(3).to_string())
print("configs", out.groupby(["fam", "ent", "d", "slice"]).ngroups)
