import pandas as pd, numpy as np
from model import *
from wf import FOLDS, summarize
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
c=pd.read_parquet('data/cand2.parquet'); c=c[~c.final].reset_index(drop=True); c=prep(c)
mc=pd.read_parquet('data/markets_close.parquet'); close_ok=set(mc.index[mc.mins_close<60])
out=[]
for a,b in FOLDS:
    te,_=run_split(c,close_ok,a,b); out.append(te); print(a,len(te),flush=True)
p=pd.concat(out); p.to_parquet('data/dev_final_preds.parquet')
