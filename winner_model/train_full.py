"""Fit the deployment model on ALL data (run after the final test; adds the test period to training)."""
import pandas as pd, numpy as np, pickle
from model import *
c=pd.read_parquet('data/cand2.parquet').reset_index(drop=True); c=prep(c)
mc=pd.read_parquet('data/markets_close.parquet'); close_ok=set(mc.index[mc.mins_close<60])
cl=closing_rows(c,close_ok); mods=fit_level(cl)
pc=cl[['unique_id','side']].assign(P_c=pred_level(mods,cl).values)
d=c.merge(pc,on=['unique_id','side'],how='left'); d['P_now']=pred_level(mods,d).values
d['drift']=d.P_c*d.odds-d.P_now*d.odds
bs=fit_drift(d[d.drift.notna()])
pickle.dump(dict(level=mods,drift=[b.model_to_string() for b in bs],trained_through=str(c.kickoff.max())),open('data/model_full.pkl','wb'))
print('saved data/model_full.pkl', len(d))
