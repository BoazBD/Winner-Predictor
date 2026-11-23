import pandas as pd, numpy as np
from lib import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
m=pd.read_parquet('data/markets_close.parquet')
d=m[(~m.final)&(m.mins_close<60)&(m.mkt=='1x2')].copy()
R=d[['r1','r2','r3']].values.astype(float); pi=1/R; ov=pi.sum(1)
T=np.array([1/0.9,1/0.85,1/0.8]); d['tier']=np.abs(ov[:,None]-T[None]).argmin(1)
def bisect(f,lo,hi,it=60):
    lo=np.full(len(pi),lo,float); hi=np.full(len(pi),hi,float)
    for _ in range(it):
        mid=(lo+hi)/2; v=f(mid); lo=np.where(v>1,mid,lo); hi=np.where(v>1,hi,mid)   # f decreasing in param
    return (lo+hi)/2
P={}
P['mult']=pi/ov[:,None]
k=bisect(lambda k:(pi**k[:,None]).sum(1),1,5); P['power']=pi**k[:,None]
c=bisect(lambda c:(pi/(c[:,None]+pi-c[:,None]*pi)).sum(1),1,10); P['oddsratio']=pi/(c[:,None]+pi-c[:,None]*pi)
# additive: p = pi - (ov-1)/3
P['add']=np.clip(pi-(ov[:,None]-1)/3,1e-4,1)
# shin
def shin_sum(z):
    z=z[:,None]; return ((np.sqrt(z**2+4*(1-z)*pi**2/ov[:,None])-z)/(2*(1-z))).sum(1)
z=bisect(shin_sum,0.0,0.6); zz=z[:,None]; P['shin']=(np.sqrt(zz**2+4*(1-zz)*pi**2/ov[:,None])-zz)/(2*(1-zz))
Y=d[['won1','wonx','won2']].values.astype(float)
fav=R.argmin(1); ar=np.arange(len(d))
print('multiclass log-loss by tier')
for nm,p in P.items():
    ll=-(Y*np.log(np.clip(p,1e-6,1))).sum(1)
    print(nm.ljust(10), {t:round(ll[d.tier.values==t].mean(),5) for t in [0,1,2]}, 'sum',p.sum(1).mean().round(4))
print('favourite calibration: mean(pred) vs mean(win), implied fav EV, by tier x fav-odds bucket')
ob=pd.cut(R[ar,fav],[1,1.1,1.2,1.3,1.45,1.6,1.8,2.1,3])
for nm,p in P.items():
    x=pd.DataFrame(dict(tier=d.tier.values,ob=ob,pred=p[ar,fav],win=Y[ar,fav],o=R[ar,fav]))
    x['ev_pred']=x.pred*x.o-1; x['real']=x.win*x.o-1
    t=x.groupby(['tier','ob'],observed=True).agg(n=('win','size'),ev_pred=('ev_pred','mean'),real=('real','mean'),se=('real','sem'))
    t['z']=(t.real-t.ev_pred)/t.se
    print(nm); print((t.assign(ev_pred=t.ev_pred*100,real=t.real*100,se=t.se*100)).round(2).unstack(0).to_string())
# longshot side calibration too
dog=R.argmax(1)
for nm,p in P.items():
    x=pd.DataFrame(dict(tier=d.tier.values,pred=p[ar,dog],win=Y[ar,dog],dr=p[ar,1],dw=Y[ar,1]))
    print(nm,'dog pred/win, draw pred/win by tier'); print(x.groupby('tier').mean().round(4).to_string())
