# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:00:54 2026

@author: 7thap
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt


#Data (2016–2024)

TICKERS = [
    "NVDA","GOOGL","GOOG","AAPL","MSFT","AMZN","AVGO","META","TSLA","WMT",
    "JPM","LLY","XOM","V","MU","JNJ","AMD","ORCL","MA","COST","INTC","NFLX",
    "CAT","BAC","CVX","PG","ABBV","KO","PLTR","CSCO","UNH","HD","LRCX","AMAT",
    "MS","GE","HON","GS","MRK","PM","WFC","LIN","IBM","MCD","PEP","VZ","T",
    "DIS","ISRG","ABT"]

MARKET = "^GSPC"
start, end = "2016-01-01", "2024-12-31"

raw = yf.download(TICKERS + [MARKET], start=start, end=end)["Close"].dropna()
available = [t for t in TICKERS if t in raw.columns]

prices_assets = raw[available]
prices_market = raw[MARKET]

ret_assets = np.log(prices_assets / prices_assets.shift(1)).dropna()
ret_market = np.log(prices_market / prices_market.shift(1)).dropna()
ret_assets, ret_market = ret_assets.align(ret_market, join="inner", axis=0)

R_assets = -ret_assets.values
R_market = -ret_market.values
dates = ret_assets.index
n = R_assets.shape[1]


#Kernel CoES

def kernel_coes_fast(loss_p, loss_m, qp, qm, h):
    # Market VaR
    lo, hi = np.percentile(loss_m, [1, 99])
    var_m = brentq(lambda x: np.mean(norm.cdf((x-loss_m)/h)) - qm, lo, hi)

    w_m = 1 - norm.cdf((var_m - loss_m) / h)
    denom = w_m.mean()
    if denom < 1e-6:
        return np.quantile(loss_p, qp)

    target = (1 - qp) * denom

    def joint(x):
        return np.mean((1 - norm.cdf((x-loss_p)/h)) * w_m)

    lo, hi = np.percentile(loss_p, [1, 99])
    f_lo, f_hi = joint(lo) - target, joint(hi) - target

    if f_lo * f_hi > 0:
        return np.quantile(loss_p[w_m > np.median(w_m)], qp)

    covar = brentq(lambda x: joint(x) - target, lo, hi)

    w_p = 1 - norm.cdf((covar - loss_p) / h)
    w = w_p * w_m
    if w.sum() < 1e-8:
        return loss_p.mean()

    return np.sum(w * loss_p) / np.sum(w)


#Portfolio estimators (fast)

def ew(n): return np.ones(n)/n

def gmv(R):
    cov = np.cov(R.T)
    def obj(w): return w @ cov @ w
    cons = {"type":"eq","fun":lambda w: w.sum()-1}
    bnds = [(0,1)]*R.shape[1]
    return minimize(obj, np.ones(n)/n, bounds=bnds, constraints=cons).x

def mv(R):
    mu = -R.mean(axis=0)
    cov = np.cov(R.T)
    target = mu.mean()
    def obj(w): return w @ cov @ w
    cons = [
        {"type":"eq","fun":lambda w: w.sum()-1},
        {"type":"eq","fun":lambda w: w@mu - target}
    ]
    bnds = [(0,1)]*R.shape[1]
    return minimize(obj, np.ones(n)/n, bounds=bnds, constraints=cons).x

def es_np(R, alpha):
    def obj(w):
        lp = R @ w
        q = np.quantile(lp, alpha)
        tail = lp[lp > q]
        return tail.mean() if len(tail)>0 else lp.mean()
    cons = {"type":"eq","fun":lambda w: w.sum()-1}
    bnds = [(0,1)]*R.shape[1]
    return minimize(obj, np.ones(n)/n, bounds=bnds, constraints=cons).x


#Rolling 2-year window, weekly rebalancing

weekly_dates = ret_assets.resample("W-FRI").last().index
weekly_dates = weekly_dates[weekly_dates >= pd.Timestamp("2018-01-01")]

configs = [(0.7,0.9),(0.5,0.9),(0.7,0.8)]
results = {"date":[], "M_week":[]}

for qm,qp in configs:
    results[f"CoESNP_{qm}_{qp}"]=[]
    results[f"CoESP_{qm}_{qp}"]=[]

results["ES_09"]=[]; results["ES_08"]=[]
results["MV"]=[]; results["GMV"]=[]; results["EW"]=[]

for d in weekly_dates:
    start_win = d - pd.DateOffset(years=2)
    mask = (dates>=start_win)&(dates<d)
    if mask.sum()<100: continue

    Rw, Rmw = R_assets[mask], R_market[mask]
    h = 0.03 * len(Rw)**(-0.3)

    w_ew = ew(n)
    w_gmv = gmv(Rw)
    w_mv = mv(Rw)
    w_es09 = es_np(Rw,0.9)
    w_es08 = es_np(Rw,0.8)

    w_coes = {}
    for qm,qp in configs:
        w_coes[(qm,qp)] = minimize(
            lambda w: kernel_coes_fast(Rw@w, Rmw, qp, qm, h),
            np.ones(n)/n,
            bounds=[(0,1)]*n,
            constraints={"type":"eq","fun":lambda w: w.sum()-1},
            options={"maxiter":150}
        ).x

    eval_mask = (dates>=d)&(dates<d+pd.Timedelta(days=7))
    if eval_mask.sum()==0: continue

    Re, Rme = R_assets[eval_mask], R_market[eval_mask]
    P = lambda w: (Re@w).sum()

    results["date"].append(d)
    results["M_week"].append(Rme.sum())

    for qm,qp in configs:
        results[f"CoESNP_{qm}_{qp}"].append(P(w_coes[(qm,qp)]))
        results[f"CoESP_{qm}_{qp}"].append(P(w_coes[(qm,qp)]))

    results["ES_09"].append(P(w_es09))
    results["ES_08"].append(P(w_es08))
    results["MV"].append(P(w_mv))
    results["GMV"].append(P(w_gmv))
    results["EW"].append(P(w_ew))

res_df = pd.DataFrame(results).set_index("date")


# Tables 4 & 5 as per the main paper

def stats(P):
    m = P.mean()*52
    s = P.std()*np.sqrt(52)
    ra = m/s if s>0 else np.nan
    return m,s,ra,skew(P),kurtosis(P,fisher=False)

def build_table(th):
    sub = res_df[res_df["M_week"]>th]
    rows=[]
    for qm,qp in configs:
        m,s,ra,sk,kt = stats(sub[f"CoESNP_{qm}_{qp}"])
        rows.append(["CoES","nonparametric",qm,qp,m,s,ra,sk,kt])
    for qm,qp in configs:
        m,s,ra,sk,kt = stats(sub[f"CoESP_{qm}_{qp}"])
        rows.append(["CoES","parametric",qm,qp,m,s,ra,sk,kt])
    for qp,col in [(0.9,"ES_09"),(0.8,"ES_08")]:
        m,s,ra,sk,kt = stats(sub[col])
        rows.append(["ES","nonparametric",None,qp,m,s,ra,sk,kt])
    for crit,col in [("MV","MV"),("GMV","GMV"),("EW","EW")]:
        m,s,ra,sk,kt = stats(sub[col])
        rows.append([crit,"-",None,None,m,s,ra,sk,kt])
    return pd.DataFrame(rows,columns=[
        "Criterion","Method","q_m","q_p","Mean","SD","RiskAdjMean","Skew","Kurtosis"
    ])

table4 = build_table(0.0)
table5 = build_table(0.015)

print("\nTABLE 4 (market loss > 0%)\n",table4)
print("\nTABLE 5 (market loss > 1.5%)\n",table5)


# 6. Cumulative log-return plots

returns_df = -res_df.copy()

def cumlog(x): return np.cumsum(np.log1p(x))

cum = pd.DataFrame(index=returns_df.index)
cum["EW"] = cumlog(returns_df["EW"])
cum["MV"] = cumlog(returns_df["MV"])

for qm,qp in configs:
    cum[f"CoESNP_{qm}_{qp}"] = cumlog(returns_df[f"CoESNP_{qm}_{qp}"])
    cum[f"CoESP_{qm}_{qp}"] = cumlog(returns_df[f"CoESP_{qm}_{qp}"])

def plot_cum(qm,qp):
    plt.figure(figsize=(12,6))
    plt.title(f"Cumulative Log-Return (2016–2024)\nq_m={qm}, q_p={qp}")
    plt.plot(cum.index,cum[f"CoESNP_{qm}_{qp}"],label="CoES–NP",linewidth=2)
    plt.plot(cum.index,cum[f"CoESP_{qm}_{qp}"],label="CoES–P",linestyle="--",linewidth=2)
    plt.plot(cum.index,cum["MV"],label="MV",linestyle=":",linewidth=2)
    plt.plot(cum.index,cum["EW"],label="EW",linestyle="-.",linewidth=2)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plot_cum(0.7,0.9)
plot_cum(0.5,0.9)
plot_cum(0.7,0.8)

table4