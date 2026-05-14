# -*- coding: utf-8 -*-
"""
Created on Thu May 14 06:59:21 2026

@author: 7thap
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, skew, kurtosis, chi2, genpareto
from scipy.optimize import minimize, linprog, brentq
import matplotlib.pyplot as plt
import time


#  DATA prep step (2016–2024)


TICKERS = [
    "NVDA","GOOGL","GOOG","AAPL","MSFT","AMZN","AVGO","META","TSLA","WMT",
    "JPM","LLY","XOM","V","MU","JNJ","AMD","ORCL","MA","COST","INTC","NFLX",
    "CAT","BAC","CVX","PG","ABBV","KO","PLTR","CSCO","UNH","HD","LRCX","AMAT",
    "MS","GE","HON","GS","MRK","PM","WFC","LIN","IBM","MCD","PEP","VZ","T",
    "DIS","ISRG","ABT"
]
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
R_market = -ret_market.values.flatten()
dates = ret_assets.index
n = R_assets.shape[1]


#KERNEL CoES (nonparametric)


def kernel_coes_fast(loss_p, loss_m, qp, qm, h):
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


# PORTFOLIO Estimator functions (EW, GMV, MV, ES-NP)


def ew(n): return np.ones(n)/n

def gmv(R):
    cov = np.cov(R.T)
    def obj(w): return w @ cov @ w
    cons = {"type":"eq","fun":lambda w: w.sum()-1}
    bnds = [(0,1)]*R.shape[1]
    return minimize(obj, np.ones(R.shape[1])/R.shape[1], bounds=bnds, constraints=cons).x

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
    return minimize(obj, np.ones(R.shape[1])/R.shape[1], bounds=bnds, constraints=cons).x

def es_np(R, alpha):
    def obj(w):
        lp = R @ w
        q = np.quantile(lp, alpha)
        tail = lp[lp > q]
        return tail.mean() if len(tail)>0 else lp.mean()
    cons = {"type":"eq","fun":lambda w: w.sum()-1}
    bnds = [(0,1)]*R.shape[1]
    return minimize(obj, np.ones(R.shape[1])/R.shape[1], bounds=bnds, constraints=cons).x


#PARAMETRIC CoES (Gaussian)


def coes_parametric(lp, alpha):
    mu = lp.mean()
    sigma = lp.std()
    if sigma == 0: 
        return mu
    z = norm.ppf(alpha)
    return mu + sigma * norm.pdf(z) / (1 - alpha)

def coes_parametric_joint(lp, lm, qp, qm):
    return coes_parametric(lp, qp) + coes_parametric(lm, qm)


#EVT MARGINALS PER ASSET (for copula scenarios)


def fit_gpd_tail_asset(x):
    x = np.asarray(x)
    u = np.quantile(x, 0.95)
    excess = x[x > u] - u
    if len(excess) < 30:
        return u, None
    c, loc, scale = genpareto.fit(excess, floc=0.0)
    return u, (c, scale)

def inv_evt_asset(x, u, params, u_prob, u_grid, x_grid, U):
    U = np.asarray(U)
    out = np.empty_like(U)
    below = U <= u_prob
    above = ~below
    if below.any():
        U_b = U[below] / u_prob
        out[below] = np.interp(U_b, u_grid, x_grid)
    if above.any():
        if params is None:
            out[above] = u
        else:
            c, scale = params
            V = (U[above] - u_prob) / (1 - u_prob)
            out[above] = u + genpareto.ppf(V, c, loc=0.0, scale=scale)
    return out

#MULTIVARIATE GAUSSIAN COPULA + EVT SCENARIOS


def simulate_multivariate_gaussian_evt(Rw, n_sim=3000):
    cov = np.cov(Rw.T)
    Z = np.random.multivariate_normal(np.zeros(Rw.shape[1]), cov, size=n_sim)
    U = norm.cdf(Z)

    sims = np.zeros_like(U)
    for j in range(Rw.shape[1]):
        x = Rw[:, j]
        u, params = fit_gpd_tail_asset(x)
        mask = x <= u
        x_grid = np.sort(x[mask])
        u_grid = np.linspace(0, 1, len(x_grid), endpoint=False)
        u_prob = (x <= u).mean()
        sims[:, j] = inv_evt_asset(x, u, params, u_prob, u_grid, x_grid, U[:, j])
    return sims


#ROCKAFELLAR–URYASEV LP FOR CoES (CVaR) MINIMISATION


def coes_lp_solver(R_scenarios, alpha):
    #This is thevRockafellar–Uryasev LP
    N, n = R_scenarios.shape
    n_var = n + 1 + N  # w (n), eta (1), z (N)

    c = np.zeros(n_var)
    c[n] = 1.0
    c[n+1:] = 1.0 / ((1-alpha)*N)

    A_ub = []
    b_ub = []

    for s in range(N):
        row = np.zeros(n_var)
        row[:n] = R_scenarios[s, :]
        row[n] = -1.0
        row[n+1+s] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    for s in range(N):
        row = np.zeros(n_var)
        row[n+1+s] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)

    A_eq = np.zeros((1, n_var))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, 1)]*n + [(None, None)] + [(0, None)]*N

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    w_opt = res.x[:n]
    return w_opt


#ROLLING BACKTEST (2-year window, weekly) Compare: CoESNP, CoESP, CoESLP, ES, MV, GMV, EW


weekly_dates = ret_assets.resample("W-FRI").last().index
weekly_dates = weekly_dates[weekly_dates >= pd.Timestamp("2018-01-01")]

configs = [(0.7,0.9),(0.5,0.9),(0.7,0.8)]

results = {"date":[], "M_week":[]}
for qm,qp in configs:
    results[f"CoESNP_{qm}_{qp}"]=[]
    results[f"CoESP_{qm}_{qp}"]=[]
    results[f"CoESLP_{qm}_{qp}"]=[]

results["ES_09"]=[]; results["ES_08"]=[]
results["MV"]=[]; results["GMV"]=[]; results["EW"]=[]

weights_hist = {
    "CoESNP_0.7_0.9": [],
    "CoESP_0.7_0.9": [],
    "CoESLP_0.7_0.9": [],
    "MV": [],
    "GMV": [],
    "EW": []
}

start_time = time.time()

for d in weekly_dates:
    start_win = d - pd.DateOffset(years=2)
    mask = (dates>=start_win)&(dates<d)
    if mask.sum()<100:
        continue

    Rw, Rmw = R_assets[mask], R_market[mask]
    h = 0.03 * len(Rw)**(-0.3)

    w_ew = ew(n)
    w_gmv = gmv(Rw)
    w_mv = mv(Rw)
    w_es09 = es_np(Rw,0.9)
    w_es08 = es_np(Rw,0.8)

    w_coes_np = {}
    w_coes_p  = {}
    w_coes_lp = {}

    scenarios = simulate_multivariate_gaussian_evt(Rw, n_sim=3000)

    for qm, qp in configs:
        w_coes_np[(qm, qp)] = minimize(
            lambda w: kernel_coes_fast(Rw @ w, Rmw, qp, qm, h),
            np.ones(n)/n,
            bounds=[(0,1)]*n,
            constraints={"type":"eq","fun":lambda w: w.sum()-1},
            options={"maxiter":80}
        ).x

        w_coes_p[(qm, qp)] = minimize(
            lambda w: coes_parametric_joint(Rw @ w, Rmw, qp, qm),
            np.ones(n)/n,
            bounds=[(0,1)]*n,
            constraints={"type":"eq","fun":lambda w: w.sum()-1},
            options={"maxiter":80}
        ).x

        w_coes_lp[(qm, qp)] = coes_lp_solver(scenarios, alpha=qp)

    eval_mask = (dates>=d)&(dates<d+pd.Timedelta(days=7))
    if eval_mask.sum()==0:
        continue

    Re, Rme = R_assets[eval_mask], R_market[eval_mask]
    P = lambda w: (Re@w).sum()

    results["date"].append(d)
    results["M_week"].append(Rme.sum())

    for qm,qp in configs:
        results[f"CoESNP_{qm}_{qp}"].append(P(w_coes_np[(qm, qp)]))
        results[f"CoESP_{qm}_{qp}"].append(P(w_coes_p[(qm, qp)]))
        results[f"CoESLP_{qm}_{qp}"].append(P(w_coes_lp[(qm, qp)]))

    results["ES_09"].append(P(w_es09))
    results["ES_08"].append(P(w_es08))
    results["MV"].append(P(w_mv))
    results["GMV"].append(P(w_gmv))
    results["EW"].append(P(w_ew))

    weights_hist["CoESNP_0.7_0.9"].append(w_coes_np[(0.7,0.9)])
    weights_hist["CoESP_0.7_0.9"].append(w_coes_p[(0.7,0.9)])
    weights_hist["CoESLP_0.7_0.9"].append(w_coes_lp[(0.7,0.9)])
    weights_hist["MV"].append(w_mv)
    weights_hist["GMV"].append(w_gmv)
    weights_hist["EW"].append(w_ew)

end_time = time.time()
total_runtime = end_time - start_time

res_df = pd.DataFrame(results).set_index("date")


#TABLES (CoESNP vs CoESP vs CoESLP vs benchmarks)

def stats(P):
    m = P.mean()*52
    s = P.std()*np.sqrt(52)
    ra = m/s if s>0 else np.nan
    return m,s,ra,skew(P),kurtosis(P,fisher=False)

def build_table(th):
    sub = res_df[res_df["M_week"]>th]
    rows=[]
    for qm,qp in configs:
        rows.append(["CoES","KDE",qm,qp,*stats(sub[f"CoESNP_{qm}_{qp}"])])
    for qm,qp in configs:
        rows.append(["CoES","Gaussian",qm,qp,*stats(sub[f"CoESP_{qm}_{qp}"])])
    for qm,qp in configs:
        rows.append(["CoES","Copula-LP (Gauss-EVT)",qm,qp,*stats(sub[f"CoESLP_{qm}_{qp}"])])
    for qp,col in [(0.9,"ES_09"),(0.8,"ES_08")]:
        rows.append(["ES","nonparametric",None,qp,*stats(sub[col])])
    for crit,col in [("MV","MV"),("GMV","GMV"),("EW","EW")]:
        rows.append([crit,"not conditional",None,None,*stats(sub[col])])
    return pd.DataFrame(rows,columns=[
        "Criterion","Method","q_m","q_p","Mean","SD","RiskAdjMean","Skew","Kurtosis"
    ])

table4 = build_table(0.0)
table5 = build_table(0.015)

print("\nTABLE 4 (market loss > 0%)\n",table4)
print("\nTABLE 5 (market loss > 1.5%)\n",table5)


#CUMULATIVE LOG-RETURN PLOTS


returns_df = -res_df.copy()

def cumlog(x): return np.cumsum(np.log1p(x))

cum = pd.DataFrame(index=returns_df.index)
cum["EW"] = cumlog(returns_df["EW"])
cum["MV"] = cumlog(returns_df["MV"])

for qm,qp in configs:
    cum[f"CoESNP_{qm}_{qp}"] = cumlog(returns_df[f"CoESNP_{qm}_{qp}"])
    cum[f"CoESP_{qm}_{qp}"] = cumlog(returns_df[f"CoESP_{qm}_{qp}"])
    cum[f"CoESLP_{qm}_{qp}"] = cumlog(returns_df[f"CoESLP_{qm}_{qp}"])

def plot_cum(qm,qp):
    plt.figure(figsize=(12,6))
    plt.title(f"Cumulative Log-Return (2018–2024)\nq_m={qm}, q_p={qp}")
    plt.plot(cum.index,cum[f"CoESNP_{qm}_{qp}"],label="CoES–KDE",linewidth=2)
    plt.plot(cum.index,cum[f"CoESP_{qm}_{qp}"],label="CoES–Gaussian",linewidth=2,linestyle=":")
    plt.plot(cum.index,cum[f"CoESLP_{qm}_{qp}"],label="CoES–Copula-LP (Gauss-EVT)",linewidth=2,linestyle="--")
    plt.plot(cum.index,cum["MV"],label="MV",linestyle="-.",linewidth=1.5)
    plt.plot(cum.index,cum["EW"],label="EW",linestyle="-.",linewidth=1.5,alpha=0.7)
    plt.xlabel("Date"); plt.ylabel("Cumulative log-return")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

plot_cum(0.7,0.9)
plot_cum(0.5,0.9)
plot_cum(0.7,0.8)


#EVALUATION METRICS


def max_drawdown(returns):
    r = np.asarray(returns)
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (peak - wealth) / peak
    return dd.max()

def cvar(losses, alpha):
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    return tail.mean() if len(tail)>0 else q

def systemic_coes(losses, market_losses, qm):
    var_m = np.quantile(market_losses, qm)
    mask = market_losses >= var_m
    if mask.sum()==0:
        return losses.mean()
    return losses[mask].mean()

def kupiec_test(losses, alpha):
    losses = np.asarray(losses)
    var = np.quantile(losses, alpha)
    N = len(losses)
    I = (losses > var).astype(int)
    x = I.sum()
    pi_hat = x / N
    if pi_hat == 0 or pi_hat == 1:
        return var, x, np.nan, np.nan
    num = (1 - alpha)**(N - x) * (alpha**x)
    den = (1 - pi_hat)**(N - x) * (pi_hat**x)
    LR = -2 * np.log(num / den)
    p_value = 1 - chi2.cdf(LR, df=1)
    return var, x, LR, p_value

def turnover_from_weights(w_list):
    if len(w_list) < 2:
        return 0.0
    w_arr = np.vstack(w_list)
    diff = np.abs(np.diff(w_arr, axis=0))
    return diff.sum(axis=1).mean()

def dm_test(loss1, loss2):
    d = loss1 - loss2
    dbar = d.mean()
    var_d = d.var(ddof=1)
    if var_d == 0:
        return np.nan
    T = len(d)
    dm = dbar / np.sqrt(var_d / T)
    return dm

strategies = {
    "CoESNP_0.7_0.9": res_df["CoESNP_0.7_0.9"],
    "CoESP_0.7_0.9": res_df["CoESP_0.7_0.9"],
    "CoESLP_0.7_0.9": res_df["CoESLP_0.7_0.9"],
    "MV": res_df["MV"],
    "GMV": res_df["GMV"],
    "EW": res_df["EW"]
}

summary_rows = []
rf = 0.0
market_losses_week = res_df["M_week"].values

for name, ret in strategies.items():
    r = ret.values
    mu = r.mean()*52
    sigma = r.std()*np.sqrt(52)
    sharpe = (mu - rf)/sigma if sigma>0 else np.nan
    mdd = max_drawdown(r)
    losses = -r
    cvar95 = cvar(losses, 0.95)
    cvar99 = cvar(losses, 0.99)
    coes_07 = systemic_coes(losses, market_losses_week, 0.7)
    var95, x95, LR95, p95 = kupiec_test(losses, 0.95)
    var99, x99, LR99, p99 = kupiec_test(losses, 0.99)
    to = turnover_from_weights(weights_hist[name])
    if name != "EW":
        dm = dm_test(-r, -strategies["EW"].values)
    else:
        dm = np.nan
    summary_rows.append([
        name, mu, sigma, sharpe, mdd,
        cvar95, cvar99, coes_07,
        x95, p95, x99, p99,
        to, dm
    ])

summary_df = pd.DataFrame(summary_rows, columns=[
    "Strategy","Mean","SD","Sharpe","MaxDrawdown",
    "CVaR_95","CVaR_99","CoES_qm0.7",
    "VaRViol_95","Kupiec_p_95","VaRViol_99","Kupiec_p_99",
    "Turnover","DM_vs_EW"
])

summary_df["Rank_Sharpe"] = (-summary_df["Sharpe"]).rank()
summary_df["Rank_CoES"] = (summary_df["CoES_qm0.7"]).rank()

print("\nEVALUATION SUMMARY (KDE vs Gaussian vs Copula-LP vs MV/GMV/EW, qm=0.7, qp=0.9)\n")
print(summary_df.sort_values("Rank_Sharpe"))
print(f"\nTotal runtime (seconds): {total_runtime:.2f}")
