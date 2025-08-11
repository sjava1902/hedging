from copy import deepcopy
from statistics import mean
import numpy as np
from gcurve import GCurve
from gcurve_ns import NSCurve
from engine import HedgeEngine
from portfolio import Portfolio

# ------------------ входные параметры --------------- ---
N_C  = 100          # кредитов
N_D  = 120          # депозитов
V    = 1_000_000   # суммарный объём кредитов / депозитов
SEED = 42
DAYS_PER_MONTH = 365.25 / 12

portfolio = Portfolio(N_C=N_C, N_D=N_D, V=V)

def bench_engine(portfolio, days=365):
    base = {0:0.09,3:0.095,6:0.10,12:0.105,24:0.11}

    gc = GCurve(portfolio.T0, base, seed=42)
    en_g = HedgeEngine(deepcopy(portfolio), gcurve=gc); en_g.enable_rebalance = False

    ns = NSCurve(portfolio.T0, base_points=base, seed=42)
    en_n = HedgeEngine(deepcopy(portfolio), gcurve=ns); en_n.enable_rebalance = False

    pnl_g, pnl_n = [], []
    for _ in range(days):
        en_g.step(1); en_n.step(1)
        pnl_g.append(en_g.bank_account + en_g.swap_account)
        pnl_n.append(en_n.bank_account + en_n.swap_account)

    def cvar_quarterly(x, alpha=0.95, period=91):
        x = np.asarray(x)
        # берём значения на концах кварталов
        vals = x[::period]
        if vals.size < 2:
            return 0.0
        d = np.diff(vals)              # квартальный PnL
        losses = -d[d < 0]            # только потери
        if losses.size == 0:
            return 0.0
        losses.sort()
        k = int(np.ceil(alpha * losses.size)) - 1
        k = max(0, min(losses.size - 1, k))
        return float(losses[k:].mean())

    def mean_quarterly(x, period=91):
        x = np.asarray(x)
        vals = x[::period]
        if vals.size < 2:
            return 0.0
        return float(np.mean(np.diff(vals)))

    return {
        "GCurve_mean_q": mean_quarterly(pnl_g),
        "GCurve_CVaR95_q": cvar_quarterly(pnl_g),
        "NS_mean_q": mean_quarterly(pnl_n),
        "NS_CVaR95_q": cvar_quarterly(pnl_n),
    }

print(bench_engine(portfolio))
