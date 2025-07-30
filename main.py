import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
#from variant import load_variant
from datetime import timedelta
from engine import HedgeEngine
from portfolio import Portfolio

# ------------------ входные параметры --------------- ---
N_C  = 100          # кредитов
N_D  = 120          # депозитов
V    = 1_000_000   # суммарный объём кредитов / депозитов
SEED = 42
DAYS_PER_MONTH = 365.25 / 12

portfolio = Portfolio(N_C=N_C, N_D=N_D, V=V)
engine = HedgeEngine(portfolio=portfolio)

credits_df = portfolio.get_credits()
deposits_df = portfolio.get_deposits()
portfolio_df = portfolio.get_portfolio()

# ─── sanity-чек ───────────────────────────────────────────────────────────
assert abs(credits_df["volume"].sum()  - V) < 1e-6
assert abs(deposits_df["volume"].sum() - V) < 1e-6
assert (credits_df["remaining_months"] <= credits_df["contract_months"]).all()
assert (deposits_df["remaining_months"] <= deposits_df["contract_months"]).all()
assert portfolio_df["rate"].min() > 0, "Ставки должны быть положительными"



print(portfolio_df.head())
