import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
#from variant import load_variant
from datetime import timedelta
from engine import HedgeEngine
from portfolio import Portfolio
from gcurve import GCurve
from gcurve_ns import NSCurve
import optimizer

# ------------------ входные параметры --------------- ---
N_C  = 100          # кредитов
N_D  = 120          # депозитов
V    = 1_000_000   # суммарный объём кредитов / депозитов
SEED = 42
DAYS_PER_MONTH = 365.25 / 12

def validate_portfolio(portfolio : Portfolio):
    credits_df = portfolio.get_credits()
    deposits_df = portfolio.get_deposits()
    portfolio_df = portfolio.get_portfolio()

    assert abs(credits_df["volume"].sum()  - V) < 1e-6
    assert abs(deposits_df["volume"].sum() - V) < 1e-6
    assert (credits_df["remaining_months"] <= credits_df["contract_months"]).all()
    assert (deposits_df["remaining_months"] <= deposits_df["contract_months"]).all()
    assert portfolio_df["rate"].min() > 0, "Ставки должны быть положительными"


if __name__ == "__main__":
    portfolio = Portfolio(N_C=N_C, N_D=N_D, V=V)
    validate_portfolio(portfolio=portfolio)
    portfolio_df = portfolio.get_portfolio()
    print(portfolio_df.head())

    base = {0: 0.09, 3: 0.095, 6: 0.10, 12: 0.105, 24: 0.11}
    gc = GCurve(portfolio.T0, base)
    print("Rates before:", gc.snapshot())
    gc.step(10)
    print("Rates after 10 dayss:", gc.snapshot())

    df = portfolio.get_portfolio()
    expected_daily = (df.loc[df.type=='loan','volume']*df.loc[df.type=='loan','rate']/365).sum() \
                     - (df.loc[df.type=='deposit','volume']*df.loc[df.type=='deposit','rate']/365).sum()
    print("expected_daily_net:", round(expected_daily, 6))

    engine = HedgeEngine(portfolio=portfolio, gcurve=gc)
    engine.optimizer = optimizer 
    engine.step(1)
    print("bank account 1d:", engine.bank_account)



# # engine.step(10)
# # print("t_curr:", engine.get_t_curr() if hasattr(engine,'get_t_curr') else engine.t_curr)
# # print("bank_account:", round(engine.bank_account, 2))
# # print("accrued_interest:", round(engine.accrued_interest, 2))

# engine.step(90)  # ещё 90 после первого дня
# print("accrued_interest (should be 0):", round(engine.accrued_interest, 6))
# print("bank_account (should be > 0):", round(engine.bank_account, 2))


# t_begin = engine.t_curr
# engine.step(40)
# after = portfolio.get_portfolio()
# mask = (after['start_date'] >= t_begin) & (after['start_date'] < engine.t_curr)
# rolled = mask.sum()
# print("rolled count (за 40 дней):", rolled)

# print(after.loc[mask, ['id','type','rate','remaining_months']].head())

# engine.add_swap(direction="pay_fixed",     term_months=12, notional=V * 0.50)
# engine.add_swap(direction="receive_fixed", term_months=6,  notional=V * 0.30)

# # прогоняем квартал
# engine.step_to_quarter_end()
# print("after rebalance: swaps_count =", len(engine.swaps))
# if not engine.swaps.empty:
#     print(engine.swaps[["id","direction","notional","term_months","fixed_rate","float_rate_q","remaining_months"]].head())
# print("swap_account:", round(engine.swap_account, 2), "accrued_swap:", round(engine.accrued_swap, 2))


# base = {0: 0.09, 3: 0.095, 6: 0.10, 12: 0.105, 24: 0.11}

# gc = GCurve(portfolio.T0, base)
# gns = NSCurve(portfolio.T0, base_points=base)
# print("GCurve:", GCurve(portfolio.T0, base).snapshot())
# print("NSCurve:", gns.snapshot())

# engine = HedgeEngine(portfolio=portfolio, gcurve=gns)
# engine.step(91)
# print("NS t_curr:", engine.t_curr)
# print("NS bank_account:", round(engine.bank_account, 2))