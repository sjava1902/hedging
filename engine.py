import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from portfolio import Portfolio
from gcurve import GCurve
import optimizer

DAYS_PER_MONTH = 365.25 / 12
QUARTER_LEN_DAYS = 91
SWAP_FLOAT_TERM = 3 

class HedgeEngine:
    t0 = datetime(2016, 12, 31)

    def __init__(self, portfolio: Portfolio, gcurve: GCurve | None = None):
        self.portfolio = portfolio
        if gcurve is None:
           base = {0: 0.09, 3: 0.095, 6: 0.10, 12: 0.105, 24: 0.11}
           self.gcurve = GCurve(portfolio.T0, base)
        else:
            self.gcurve = gcurve
        self.t0 = portfolio.T0
        self.t_curr = self.t0
        self.bank_account = 0.0          
        self.days_since_quarter_start = 0
        self.days_since_month_start = 0
        self.swap_account = 0.0
        self.accrued_swap = 0.0
        self.accumulating_account = 0

        self.swaps = pd.DataFrame(columns=[
            "id", "direction", "notional",
            "term_months", "remaining_months",
            "fixed_rate", "float_rate_q",
            "start_date", "maturity_date"
        ])
        self._swap_id = 1
    
    def step(self, days: int = 1):
        for _ in range(days):
            self._accrue_swaps_one_day() 

            portfolio_df = self.portfolio.get_portfolio()

            if "next_payout_date" not in portfolio_df.columns:
                portfolio_df["next_payout_date"] = portfolio_df["start_date"].apply(
                    lambda d: d + relativedelta(months=+1)
                )

            due_mask = portfolio_df["next_payout_date"] <= self.t_curr
            if due_mask.any():
                due = portfolio_df.loc[due_mask]

                cash_loans = (due.loc[due["type"] == "loan",    "volume"]
                            * due.loc[due["type"] == "loan",    "rate"] / 12.0).sum()
                cash_deps  = (due.loc[due["type"] == "deposit", "volume"]
                            * due.loc[due["type"] == "deposit", "rate"] / 12.0).sum()

                self.bank_account += float(cash_loans - cash_deps)

                portfolio_df.loc[due_mask, "next_payout_date"] = portfolio_df.loc[
                    due_mask, "next_payout_date"
                ].apply(lambda d: d + relativedelta(months=+1))

            portfolio_df.loc[:, "remaining_months"] = portfolio_df["remaining_months"] - (1.0 / DAYS_PER_MONTH)

            matured_idx = portfolio_df.index[portfolio_df["remaining_months"] <= 0].tolist()
            if matured_idx:
                for idx in matured_idx:
                    row = portfolio_df.loc[idx]
                    term = int(row["contract_months"])
                    rate = row["rate"]
                    volume = row["volume"]
                    monthly_rate = rate / 12
                    type  = row["type"]
                    if (type == "loan"):
                        self.bank_account += volume * monthly_rate
                    elif (type == "deposit"):
                        self.bank_account -= volume * monthly_rate

                    new_rate = float(self.gcurve.rate(term))
                    portfolio_df.at[idx, "start_date"] = self.t_curr
                    portfolio_df.at[idx, "maturity_date"] = self.t_curr + relativedelta(months=term)
                    portfolio_df.at[idx, "remaining_months"] = float(term)
                    portfolio_df.at[idx, "rate"] = new_rate
                    portfolio_df.at[idx, "type"] = type

            self.portfolio.set_portfolio(portfolio_df)         
            self._age_swaps_and_rollover()    
            self._quarterly_settle()

            # овернайт на остаток текущего дня
            rate_over_night = self.gcurve.rate_overnight()
            self.bank_account *= (1.0 + rate_over_night / 365.0)
            # сдвиг на день
            self.gcurve.step()
            self.t_curr += timedelta(days=1)
            
            

    def snapshot_state(self):
        return {
            "date": self.t_curr,
            "bank_account": self.bank_account,
            "swap_account": self.swap_account,
            "accrued_swap": self.accrued_swap,
            "gcurve": self.gcurve.snapshot(),
            "portfolio_total_loans": float(self.portfolio.get_portfolio().query("type=='loan'")["volume"].sum()),
            "portfolio_total_deps": float(self.portfolio.get_portfolio().query("type=='deposit'")["volume"].sum()),
            "swaps_count": 0 if self.swaps.empty else int(len(self.swaps)),
        }
    
    def step_to_quarter_end(self):
        days_left = QUARTER_LEN_DAYS - self.days_since_quarter_start
        if days_left > 0:
            self.step(days_left)

    def get_t_start(self):
        return self.t0
    def set_t_start(self, t0):
        self.t0 = t0

    def get_t_curr(self):
        return self.t_curr
    def set_t_curr(self, t_curr):
        self.t_curr = t_curr

    def _quarterly_settle(self):
        """Раз в квартал — переводим накопленные проценты на счёт."""
        self.days_since_quarter_start += 1
        if self.days_since_quarter_start >= QUARTER_LEN_DAYS:
            if hasattr(self, "optimizer") and callable(getattr(self.optimizer, "rebalance_once", None)):
                decision = self.optimizer.rebalance_once(self)   # вернёт x_6,x_12,x_24
                if decision.x_6 != 0:
                    self.add_swap("receive_fixed" if decision.x_6>0 else "pay_fixed", 6,  abs(decision.x_6))
                if decision.x_12 != 0:
                    self.add_swap("receive_fixed" if decision.x_12>0 else "pay_fixed", 12, abs(decision.x_12))
                if decision.x_24 != 0:
                    self.add_swap("receive_fixed" if decision.x_24>0 else "pay_fixed", 24, abs(decision.x_24))
            #self.bank_account += self.accrued_interest
            #self.accrued_interest = 0.0

            self.swap_account += self.accrued_swap
            self.accrued_swap = 0.0

            if not self.swaps.empty:
                new_flt = float(self.gcurve.rate(SWAP_FLOAT_TERM))
                self.swaps.loc[:, "float_rate_q"] = new_flt

            self.days_since_quarter_start = 0
    
    def add_swap(self, direction: str, term_months: int, notional: float):
        """
        direction: 'pay_fixed' или 'receive_fixed'
        term_months: 6/12/24
        notional: номинал свопа
        """
        if direction not in ("pay_fixed", "receive_fixed"):
            raise ValueError("direction must be 'pay_fixed' or 'receive_fixed'")

        fixed = float(self.gcurve.rate(term_months))
        flt   = float(self.gcurve.rate(SWAP_FLOAT_TERM))

        start = self.t_curr
        mat   = self.t_curr + relativedelta(months=term_months)

        row = {
            "id": self._swap_id,
            "direction": direction,
            "notional": float(notional),
            "term_months": int(term_months),
            "remaining_months": float(term_months),
            "fixed_rate": fixed,
            "float_rate_q": flt,         
            "start_date": start,
            "maturity_date": mat,
        }
        self.swaps.loc[len(self.swaps)] = row
        self._swap_id += 1

    def _accrue_swaps_one_day(self):
        if self.swaps.empty:
            rate_over_night = self.gcurve.rate_overnight()
            self.swap_account *= (1.0 + rate_over_night / 365.0)
            return

        daily_net = 0.0
        for _, s in self.swaps.iterrows():
            fixed_rate = s["notional"] * s["fixed_rate"] / 365.0
            float_rate = s["notional"] * s["float_rate_q"] / 365.0

            if s["direction"] == "pay_fixed":
                # платим фикс, получаем плавающую
                daily_net += (float_rate - fixed_rate)
            else:  # receive_fixed
                daily_net += (fixed_rate - float_rate)

        self.accrued_swap += float(daily_net)

        # овернайт на своп-счёт
        rate_over_night = self.gcurve.rate_overnight()
        self.swap_account *= (1.0 + rate_over_night / 365.0)
    
    def _age_swaps_and_rollover(self):
        if self.swaps.empty:
            return

        # уменьшаем срок
        self.swaps.loc[:, "remaining_months"] = self.swaps["remaining_months"] - (1.0 / DAYS_PER_MONTH)

        matured = self.swaps.index[self.swaps["remaining_months"] <= 0].tolist()
        if not matured:
            return

        for idx in matured:
            s = self.swaps.loc[idx]
            term = int(s["term_months"])

            self.swaps.at[idx, "start_date"] = self.t_curr
            self.swaps.at[idx, "maturity_date"] = self.t_curr + relativedelta(months=term)
            self.swaps.at[idx, "remaining_months"] = float(term)
            self.swaps.at[idx, "fixed_rate"] = float(self.gcurve.rate(term))
            self.swaps.at[idx, "float_rate_q"] = float(self.gcurve.rate(SWAP_FLOAT_TERM))
