import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

LOAN_TERM_OPTS = np.array([6, 12, 24])
DEP_TERM_OPTS  = np.array([3,  6, 12])
LOAN_TERM_PROB = None        # равномерно
DEP_TERM_PROB  = None        # равномерно
DAYS_PER_MONTH = 365.25 / 12

class Portfolio:
    N_C = 0
    N_D = 0
    V = 0
    T0 = datetime(2016, 12, 31)
    credits = pd.DataFrame()
    deposits = pd.DataFrame()
    portfolio = pd.DataFrame()
    
    rng = np.random.default_rng(42)

    def __init__(self, N_C=0, N_D=0, V=0):
        self.N_C = N_C
        self.N_D = N_D
        self.V = V

        # 1. объёмы ---------------------------------------------------------------
        u_loans = self.rng.random(N_C)
        u_deps  = self.rng.random(N_D)
        vol_loans = V * u_loans / u_loans.sum()
        vol_deps  = V * u_deps  / u_deps.sum()

        # 2. контрактные сроки ----------------------------------------------------
        loan_terms = self.rng.choice(LOAN_TERM_OPTS, size=N_C, p=LOAN_TERM_PROB)
        dep_terms  = self.rng.choice(DEP_TERM_OPTS,  size=N_D, p=DEP_TERM_PROB)

        # 3. оставшийся срок / даты погашения -------------------------------------
        rem_loans = loan_terms * self.rng.random(N_C)
        rem_deps  = dep_terms  * self.rng.random(N_D)
        eps = 1e-6
        rem_loans[rem_loans < eps] = eps
        rem_deps [rem_deps  < eps] = eps


        mat_loans = [
            self.T0 + timedelta(days=float(m) * DAYS_PER_MONTH)
            for m in rem_loans
        ]
        mat_deps = [
            self.T0 + timedelta(days=float(m) * DAYS_PER_MONTH)
            for m in rem_deps
        ]

        start_loans = [
            mat - relativedelta(months=int(term))
            for mat, term in zip(mat_loans, loan_terms)
        ]
        start_deps = [
            mat - relativedelta(months=int(term))
            for mat, term in zip(mat_deps, dep_terms)
        ]
        # 4. ставка: строим кривые на t_min ---------------------------------------
        #t_min = min(start_loans + start_deps)

        rates_loans = [self.loan_curve(term) for term in loan_terms]
        rates_deps  = [self.dep_curve(term)  for term in dep_terms]

        # 5. собираем всё в DataFrame ---------------------------------------------
        self.credits = pd.DataFrame(
            {
                "id":               range(1, N_C + 1),
                "type":             "loan",
                "volume":           vol_loans,
                "contract_months":  loan_terms,
                "remaining_months": rem_loans,
                "start_date":       start_loans,
                "maturity_date":    mat_loans,
                "rate":             rates_loans,   # годовая ставка
            }
        )

        self.deposits = pd.DataFrame(
            {
                "id":               range(1, N_D + 1),
                "type":             "deposit",
                "volume":           vol_deps,
                "contract_months":  dep_terms,
                "remaining_months": rem_deps,
                "start_date":       start_deps,
                "maturity_date":    mat_deps,
                "rate":             rates_deps,
            }
        )

        self.portfolio = pd.concat([self.credits, self.deposits], ignore_index=True)
        self.portfolio["start_date"]    = (
            pd.to_datetime(self.portfolio["start_date"]).dt.normalize()
        )
        self.portfolio["maturity_date"] = (
            pd.to_datetime(self.portfolio["maturity_date"]).dt.normalize()
)

    def loan_curve(self, term_months: int, noise=0.0005):
        """
        Игрушечная «кривая» ставок для кредитов non-fin, %
        Базовая линия 10% годовых, чуть падает c дюрацией + шум.
        """
        return 0.10 - 0.003 * term_months / 12 + self.rng.normal(0, noise)

    def dep_curve(self, term_months: int, noise=0.0005):
        """
        Кривая ставок для депозитов - чуть ниже кредитной.
        """
        return 0.08 - 0.0025 * term_months / 12 + self.rng.normal(0, noise)

    def get_credits(self):
        return self.credits

    def get_deposits(self):
        return self.deposits
    
    def get_portfolio(self):
        return self.portfolio