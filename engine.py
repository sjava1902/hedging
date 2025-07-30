from datetime import datetime, timedelta

from portfolio import Portfolio

class HedgeEngine:
    t0 = datetime(2016, 12, 31)  # старт моделирования
    t_curr = t0
    accumulatingAaccount = 0 # общий счёт
    portfolio = Portfolio()

    def __init__(self, portfolio):
        self.portfolio = portfolio

    
    def step(self):
        self.t_curr += timedelta(days=1)

    def get_t_start(self):
        return self.t0
    def set_t_start(self, t0):
        self.t0 = t0

    def get_t_curr(self):
        return self.t_curr
    def set_t_curr(self, t_curr):
        self.t_curr = t_curr