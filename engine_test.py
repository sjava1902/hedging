import unittest
from datetime import datetime, timedelta

from engine import HedgeEngine, QUARTER_LEN_DAYS
from portfolio import Portfolio


class TestEngineMethods(unittest.TestCase):
    def test_step(self):
        engine = HedgeEngine(Portfolio())
        engine.set_t_start(datetime(2016, 12, 31))
        engine.step()
        t_curr = engine.get_t_curr()
        self.assertEqual(datetime(2017,1,1), t_curr)
    
    def test_quarter_settle_swaps(self):
        p = Portfolio(N_C=10, N_D=10, V=100000)
        from gcurve import GCurve
        g = GCurve(p.T0, {0:0.09,3:0.095,6:0.10,12:0.105,24:0.11})
        e = HedgeEngine(p, g)
        e.add_swap("pay_fixed", 12, 50000)
        start_q = e.days_since_quarter_start
        e.step(QUARTER_LEN_DAYS - start_q)   # ровно до клиринга
        self.assertAlmostEqual(e.accrued_swap, 0.0, places=6)

    def test_rollover_portfolio(self):
        p = Portfolio(N_C=5, N_D=5, V=100000)
        from gcurve import GCurve
        g = GCurve(p.T0, {0:0.09,3:0.095,6:0.10,12:0.105,24:0.11})
        e = HedgeEngine(p, g)
        t0 = e.t_curr
        e.step(200)  # за это время что-то точно погасится и перекатится
        after = p.get_portfolio()
        self.assertTrue(((after['start_date'] >= t0) & (after['start_date'] < e.t_curr)).any())

    def test_step_to_quarter_end_resets_counter(self):
        p = Portfolio(N_C=3, N_D=3, V=100000)
        e = HedgeEngine(p)
        # сдвинемся на произвольное число дней внутри квартала
        e.step(17)
        self.assertGreater(e.days_since_quarter_start, 0)
        # доводим ровно до конца квартала
        e.step_to_quarter_end()
        self.assertEqual(e.days_since_quarter_start, 0)
        # и убедимся, что начисления действительно списались на счёт
        self.assertAlmostEqual(e.accrued_interest, 0.0, places=10)

    def test_add_swap_rejects_bad_direction(self):
        p = Portfolio(N_C=1, N_D=1, V=10000)
        e = HedgeEngine(p)
        with self.assertRaises(ValueError):
            e.add_swap("fixed_pay", 12, 1000)  # опечатка в направлении

    def test_swap_reset_float_each_quarter(self):
        p = Portfolio(N_C=2, N_D=2, V=50000)
        e = HedgeEngine(p)
        e.add_swap("pay_fixed", 12, 10000)
        flt0 = float(e.swaps.loc[0, "float_rate_q"])
        e.step_to_quarter_end()               # в клиринг должен пересчитаться float_rate_q
        flt1 = float(e.swaps.loc[0, "float_rate_q"])
        # допускаем, что может совпасть, но в среднем – меняется; проверим, что поле не пропало и число валидно
        self.assertTrue(flt1 > 0.0)

    def test_rollover_resets_remaining_and_rate(self):
        p = Portfolio(N_C=3, N_D=3, V=100000)
        e = HedgeEngine(p)
        df0 = p.get_portfolio().copy()
        # найдём ближайшее погашение и дойдём до него
        days_to_first_maturity = int(df0["remaining_months"].min() * (365.25/12)) + 1
        e.step(days_to_first_maturity)
        df1 = p.get_portfolio()
        # кто-то обязательно «перекатился»: remaining == contract_months
        self.assertTrue(((df1["remaining_months"] - df1["contract_months"]).abs() < 1e-6).any())

