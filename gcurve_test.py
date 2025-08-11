import unittest
from datetime import datetime
import numpy as np
from gcurve_ns import NSCurve
from gcurve import GCurve
from gcurve_ns import NSCurve

class TestNSCurve(unittest.TestCase):
    def test_fit_and_rates(self):
        t0 = datetime(2016,12,31)
        base = {0: 0.09, 3: 0.095, 6: 0.10, 12: 0.105, 24: 0.11}
        ns = NSCurve(t0, base_points=base)
        snap = ns.snapshot()
        # MSE на узлах должен быть небольшим
        y_hat = np.array([snap[m] for m in [0,3,6,12,24]])
        y = np.array([base[m] for m in [0,3,6,12,24]])
        mse = float(np.mean((y_hat - y)**2))
        self.assertLess(mse, 1e-4)

    def test_step_moves_but_not_explodes(self):
        t0 = datetime(2016,12,31)
        base = {0: 0.09, 3: 0.095, 6: 0.10, 12: 0.105, 24: 0.11}
        ns = NSCurve(t0, base_points=base, seed=123)
        r0 = ns.rate(12)
        ns.step(30)
        r1 = ns.rate(12)
        self.assertNotEqual(r0, r1)
        self.assertGreater(ns.params.tau, 0.0)


TERMS_FULL = [1,2,3,4,6,9,12,18,24,36]  # мес (на сетке будет линейная интерполяция для GCurve)

def _disc(y, T_years):  # дисконт-фактор из спот-ставки y
    return float(np.exp(-y * T_years))

def _gcurve_rate_linear(c: GCurve, m: int) -> float:
    knots = [0, 3, 6, 12, 24]
    r = {k: c.rate(k) for k in knots}
    if m in r:
        return r[m]
    if m < 0:
        raise ValueError(m)
    if m > knots[-1]:
        lo, hi = knots[-2], knots[-1]     # 12 -> 24
        slope = (r[hi] - r[lo]) / (hi - lo)
        return r[hi] + slope * (m - hi)   # линейная экстраполяция до 36
    lo = max(k for k in knots if k < m)
    hi = min(k for k in knots if k > m)
    w = (m - lo) / (hi - lo)
    return (1 - w) * r[lo] + w * r[hi]

class TestCurveQuality(unittest.TestCase):
    def setUp(self):
        self.t0 = datetime(2016,12,31)
        self.base = {0:0.09,3:0.095,6:0.10,12:0.105,24:0.11}

    def test_noarb_discount_monotone(self):
        for Curve in (lambda: GCurve(self.t0, self.base),
                      lambda: NSCurve(self.t0, base_points=self.base)):
            c = Curve()
            # пропускаем 0м; проверяем D(T) убывает
            Ds = [_disc((c.rate(m)), m/12.0) for m in [3,6,12,24]]
            self.assertTrue(all(Ds[i+1] <= Ds[i] + 1e-10 for i in range(len(Ds)-1)))

    def test_smoothness_over_terms(self):
        # NS обычно гладче линейной интерполяции по узлам (вторая разность меньше)
        gc = GCurve(self.t0, self.base)
        ns = NSCurve(self.t0, base_points=self.base)
        y_gc = np.array([_gcurve_rate_linear(gc, m) for m in TERMS_FULL])
        y_ns = np.array([ns.rate(m) for m in TERMS_FULL])
        dd_gc = np.diff(y_gc, n=2)
        dd_ns = np.diff(y_ns, n=2)
        self.assertLess(np.sum(dd_ns**2), np.sum(dd_gc**2) * 1.05)  # небольшое послабление

    def test_walkforward_1day_rmse(self):
        # синтетика: крутанём 120 дней, меряем RMSE прогноза на 1 день
        steps = 120
        gc = GCurve(self.t0, self.base, seed=123)
        ns = NSCurve(self.t0, base_points=self.base, seed=123)
        # прогноз на 1 день: "ставка завтра ~= ставка сегодня" (naive) → сравним, чья траектория менее зубастая
        errs_gc, errs_ns = [], []
        prev_gc = gc.rate(12); prev_ns = ns.rate(12)
        for _ in range(steps):
            gc.step(1); ns.step(1)
            errs_gc.append((gc.rate(12) - prev_gc)**2)
            errs_ns.append((ns.rate(12) - prev_ns)**2)
            prev_gc = gc.rate(12); prev_ns = ns.rate(12)
        rmse_gc = float(np.sqrt(np.mean(errs_gc)))
        rmse_ns = float(np.sqrt(np.mean(errs_ns)))
        # у NS часто плавнее динамика → RMSE шаговых изменений не больше
        self.assertLessEqual(rmse_ns, rmse_gc * 1.10)
