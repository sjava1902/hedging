# gcurve_ns.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

TERMS = [0, 3, 6, 12, 24]  # месяцы: 0 == o/n

def _ns_yield(t_years: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
    # Nelson–Siegel (спот-ставка, годовая, простая)
    if t_years <= 0:
        # асимптота при t->0: beta0 + beta1
        return float(beta0 + beta1)
    x = t_years / tau
    e = np.exp(-x)
    return float(beta0 + beta1 * e + beta2 * x * e)

@dataclass
class NSParams:
    beta0: float
    beta1: float
    beta2: float
    tau: float

class NSCurve:
    """
    Кривая Нельсона–Зигеля с дневной динамикой параметров (AR(1) вокруг 'долгосрочных' уровней).
    Имеет тот же интерфейс, что и твоя GCurve.
    """
    def __init__(
        self,
        t0: datetime,
        # можно задать сразу параметры, либо откалибровать из точек base_points
        params: NSParams | None = None,
        base_points: dict[int, float] | None = None,  # {месяцы: ставка}
        phi: float = 0.98,
        sigma: dict[str, float] | None = None,        # дневные валы по betas/tau
        seed: int | None = 42,
        tau_grid: tuple[float, float, int] = (0.2, 5.0, 80),  # сетка для калибровки tau
    ):
        self.t_curr = t0
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.phi = float(phi)
        if sigma is None:
            sigma = {"beta0": 0.0006, "beta1": 0.0008, "beta2": 0.0008, "tau": 0.002}
        self.sigma = sigma

        if params is not None:
            self.params = params
        elif base_points is not None:
            # калибруем НС из опорных точек (месяцы->ставка), делая grid по tau и OLS по beta*
            self.params = self._fit_from_points(base_points, tau_grid=tau_grid)
        else:
            raise ValueError("Provide either params or base_points")

        # долгосрочные уровни (средние) = стартовые параметры; AR(1) будет притягиваться к ним
        self.mu = NSParams(**self.params.__dict__)

    # ---- калибровка NS из точек (без внешних библиотек) ----
    def _fit_from_points(self, pts: dict[int, float], tau_grid=(0.2, 5.0, 80)) -> NSParams:
        months = np.array(sorted(pts.keys()), dtype=float)
        y_obs = np.array([pts[int(m)] for m in months], dtype=float)
        t = months / 12.0

        tau_min, tau_max, tau_n = tau_grid
        taus = np.linspace(tau_min, tau_max, tau_n)

        best = None
        for tau in taus:
            # X @ [b0,b1,b2] ~= y
            # колонки: 1, exp(-t/τ), (t/τ)*exp(-t/τ)
            x = t / tau
            e = np.exp(-x, dtype=float)
            X = np.column_stack([np.ones_like(t), e, x * e])
            # OLS
            try:
                beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
            except Exception:
                continue
            y_hat = X @ beta
            sse = float(np.sum((y_hat - y_obs) ** 2))
            if (best is None) or (sse < best[0]):
                best = (sse, beta, tau)

        if best is None:
            raise RuntimeError("NS fit failed")
        _, beta, tau = best
        return NSParams(beta0=float(beta[0]), beta1=float(beta[1]), beta2=float(beta[2]), tau=float(tau))

    # ---- API, совместимый с движком ----
    def r_overnight(self) -> float:
        return _ns_yield(1.0 / 365.0, **self.params.__dict__)

    def rate(self, term_months: int) -> float:
        if term_months not in TERMS:
            raise ValueError(f"Unsupported term: {term_months}. Allowed: {TERMS}")
        return _ns_yield(term_months / 12.0, **self.params.__dict__)

    def step(self, days: int = 1) -> None:
        # AR(1) по параметрам вокруг mu
        for _ in range(days):
            # беты
            for k in ("beta0", "beta1", "beta2"):
                r_prev = getattr(self.params, k)
                mu = getattr(self.mu, k)
                eps = self.rng.normal(0.0, 1.0)
                r_new = mu + self.phi * (r_prev - mu) + self.sigma[k] * eps
                setattr(self.params, k, float(r_new))
            # tau > 0: моделируем в лог-пространстве
            log_tau_prev = np.log(self.params.tau)
            log_tau_mu = np.log(self.mu.tau)
            eps = self.rng.normal(0.0, 1.0)
            log_tau_new = log_tau_mu + self.phi * (log_tau_prev - log_tau_mu) + self.sigma["tau"] * eps
            self.params.tau = float(np.exp(log_tau_new))
            self.t_curr += timedelta(days=1)

    def snapshot(self) -> dict:
        snap = {"date": self.t_curr}
        for m in TERMS:
            snap[m] = round(self.rate(m), 6)
        # можно добавить и параметры
        snap.update({k: round(v, 6) for k, v in self.params.__dict__.items()})
        return snap
