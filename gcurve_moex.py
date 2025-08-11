# gcurve_moex.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

@dataclass
class MOEXParams:
    beta0: float
    beta1: float
    beta2: float
    tau: float
    g: np.ndarray = field(default_factory=lambda: np.zeros(_NB, dtype=float))

# --- default MOEX params (basis points) for 14.08.2025 ---
DEFAULT_PARAMS_BPS = MOEXParams(
    beta0=1290.50,
    beta1=268.81,
    beta2=-516.78,
    tau=0.50,
    g=np.array([-0.62, -1.72, -2.33, -2.62, -1.30, 7.69, -1.70, 0.00, 0.00], dtype=float)
)

TERMS = [0, 3, 6, 12, 24]   # 0 == o/n

# фиксированные параметры «бампов» из описания МОЕХ
_A1 = 0.0
_A2 = 0.6
_K  = 1.6
_NB = 9  # число добавочных членов

def _gen_ab():
    a = np.zeros(_NB, dtype=float)
    b = np.zeros(_NB, dtype=float)
    a[0] = _A1           # 0.0
    a[1] = _A2           # 0.6
    for i in range(2, _NB):
        # a_{i+1} = a_i + a2 * k^{i-1},  i=2..8  (индексация 1-based в формуле)
        a[i] = a[i-1] + _A2 * (_K ** (i-1))
    b[0] = _A2
    for i in range(1, _NB):
        # b_{i+1} = b_i * k
        b[i] = b[i-1] * _K
    return a, b

_A_VEC, _B_VEC = _gen_ab()

def _moex_yield_cont(t_years: float, beta0: float, beta1: float, beta2: float, tau: float, g_vec) -> float:
    """
    Непрерывно начисляемая zero-ставка G(t) модели MOEX (в долях годовых, НЕ в б.п.).
    Формула:  β0 + (β1+β2)*f1 - β2*e^{-t/τ} + Σ g_i * exp(-((t-a_i)^2)/b_i^2),
    где f1 = (1 - e^{-t/τ}) / (t/τ). Для t→0 берём предел f1→1.
    """
    t = float(t_years)
    if t <= 0:
        f1 = 1.0
        e = 1.0
    else:
        x = t / float(tau)
        e = np.exp(-x)
        f1 = (1.0 - e) / x
    ns = beta0 + (beta1 + beta2) * f1 - beta2 * e
    bumps = np.exp(-((t - _A_VEC) ** 2) / (_B_VEC ** 2))
    return float(ns + np.dot(np.asarray(g_vec, float), bumps))


class MOEXCurve:
    """
    Кривая по спецификации MOEX GCURVE: Нельсон–Зигель + 9 гауссовых добавок.
    API совпадает с твоими кривыми: r_overnight(), rate(term), step(), snapshot().
    """
    def __init__(
        self,
        t0: datetime,
        params: MOEXParams | None = None,
        base_points: dict[int, float] | None = None,   # {месяцы: ставка в долях годовых}
        phi: float = 0.98,
        sigma: dict[str, float] | None = None,         # дневные волы для AR(1) по параметрам
        seed: int | None = 42,
        tau_grid: tuple[float, float, int] = (0.2, 5.0, 80),  # перебор tau при калибровке
        from_bps: bool = False,    # если True, входные ставки/параметры заданы в б.п.
        continuous_out: bool = False,  # если True, .rate() вернёт непрерывную ставку; иначе конвертнём в простую годовую
    ):
        self.t_curr = t0
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.phi = float(phi)
        if sigma is None:
            sigma = {"beta0": 0.0006, "beta1": 0.0008, "beta2": 0.0008, "tau": 0.002, "g": 0.0005}
        self.sigma = sigma
        self.continuous_out = bool(continuous_out)

        if params is not None:
            self.params = self._convert_params(params, from_bps)
        elif base_points is not None:
            bp = {k: (v/10000.0 if from_bps else float(v)) for k, v in base_points.items()}
            self.params = self._fit_from_points(bp, tau_grid=tau_grid)
        else:
            self.params = self._convert_params(DEFAULT_PARAMS_BPS, from_bps=True)

        # долгосрочные уровни для AR(1) — фиксируем по старту
        self.mu = MOEXParams(self.params.beta0, self.params.beta1, self.params.beta2, self.params.tau, self.params.g.copy())

    # ---- калибровка по точкам: grid по tau + OLS по (β0,β1,β2,g1..g9) ----
    def _fit_from_points(self, pts: dict[int, float], tau_grid=(0.2, 5.0, 80)) -> MOEXParams:
        months = np.array(sorted(pts.keys()), dtype=float)
        y_obs = np.array([pts[int(m)] for m in months], dtype=float)  # доли годовых
        t = months / 12.0

        tau_min, tau_max, tau_n = tau_grid
        taus = np.linspace(tau_min, tau_max, tau_n)

        best = None
        for tau in taus:
            # столбцы: 1, f1, (f1 - e), G1..G9
            if np.any(t == 0.0):
                f1 = np.where(t == 0.0, 1.0, (1.0 - np.exp(-t / tau)) / (t / tau))
                e  = np.where(t == 0.0, 1.0, np.exp(-t / tau))
            else:
                e  = np.exp(-t / tau)
                f1 = (1.0 - e) / (t / tau)
            X = [np.ones_like(t), f1, (f1 - e)]
            for i in range(_NB):
                Gi = np.exp(-((t - _A_VEC[i]) ** 2) / (_B_VEC[i] ** 2))
                X.append(Gi)
            X = np.column_stack(X)
            try:
                beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
            except Exception:
                continue
            y_hat = X @ beta
            sse = float(np.sum((y_hat - y_obs) ** 2))
            if (best is None) or (sse < best[0]):
                best = (sse, beta, tau)
        if best is None:
            raise RuntimeError("MOEX fit failed")

        _, coef, tau = best
        beta0 = float(coef[0])
        beta1 = float(coef[1])
        beta2 = float(coef[2])
        g_vec = np.asarray(coef[3:3+_NB], dtype=float)
        return MOEXParams(beta0, beta1, beta2, float(tau), g_vec)

    @staticmethod
    def _convert_params(p: MOEXParams, from_bps: bool) -> MOEXParams:
        if not from_bps:
            return p
        scale = 1.0 / 10000.0
        return MOEXParams(p.beta0*scale, p.beta1*scale, p.beta2*scale, p.tau, np.asarray(p.g, float)*scale)

    # ---- API совместимый с движком ----
    def _rate_cont(self, term_months: int) -> float:
        return _moex_yield_cont(term_months/12.0, self.params.beta0, self.params.beta1, self.params.beta2, self.params.tau, self.params.g)

    def r_overnight(self) -> float:
        r = _moex_yield_cont(1.0/365.0, self.params.beta0, self.params.beta1, self.params.beta2, self.params.tau, self.params.g)
        return r if self.continuous_out else float(np.expm1(r))  # e^r - 1

    def rate(self, term_months: int) -> float:
        if term_months not in TERMS:
            raise ValueError(f"Unsupported term: {term_months}. Allowed: {TERMS}")
        r = self._rate_cont(term_months)
        return r if self.continuous_out else float(np.expm1(r))  # конвертируем «cont» → простая годовая

    def step(self, days: int = 1) -> None:
        for _ in range(days):
            # AR(1) по (β0,β1,β2)
            for k in ("beta0", "beta1", "beta2"):
                prev = getattr(self.params, k)
                mu   = getattr(self.mu, k)
                eps  = self.rng.normal(0.0, 1.0)
                newv = mu + self.phi * (prev - mu) + self.sigma[k] * eps
                setattr(self.params, k, float(newv))
            # τ > 0: моделируем в лог-пространстве
            lt_prev = np.log(self.params.tau)
            lt_mu   = np.log(self.mu.tau)
            eps = self.rng.normal(0.0, 1.0)
            self.params.tau = float(np.exp(lt_mu + self.phi * (lt_prev - lt_mu) + self.sigma["tau"] * eps))
            # g-вектор
            eps_g = self.rng.normal(0.0, 1.0, size=_NB)
            self.params.g = self.mu.g + self.phi * (self.params.g - self.mu.g) + self.sigma["g"] * eps_g
            self.t_curr += timedelta(days=1)

    def snapshot(self) -> dict:
        snap = {"date": self.t_curr}
        for m in TERMS:
            snap[m] = round(self.rate(m), 6)
        snap.update({
            "beta0": round(self.params.beta0, 6),
            "beta1": round(self.params.beta1, 6),
            "beta2": round(self.params.beta2, 6),
            "tau":   round(self.params.tau, 6),
            "g":     [round(float(x), 6) for x in np.asarray(self.params.g).tolist()],
        })
        return snap
