# gcurve.py
import numpy as np
from datetime import datetime, timedelta

TERMS = [0, 3, 6, 12, 24]

class GCurve:
    def __init__(self, t0: datetime, base: dict, phi: float = 0.97, sigma: dict | None = None, seed: int = 42):
        if set(base.keys()) != set(TERMS):
            raise ValueError(f"base must have keys {TERMS}, got {sorted(base.keys())}")
        self.mu = {m: float(base[m]) for m in TERMS}
        self.phi = float(phi)
        self.t_curr = t0
        self.rng = np.random.default_rng(seed)
        self.current = dict(base)
        if sigma is None:
            sigma = {0: 0.0008, 3: 0.0006, 6: 0.0006, 12: 0.0005, 24: 0.0005}
        if set(sigma.keys()) != set(TERMS):
            raise ValueError(f"sigma must have keys {TERMS}")
        self.sigma = {m: float(sigma[m]) for m in TERMS}

    def rate_overnight(self) -> float:
        return self.current[0]

    def rate(self, term_months: int) -> float:
        if term_months not in TERMS:
            raise ValueError(f"Unsupported term: {term_months}. Allowed: {TERMS}")
        return float(self.current[term_months])

    def step(self, days: int = 1) -> None:
        for _ in range(days):
            for m in TERMS:
                mu = self.mu[m]
                rate_prev = float(self.current[m])
                eps = self.rng.normal(0.0, 1.0)
                r_new = mu + self.phi * (rate_prev - mu) + self.sigma[m] * eps
                self.current[m] = max(r_new, 0.0)
            self.t_curr += timedelta(days=1)

    def snapshot(self) -> dict:
        snap = {m: round(float(self.current[m]), 6) for m in TERMS}
        snap['date'] = self.t_curr
        return snap
    
