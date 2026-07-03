"""Raw SVI parametrization, slice fitting, and static-arbitrage diagnostics.

Sources:
- Gatheral (2004), "A parsimonious arbitrage-free implied volatility
  parametrization with application to the valuation of volatility derivatives".
- Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces",
  Quantitative Finance 14(1). The butterfly test below is their g-function
  (Theorem 2.1 / eq. 2.1): a slice is free of butterfly arbitrage iff
  g(k) >= 0 for all k (and total variance stays positive).

Raw SVI total implied variance, with k = log-moneyness ln(K/F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

Fitting note: we constrain a >= 0 (with b >= 0, |rho| < 1, sigma > 0), which
guarantees w > 0 everywhere at a small cost in fit flexibility versus the
weaker Gatheral condition a + b*sigma*sqrt(1-rho^2) >= 0. Documented trade-off
for robustness on noisy Yahoo quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


def svi_total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """w(k): raw SVI total implied variance."""
    k = np.asarray(k, dtype=float)
    d = k - p.m
    return p.a + p.b * (p.rho * d + np.sqrt(d * d + p.sigma**2))


def svi_derivatives(k: np.ndarray, p: SVIParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, w', w'') of raw SVI — analytic."""
    k = np.asarray(k, dtype=float)
    d = k - p.m
    root = np.sqrt(d * d + p.sigma**2)
    w = p.a + p.b * (p.rho * d + root)
    w1 = p.b * (p.rho + d / root)
    w2 = p.b * p.sigma**2 / root**3
    return w, w1, w2


def g_function(k: np.ndarray, p: SVIParams) -> np.ndarray:
    """Gatheral-Jacquier butterfly-arbitrage density term g(k) (eq. 2.1).

    g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    The slice is butterfly-arbitrage-free iff g(k) >= 0 wherever w(k) > 0.
    """
    k = np.asarray(k, dtype=float)
    w, w1, w2 = svi_derivatives(k, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1**2 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    return np.where(w > 0, g, -np.inf)


@dataclass(frozen=True)
class SliceFit:
    params: SVIParams
    rmse: float                 # in total-variance units
    n_quotes: int
    g_min: float                # min of g on the diagnostic grid
    butterfly_arbitrage_free: bool


def fit_svi_slice(
    k: np.ndarray,
    w_market: np.ndarray,
    weights: np.ndarray | None = None,
) -> SliceFit:
    """Least-squares fit of raw SVI to one expiry's (k, total variance) quotes.

    Multiple starts guard against the well-known local minima of raw SVI.
    """
    k = np.asarray(k, dtype=float)
    w_mkt = np.asarray(w_market, dtype=float)
    if k.size != w_mkt.size or k.size < 5:
        raise ValueError("need at least 5 (k, w) quotes to fit a slice")
    wts = np.ones_like(w_mkt) if weights is None else np.asarray(weights, dtype=float)

    w_max = float(w_mkt.max())
    k_span = float(k.max() - k.min()) or 1.0
    lb = [0.0, 1e-6, -0.999, k.min() - k_span, 1e-4]
    ub = [w_max * 2.0 + 1e-8, 10.0, 0.999, k.max() + k_span, 5.0]

    def residuals(x):
        p = SVIParams(*x)
        return (svi_total_variance(k, p) - w_mkt) * wts

    starts = [
        [max(w_mkt.min() * 0.5, 1e-8), 0.1, -0.5, 0.0, 0.1],
        [max(w_mkt.min() * 0.9, 1e-8), 0.3, -0.7, float(k[np.argmin(w_mkt)]), 0.2],
        [1e-6, 0.05, 0.0, 0.0, 0.5],
    ]
    best = None
    for x0 in starts:
        x0 = np.clip(x0, lb, ub)
        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), method="trf")
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("SVI fit failed from every start")

    params = SVIParams(*best.x)
    rmse = float(np.sqrt(np.mean((svi_total_variance(k, params) - w_mkt) ** 2)))
    k_grid = np.linspace(k.min() - 0.5 * k_span, k.max() + 0.5 * k_span, 401)
    g = g_function(k_grid, params)
    g_min = float(np.min(g))
    return SliceFit(
        params=params,
        rmse=rmse,
        n_quotes=int(k.size),
        g_min=g_min,
        butterfly_arbitrage_free=bool(g_min >= -1e-8),
    )


def calendar_violations(
    slices: list[tuple[float, SVIParams]],
    k_grid: np.ndarray | None = None,
) -> int:
    """Count grid points where total variance DECREASES with maturity.

    ``slices`` is a list of (T_years, params) sorted by T. Zero means no
    calendar-spread arbitrage on the diagnostic grid.
    """
    if len(slices) < 2:
        return 0
    grid = np.linspace(-0.5, 0.5, 101) if k_grid is None else np.asarray(k_grid)
    violations = 0
    ordered = sorted(slices, key=lambda s: s[0])
    prev = svi_total_variance(grid, ordered[0][1])
    for _, params in ordered[1:]:
        cur = svi_total_variance(grid, params)
        violations += int(np.sum(cur < prev - 1e-10))
        prev = cur
    return violations
