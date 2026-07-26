"""Dupire local volatility from the fitted SSVI surface.

Source: Gatheral (2006), "The Volatility Surface", Ch. 1. In terms of total
implied variance w(k, t) with k = log-moneyness, the Dupire local variance is

    v_L(k, t) = (∂w/∂t) / g(k, t)

where g is EXACTLY the Gatheral-Jacquier butterfly function already implemented
in :func:`svi_lab.svi.g_function`:

    g = (1 − k·w_k/(2w))² − (w_k²/4)(1/w + 1/4) + w_kk/2.

So the two arbitrage checks the lab already runs are precisely the conditions
for a valid local-vol surface: g(k) > 0 (butterfly ⇒ positive denominator) and
∂w/∂t > 0 (calendar ⇒ positive numerator). Local variance is non-negative
exactly on the arbitrage-free region — the same statement, seen a third way
(after the smile fit and the risk-neutral density).

We build w(k, t) from the fitted SSVI surface: the global (ρ, η, γ) with the
ATM total variance θ_t interpolated across the fitted expiries (linear in t;
θ_t is monotone by construction after the isotonic projection in ``fit_ssvi``,
so dθ/dt ≥ 0). ∂w/∂t is a central finite difference in t held inside the fitted
maturity range; g and its k-derivatives are analytic.
"""

from __future__ import annotations

import numpy as np

from .ssvi import SSVIFit, ssvi_slice_params, ssvi_total_variance
from .svi import g_function


def _theta_at(fit: SSVIFit, t: float) -> float:
    """ATM total variance θ_t interpolated (linear in t) across fitted expiries."""
    return float(np.interp(t, fit.t_years, fit.thetas))


def _w_at(fit: SSVIFit, k: np.ndarray, t: float) -> np.ndarray:
    theta = _theta_at(fit, t)
    return ssvi_total_variance(k, theta, fit.rho, fit.eta, fit.gamma)


def local_variance(
    fit: SSVIFit, k: np.ndarray, t: float, dt: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dupire local variance v_L(k, t) = (∂w/∂t) / g(k).

    Returns ``(local_variance, dw_dt, g)`` so callers can inspect the numerator
    and denominator separately. ``t`` is clamped to the fitted maturity range,
    and the finite-difference stencil for ∂w/∂t is kept inside it.
    """
    k = np.asarray(k, dtype=float)
    t_min, t_max = float(fit.t_years.min()), float(fit.t_years.max())
    t = float(np.clip(t, t_min, t_max))
    if dt is None:
        dt = 0.01 * (t_max - t_min) or 1e-3
    tp = min(t + dt, t_max)
    tm = max(t - dt, t_min)
    if tp == tm:  # single-expiry or degenerate range
        tp, tm = t + 1e-4, max(t - 1e-4, 1e-6)
    dw_dt = (_w_at(fit, k, tp) - _w_at(fit, k, tm)) / (tp - tm)

    theta = _theta_at(fit, t)
    g = g_function(k, ssvi_slice_params(theta, fit.rho, fit.eta, fit.gamma))
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = dw_dt / g
    return lv, dw_dt, g


def local_vol(fit: SSVIFit, k: np.ndarray, t: float) -> np.ndarray:
    """Local volatility √max(v_L, 0) at log-moneyness ``k`` and maturity ``t``."""
    lv, _, _ = local_variance(fit, k, t)
    return np.sqrt(np.maximum(lv, 0.0))


def local_vol_surface(
    fit: SSVIFit,
    k_grid: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    n_k: int = 60,
    n_t: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local-vol surface over (k, t). Returns ``(k_grid, t_grid, surface)`` where
    ``surface`` has shape ``(len(t_grid), len(k_grid))``."""
    t_min, t_max = float(fit.t_years.min()), float(fit.t_years.max())
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.5, n_k)
    if t_grid is None:
        t_grid = np.linspace(t_min, t_max, n_t)
    surface = np.empty((t_grid.size, k_grid.size))
    for i, t in enumerate(t_grid):
        surface[i] = local_vol(fit, k_grid, float(t))
    return k_grid, t_grid, surface
