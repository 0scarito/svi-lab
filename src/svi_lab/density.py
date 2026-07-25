"""Risk-neutral density extraction from a fitted SVI smile.

Breeden & Litzenberger (1978) show the risk-neutral density of the underlying
is the (undiscounted) second strike-derivative of the call price. Gatheral
(2006, "The Volatility Surface: A Practitioner's Guide") rewrites it in closed
form straight from an SVI total-variance slice, so the density falls out of the
SAME g-function that flags butterfly arbitrage — no numerical differentiation of
noisy quotes:

    p(k) = g(k) / sqrt(2*pi*w(k)) * exp(-d_minus(k)^2 / 2)

is the density of the log-return k = ln(S_T/F) in log-strike, with

    d_minus(k) = -k / sqrt(w(k)) - sqrt(w(k)) / 2,

w(k) the SVI total implied variance and g(k) the Gatheral-Jacquier butterfly
term (``svi.g_function``, reused verbatim). p integrates to one over the real
line by construction; it is a *valid* (non-negative) probability density iff
g(k) >= 0 everywhere — i.e. iff the slice is free of butterfly arbitrage. An
arbitrage-free smile and a valid risk-neutral density are the same statement,
which is why this module reuses the arbitrage machinery rather than adding any.

Sources:
- Breeden & Litzenberger (1978), "Prices of State-Contingent Claims Implicit in
  Option Prices", Journal of Business 51(4), 621-651.
- Gatheral (2006), "The Volatility Surface: A Practitioner's Guide", Wiley —
  the SVI closed-form density above, in the notation d_+/d_-.
"""

from __future__ import annotations

import numpy as np

from .ssvi import SSVIFit
from .svi import SliceFit, SVIParams, g_function, svi_total_variance


def risk_neutral_density(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Breeden-Litzenberger risk-neutral density of the log-return, in log-strike.

    Gatheral's SVI closed form (see module docstring):
        p(k) = g(k) / sqrt(2*pi*w(k)) * exp(-d_minus(k)^2 / 2),
        d_minus(k) = -k/sqrt(w) - sqrt(w)/2.

    ``g`` is ``svi.g_function`` verbatim, so the density is exact up to float
    error. Where g(k) < 0 the returned "density" is negative — that is exactly
    the butterfly arbitrage the g-function detects, surfaced rather than hidden.
    Works for any ``SVIParams``: raw slice fits, and SSVI slices via
    ``ssvi.ssvi_slice_params`` / ``SSVIFit.slice_params``.
    """
    k = np.asarray(k, dtype=float)
    w = svi_total_variance(k, params)
    g = g_function(k, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_w = np.sqrt(w)
        d_minus = -k / sqrt_w - sqrt_w / 2.0
        p = g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus**2)
    return np.where(w > 0, p, 0.0)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Composite trapezoid integral of y over x (numpy-version agnostic)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def density_stats(k_grid: np.ndarray, params: SVIParams) -> tuple[float, float, float]:
    """Total mass, mean and variance of the risk-neutral density over ``k_grid``.

    Integrates ``risk_neutral_density`` by the trapezoid rule. ``total_mass``
    should be ~1 on a wide, fine grid for an arbitrage-free slice — that is the
    Breeden-Litzenberger normalization, and drifting from 1 is a sign the fit is
    ragged or the grid clips the tails. Mean and variance are the first two
    moments of the log-return, normalized by the realized mass so they stay
    meaningful even when a finite grid trims a sliver of tail.
    """
    k = np.asarray(k_grid, dtype=float)
    p = risk_neutral_density(k, params)
    total_mass = _trapezoid(p, k)
    if total_mass <= 0.0:
        return total_mass, float("nan"), float("nan")
    mean = _trapezoid(k * p, k) / total_mass
    second = _trapezoid(k * k * p, k) / total_mass
    variance = second - mean * mean
    return total_mass, mean, variance


def slice_density(
    k: np.ndarray,
    fit: SliceFit | SSVIFit,
    i: int | None = None,
) -> np.ndarray:
    """Risk-neutral density for a fitted object, dispatching on its type.

    - ``SliceFit`` (raw per-expiry fit): pass the fit; ``i`` is ignored.
    - ``SSVIFit`` (whole surface): pass the fit and the slice index ``i`` — the
      SSVI slice is mapped to raw-SVI params via Lemma 3.1, then reuses the same
      closed form.
    """
    if isinstance(fit, SSVIFit):
        if i is None:
            raise ValueError("SSVIFit is a surface; pass the slice index i")
        params = fit.slice_params(i)
    else:
        params = fit.params
    return risk_neutral_density(k, params)
