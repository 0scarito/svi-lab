"""Dupire local-volatility tests — Gatheral (2006), all offline."""

import numpy as np

from svi_lab.localvol import local_variance, local_vol, local_vol_surface
from svi_lab.ssvi import SSVIFit, ssvi_slice_params
from svi_lab.svi import g_function


def _make_fit(rho=-0.5, eta=1.0, gamma=0.5, thetas=None, t_years=None) -> SSVIFit:
    if t_years is None:
        t_years = np.array([0.1, 0.2, 0.3, 0.5])
    if thetas is None:
        thetas = np.array([0.01, 0.02, 0.03, 0.05])  # monotone increasing
    n = len(t_years)
    return SSVIFit(
        rho=rho, eta=eta, gamma=gamma,
        thetas=np.asarray(thetas, float), t_years=np.asarray(t_years, float),
        rmse_per_slice=np.zeros(n), rmse_global=0.0, bound=eta * (1 + abs(rho)),
    )


def test_flat_smile_linear_variance_recovers_implied_vol():
    """Flat smile (eta -> 0 so w is flat in k) with total variance linear in t
    (theta_t = c*t) has local vol == implied vol == sqrt(c) everywhere."""
    c = 0.20
    t_years = np.array([0.1, 0.2, 0.3, 0.4])
    fit = _make_fit(rho=0.0, eta=1e-7, gamma=0.5, thetas=c * t_years, t_years=t_years)
    k = np.linspace(-0.3, 0.3, 21)
    lv = local_vol(fit, k, t=0.25)
    np.testing.assert_allclose(lv, np.sqrt(c), atol=1e-3)


def test_denominator_is_the_butterfly_g_function():
    """The Dupire denominator returned by local_variance IS svi.g_function on
    the SSVI slice at theta_t — the elegant reuse claim, made a test."""
    fit = _make_fit()
    k = np.linspace(-0.5, 0.5, 41)
    _, _, g = local_variance(fit, k, t=0.2)
    theta = float(np.interp(0.2, fit.t_years, fit.thetas))
    g_direct = g_function(k, ssvi_slice_params(theta, fit.rho, fit.eta, fit.gamma))
    np.testing.assert_allclose(g, g_direct, rtol=1e-12)


def test_local_variance_nonnegative_on_arbitrage_free_fit():
    """An arbitrage-free SSVI fit (bound <= 2) has g > 0 and dw/dt > 0, so local
    variance is non-negative across the fitted region."""
    fit = _make_fit(rho=-0.6, eta=1.1, gamma=0.5)  # bound = 1.76 <= 2
    for t in (0.12, 0.2, 0.35, 0.45):
        lv, dwdt, g = local_variance(fit, np.linspace(-0.6, 0.6, 61), t)
        assert np.all(g > 0), f"g not positive at t={t}"
        assert np.all(dwdt > 0), f"calendar (dw/dt) not positive at t={t}"
        assert np.all(lv >= 0), f"negative local variance at t={t}"


def test_surface_shape_and_finiteness():
    fit = _make_fit()
    k_grid, t_grid, surf = local_vol_surface(fit, n_k=40, n_t=15)
    assert surf.shape == (15, 40)
    assert np.all(np.isfinite(surf))
    assert np.all(surf >= 0)


def test_maturity_clamped_to_fitted_range():
    """Querying outside the fitted maturities clamps rather than extrapolating
    wildly — local vol stays finite and non-negative."""
    fit = _make_fit()
    lv_below = local_vol(fit, np.array([0.0]), t=0.001)
    lv_above = local_vol(fit, np.array([0.0]), t=5.0)
    assert np.isfinite(lv_below).all() and lv_below[0] >= 0
    assert np.isfinite(lv_above).all() and lv_above[0] >= 0


def test_skew_makes_local_vol_asymmetric():
    """A negative-rho (downward-skewed) surface gives a downward-sloping local
    vol in log-moneyness: OTM puts (k<0) carry higher local vol than OTM calls."""
    fit = _make_fit(rho=-0.7, eta=1.0, gamma=0.5)
    k = np.array([-0.3, 0.3])
    lv = local_vol(fit, k, t=0.25)
    assert lv[0] > lv[1]
