import numpy as np
import pytest

from svi_lab import (
    SVIParams,
    calendar_violations,
    fit_svi_slice,
    g_function,
    svi_derivatives,
    svi_total_variance,
)

BENIGN = SVIParams(a=0.02, b=0.4, rho=-0.4, m=0.05, sigma=0.2)  # Gatheral-style slice


def test_derivatives_match_finite_differences():
    """Analytic w' and w'' agree with central differences — validates g(k)."""
    k = np.linspace(-0.8, 0.8, 41)
    h = 1e-5
    w, w1, w2 = svi_derivatives(k, BENIGN)
    w_p = svi_total_variance(k + h, BENIGN)
    w_m = svi_total_variance(k - h, BENIGN)
    np.testing.assert_allclose(w1, (w_p - w_m) / (2 * h), atol=1e-6)
    np.testing.assert_allclose(w2, (w_p - 2 * w + w_m) / h**2, atol=1e-4)


def test_benign_slice_is_butterfly_free():
    k = np.linspace(-1.0, 1.0, 201)
    assert np.all(g_function(k, BENIGN) >= 0)


def test_known_arbitrageable_slice_flagged():
    """Extreme wings (b*(1+|rho|) too large) violate Roger Lee / butterfly."""
    bad = SVIParams(a=0.01, b=4.0, rho=-0.9, m=0.0, sigma=0.05)
    k = np.linspace(-1.0, 1.0, 401)
    assert np.min(g_function(k, bad)) < 0


def test_fit_recovers_synthetic_parameters():
    rng = np.random.default_rng(3)
    k = np.linspace(-0.5, 0.5, 40)
    w_true = svi_total_variance(k, BENIGN)
    w_noisy = w_true + rng.normal(0, 1e-4, size=k.size)
    fit = fit_svi_slice(k, w_noisy)
    w_fit = svi_total_variance(k, fit.params)
    np.testing.assert_allclose(w_fit, w_true, atol=5e-4)
    assert fit.rmse < 5e-4
    assert fit.butterfly_arbitrage_free


def test_fit_rejects_tiny_slices():
    with pytest.raises(ValueError):
        fit_svi_slice(np.array([0.0, 0.1]), np.array([0.02, 0.021]))


def test_calendar_violations_detects_crossing():
    near = SVIParams(a=0.05, b=0.1, rho=0.0, m=0.0, sigma=0.2)   # fat short maturity
    far = SVIParams(a=0.01, b=0.1, rho=0.0, m=0.0, sigma=0.2)    # thinner long one
    assert calendar_violations([(0.25, near), (1.0, far)]) > 0
    assert calendar_violations([(0.25, far), (1.0, near)]) == 0
