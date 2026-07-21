"""SSVI tests — Gatheral & Jacquier (2014) Section 4, all offline."""

import numpy as np
import pytest

from svi_lab.chain import ChainSlice
from svi_lab.ssvi import (
    atm_total_variance,
    check_static_arbitrage,
    fit_ssvi,
    phi_power_law,
    ssvi_slice_params,
    ssvi_total_variance,
)
from svi_lab.svi import calendar_violations, g_function, svi_total_variance

RHO, ETA, GAMMA = -0.55, 1.1, 0.45  # eta*(1+|rho|) = 1.705 <= 2: arb-free zone


def test_ssvi_formula_hand_computed():
    """w(k, theta) at k=0 must equal theta exactly (ATM anchoring), and a
    hand-computed off-ATM value must match to 12 digits."""
    theta = 0.04
    assert ssvi_total_variance(0.0, theta, RHO, ETA, GAMMA) == pytest.approx(theta, rel=1e-12)
    # hand computation at k = 0.1: phi = eta/(theta^g (1+theta)^(1-g))
    p = ETA / (theta**GAMMA * (1 + theta) ** (1 - GAMMA))
    expected = 0.5 * theta * (1 + RHO * p * 0.1 + np.sqrt((p * 0.1 + RHO) ** 2 + 1 - RHO**2))
    assert ssvi_total_variance(0.1, theta, RHO, ETA, GAMMA) == pytest.approx(expected, rel=1e-12)


def test_phi_power_law_monotone_decreasing():
    th = np.linspace(0.001, 2.0, 500)
    p = phi_power_law(th, ETA, GAMMA)
    assert np.all(np.diff(p) < 0)


def test_lemma_31_slice_mapping_equivalence():
    """An SSVI slice IS a raw SVI slice: eq. (4.1) and the Lemma 3.1 raw
    parameters must agree everywhere."""
    k = np.linspace(-1.5, 1.5, 301)
    for theta in (0.005, 0.04, 0.3):
        direct = ssvi_total_variance(k, theta, RHO, ETA, GAMMA)
        via_raw = svi_total_variance(k, ssvi_slice_params(theta, RHO, ETA, GAMMA))
        np.testing.assert_allclose(via_raw, direct, rtol=1e-12)


def test_bound_respected_means_butterfly_free():
    """eta*(1+|rho|) <= 2 -> g(k) >= 0 on every slice (checked through the
    SAME g-function used for raw fits, via the Lemma 3.1 mapping)."""
    k = np.linspace(-2.0, 2.0, 801)
    for theta in (0.002, 0.02, 0.2, 1.0):
        g = g_function(k, ssvi_slice_params(theta, RHO, ETA, GAMMA))
        assert np.min(g) >= -1e-9, f"butterfly arb at theta={theta}"


def test_bound_violation_can_produce_butterfly_arb():
    """Far outside the bound (eta*(1+|rho|) >> 2) the density goes negative
    somewhere — the bound is doing real work."""
    rho, eta, gamma = -0.9, 4.0, 0.45  # bound = 7.6 >> 2
    found_negative = False
    for theta in (0.5, 1.0, 2.0, 4.0):
        g = g_function(np.linspace(-3, 3, 1201), ssvi_slice_params(theta, rho, eta, gamma))
        if np.min(g) < -1e-8:
            found_negative = True
            break
    assert found_negative


def _synthetic_slices(rho=RHO, eta=ETA, gamma=GAMMA, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    slices = []
    for t, theta in [(0.05, 0.008), (0.15, 0.02), (0.3, 0.045), (0.6, 0.09)]:
        k = np.linspace(-0.4, 0.3, 40)
        w = ssvi_total_variance(k, theta, rho, eta, gamma)
        w = w + rng.normal(0, noise, size=k.size)
        iv = np.sqrt(w / t)
        slices.append(ChainSlice(expiry=f"t{t}", t_years=t, forward=100.0, k=k, w=w, iv=iv))
    return slices


def test_fit_recovers_synthetic_parameters():
    fit = fit_ssvi(_synthetic_slices(noise=1e-5, seed=3))
    assert fit.rho == pytest.approx(RHO, abs=0.03)
    assert fit.eta == pytest.approx(ETA, abs=0.08)
    assert fit.gamma == pytest.approx(GAMMA, abs=0.08)
    assert fit.rmse_global < 5e-4
    assert fit.bound <= 2.0 + 1e-12


def test_fit_enforces_bound_even_on_hostile_data():
    """Data generated OUTSIDE the arb-free zone: the fit must still come back
    with eta*(1+|rho|) <= 2 — worse fit, but never an arbitrageable surface."""
    hostile = _synthetic_slices(rho=-0.9, eta=3.5, gamma=0.4, noise=0.0)
    fit = fit_ssvi(hostile)
    assert fit.bound <= 2.0 + 1e-12
    checks = check_static_arbitrage(fit)
    assert checks["all"], checks


def test_fitted_surface_has_zero_calendar_violations():
    fit = fit_ssvi(_synthetic_slices(noise=1e-5, seed=7))
    slices = [(float(t), fit.slice_params(i)) for i, t in enumerate(fit.t_years)]
    assert calendar_violations(slices) == 0


def test_check_static_arbitrage_all_four_conditions():
    fit = fit_ssvi(_synthetic_slices(noise=1e-5, seed=1))
    checks = check_static_arbitrage(fit)
    assert checks == {
        "theta_nondecreasing": True,
        "phi_slope_bounds": True,
        "wing_slope_lt_4": True,
        "curvature_le_4": True,
        "all": True,
    }


def test_atm_theta_isotonic_projection():
    """A noisy dip in the ATM term structure is projected up, never down-crossed."""
    slices = _synthetic_slices(noise=0.0)
    # corrupt slice 2's ATM region downward to force a violation
    bad = slices[2]
    slices[2] = ChainSlice(
        expiry=bad.expiry, t_years=bad.t_years, forward=bad.forward,
        k=bad.k, w=bad.w * 0.3, iv=bad.iv,
    )
    thetas = atm_total_variance(slices)
    assert np.all(np.diff(thetas) >= 0)


def test_fit_requires_two_expiries():
    with pytest.raises(ValueError):
        fit_ssvi(_synthetic_slices()[:1])
