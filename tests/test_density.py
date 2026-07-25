"""Risk-neutral density tests — Breeden-Litzenberger via Gatheral's SVI closed
form, all offline and seeded. The density reuses the SAME g-function that flags
butterfly arbitrage, so 'arbitrage-free slice' and 'valid density integrating
to one' are two names for one fact — the tests below pin down both directions.
"""

import numpy as np
import pytest

from svi_lab import SVIParams
from svi_lab.chain import ChainSlice
from svi_lab.density import density_stats, risk_neutral_density, slice_density
from svi_lab.ssvi import fit_ssvi, ssvi_slice_params, ssvi_total_variance
from svi_lab.svi import fit_svi_slice, g_function, svi_total_variance

# A realistic, benign slice: small total variance, g >= 0 everywhere, so its
# density is a proper probability density that all but vanishes past |k| = 3.
BENIGN = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.15)

# SSVI arb-free zone: eta*(1+|rho|) = 1.705 <= 2.
RHO, ETA, GAMMA = -0.55, 1.1, 0.45


def test_density_integrates_to_one():
    """Breeden-Litzenberger normalization: an arbitrage-free slice integrates to
    ~1 over a wide log-strike grid."""
    k = np.linspace(-3.0, 3.0, 4001)
    mass, _, _ = density_stats(k, BENIGN)
    assert mass == pytest.approx(1.0, abs=1e-2)


def test_density_nonnegative_where_g_positive():
    """Where g(k) >= 0 the density is non-negative — the two are the same claim
    (p = g * positive factor)."""
    k = np.linspace(-3.0, 3.0, 2001)
    assert np.all(g_function(k, BENIGN) >= 0.0)  # precondition: benign slice
    assert np.all(risk_neutral_density(k, BENIGN) >= -1e-12)


def test_negative_density_where_g_negative():
    """The flip side: a butterfly-arbitrageable slice yields a locally NEGATIVE
    density — surfaced, not hidden (same slice test_svi flags via g_min < 0)."""
    bad = SVIParams(a=0.01, b=4.0, rho=-0.9, m=0.0, sigma=0.05)
    k = np.linspace(-1.0, 1.0, 801)
    assert np.min(g_function(k, bad)) < 0
    assert np.min(risk_neutral_density(k, bad)) < 0


def test_lognormal_limit():
    """Flat smile (b -> 0) => constant total variance w0 = sigma_bs^2 * T, and
    g(k) -> 1, so the density collapses to the Black-Scholes lognormal: the
    normal density of the log-return with mean -w0/2 and variance w0."""
    w0 = 0.04  # e.g. sigma_bs = 0.2, T = 1
    flat = SVIParams(a=w0, b=1e-6, rho=-0.3, m=0.0, sigma=0.3)
    k = np.linspace(-1.5, 1.5, 601)
    p = risk_neutral_density(k, flat)
    normal = np.exp(-((k - (-w0 / 2.0)) ** 2) / (2.0 * w0)) / np.sqrt(2.0 * np.pi * w0)
    np.testing.assert_allclose(p, normal, atol=1e-3)
    # and the moments match the analytic lognormal ones
    mass, mean, var = density_stats(np.linspace(-4.0, 4.0, 8001), flat)
    assert mass == pytest.approx(1.0, abs=1e-3)
    assert mean == pytest.approx(-w0 / 2.0, abs=1e-3)
    assert var == pytest.approx(w0, abs=1e-3)


def test_density_stats_positive_variance_finite_mean():
    k = np.linspace(-3.0, 3.0, 4001)
    _, mean, var = density_stats(k, BENIGN)
    assert np.isfinite(mean)
    assert var > 0.0


def _synthetic_slices(rho=RHO, eta=ETA, gamma=GAMMA, noise=0.0, seed=0):
    """SSVI-generated chain slices (mirrors the SSVI test fixtures)."""
    rng = np.random.default_rng(seed)
    slices = []
    for t, theta in [(0.05, 0.008), (0.15, 0.02), (0.3, 0.045), (0.6, 0.09)]:
        k = np.linspace(-0.4, 0.3, 40)
        w = ssvi_total_variance(k, theta, rho, eta, gamma) + rng.normal(0, noise, size=k.size)
        iv = np.sqrt(w / t)
        slices.append(ChainSlice(expiry=f"t{t}", t_years=t, forward=100.0, k=k, w=w, iv=iv))
    return slices


def test_slice_density_helper_raw_and_ssvi():
    """The convenience helper dispatches: SliceFit uses its params, SSVIFit maps
    slice i to raw-SVI params — both agree with the underlying closed form."""
    k = np.linspace(-1.0, 1.0, 301)

    # raw SliceFit
    kk = np.linspace(-0.5, 0.5, 40)
    fit = fit_svi_slice(kk, svi_total_variance(kk, BENIGN))
    np.testing.assert_allclose(
        slice_density(k, fit), risk_neutral_density(k, fit.params), rtol=1e-12
    )

    # SSVI surface fit, slice index i
    ssvi = fit_ssvi(_synthetic_slices(noise=1e-5, seed=3))
    for i in range(len(ssvi.t_years)):
        np.testing.assert_allclose(
            slice_density(k, ssvi, i),
            risk_neutral_density(k, ssvi.slice_params(i)),
            rtol=1e-12,
        )
    with pytest.raises(ValueError):
        slice_density(k, ssvi)  # surface needs an index


def test_ssvi_slice_density_integrates_to_one():
    """SSVI is arbitrage-free by construction, so every slice's density is a
    valid probability density integrating to ~1 — verified through the shared
    density machinery, no SSVI-specific code."""
    k = np.linspace(-3.0, 3.0, 4001)
    for theta in (0.008, 0.02, 0.045, 0.09):
        params = ssvi_slice_params(theta, RHO, ETA, GAMMA)
        assert np.all(risk_neutral_density(k, params) >= -1e-12)
        mass, _, _ = density_stats(k, params)
        assert mass == pytest.approx(1.0, abs=1e-2)
