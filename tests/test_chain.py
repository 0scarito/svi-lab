import numpy as np
import pandas as pd

from svi_lab import parity_forward, slice_from_frames


def _frames(forward=105.0, n=15, iv=0.22, t=0.25):
    """Synthetic calls/puts whose mids satisfy parity around ``forward``."""
    strikes = np.linspace(80, 130, n)
    intrinsic_c = np.maximum(forward - strikes, 0) + 2.0   # crude convex proxy
    calls = pd.DataFrame({
        "strike": strikes,
        "bid": intrinsic_c - 0.05,
        "ask": intrinsic_c + 0.05,
        "impliedVolatility": np.full(n, iv) + 0.06 * ((strikes - forward) / forward) ** 2,
    })
    # parity: P = C - (F - K)  (r ~ 0)
    p_mid = intrinsic_c - (forward - strikes)
    puts = pd.DataFrame({
        "strike": strikes,
        "bid": p_mid - 0.05,
        "ask": p_mid + 0.05,
        "impliedVolatility": np.full(n, iv) + 0.06 * ((strikes - forward) / forward) ** 2,
    })
    return calls, puts


def test_parity_forward_recovers_input():
    calls, puts = _frames(forward=105.0)
    f = parity_forward(calls, puts)
    assert abs(f - 105.0) < 1.5


def test_slice_from_frames_builds_otm_slice():
    calls, puts = _frames(forward=105.0)
    s = slice_from_frames("2026-12-18", 0.25, calls, puts)
    assert s is not None
    assert s.k.size >= 8
    # OTM construction: k spans both sides of 0
    assert s.k.min() < 0 < s.k.max()
    # total variance is iv^2 * T
    np.testing.assert_allclose(s.w, s.iv**2 * 0.25)


def test_slice_returns_none_when_too_few_quotes():
    calls, puts = _frames(n=5)
    assert slice_from_frames("2026-12-18", 0.25, calls, puts) is None
